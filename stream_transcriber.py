"""
Streaming transcription tool with real-time/VAD chunking, optional floating UI, and logging.

Features
- Real-time audio capture from microphone, system (loopback), or both
- VAD-based chunking (~1–2 seconds) using WebRTC VAD
- Transcription via faster-whisper with GPU support when available
- Background-friendly, optional always-on-top Tkinter floating window
- Continuous console updates and timestamped .txt logging
- Pause/Resume control (UI button) and CLI options
- Optional GPT summarization stub (disabled unless configured)

Dependencies (install as needed)
- numpy
- soundcard (system audio loopback + microphone)
- webrtcvad
- faster-whisper
- tkinter (included with many Python distributions; on Linux may need tk)
- scipy (optional: for resampling; we fallback if missing)
- openai (optional: for summarization)

Examples
- Microphone only, console only:
  python stream_transcriber.py --mode mic --model tiny --no-ui

- System audio (meeting), floating window:
  python stream_transcriber.py --mode system --model small --ui

- Both mic + system, GPU if present, log to file:
  python stream_transcriber.py --mode both --model small --log transcripts/meeting.txt

Notes
- On Linux, system capture depends on PulseAudio/PipeWire support via soundcard.
- VAD operates on 16 kHz mono, 10/20/30 ms frames; we use 30 ms by default.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import queue
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Third-party imports with graceful fallbacks/logging
try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required: pip install numpy") from exc

try:
    import soundcard as sc
except Exception as exc:  # pragma: no cover
    sc = None

try:
    import webrtcvad
except Exception as exc:  # pragma: no cover
    webrtcvad = None

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # pragma: no cover
    WhisperModel = None

try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
except Exception:  # pragma: no cover
    tk = None
    ScrolledText = None

# Optional resampling support
try:
    from scipy.signal import resample_poly  # type: ignore
except Exception:
    resample_poly = None


@dataclass
class AudioChunk:
    samples: np.ndarray  # mono float32 in [-1.0, 1.0]
    start_time: float    # epoch seconds
    end_time: float      # epoch seconds


class GracefulKiller:
    def __init__(self) -> None:
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def is_stopped(self) -> bool:
        return self._stop.is_set()


def dbfs(signal: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(signal))) + 1e-12
    return 20.0 * math.log10(rms)


def normalize_audio(signal: np.ndarray, target_dbfs: float = -20.0, limit: float = 0.99) -> np.ndarray:
    current = dbfs(signal)
    gain_db = target_dbfs - current
    gain = math.pow(10.0, gain_db / 20.0)
    normalized = signal * gain
    clipped = np.clip(normalized, -limit, limit)
    return clipped.astype(np.float32, copy=False)


def downmix_to_mono(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return frames.astype(np.float32, copy=False)
    return np.mean(frames, axis=1).astype(np.float32, copy=False)


def resample_audio(frames: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return frames.astype(np.float32, copy=False)
    if resample_poly is not None:
        # Use high-quality polyphase resampling
        g = math.gcd(src_sr, dst_sr)
        up = dst_sr // g
        down = src_sr // g
        return resample_poly(frames, up, down).astype(np.float32, copy=False)
    # Fallback: simple linear interpolation (lower quality)
    duration_sec = frames.shape[0] / float(src_sr)
    new_len = int(round(duration_sec * dst_sr))
    if new_len <= 1:
        return frames.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, duration_sec, num=frames.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration_sec, num=new_len, endpoint=False)
    interp = np.interp(x_new, x_old, frames)
    return interp.astype(np.float32, copy=False)


def float_to_int16_pcm(frames: np.ndarray) -> bytes:
    # Expect float32 in [-1, 1]
    scaled = np.clip(frames, -1.0, 1.0)
    scaled = (scaled * 32767.0).astype(np.int16)
    return scaled.tobytes()


class AudioCapture(threading.Thread):
    def __init__(
        self,
        out_queue: queue.Queue[np.ndarray],
        mode: str = "mic",  # mic | system | both
        samplerate: int = 16000,
        frame_ms: int = 30,
        normalize: bool = True,
        killer: Optional[GracefulKiller] = None,
        mic_name: Optional[str] = None,
        speaker_name: Optional[str] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.mode = mode
        self.samplerate = int(samplerate)
        self.frame_samples = int(self.samplerate * frame_ms / 1000)
        self.normalize = normalize
        self.killer = killer or GracefulKiller()
        self.mic_name = mic_name
        self.speaker_name = speaker_name
        self._pause = threading.Event()
        self._pause.clear()

        if sc is None:
            raise RuntimeError("soundcard not available. Install with: pip install soundcard")

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def toggle(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
        else:
            self._pause.set()

    def _open_mic(self):
        if self.mic_name:
            mics = sc.all_microphones()
            mic = next((m for m in mics if self.mic_name in m.name), None)
            if mic is None:
                print(f"[AudioCapture] Microphone '{self.mic_name}' not found; using default.")
                mic = sc.default_microphone()
        else:
            mic = sc.default_microphone()
        return mic.recorder(samplerate=self.samplerate)

    def _open_speaker(self):
        if self.speaker_name:
            spks = sc.all_speakers()
            spk = next((s for s in spks if self.speaker_name in s.name), None)
            if spk is None:
                print(f"[AudioCapture] Speaker '{self.speaker_name}' not found; using default.")
                spk = sc.default_speaker()
        else:
            spk = sc.default_speaker()
        return spk.recorder(samplerate=self.samplerate)

    def run(self) -> None:
        mic_rec = None
        spk_rec = None
        try:
            if self.mode in ("mic", "both"):
                mic_rec = self._open_mic()
            if self.mode in ("system", "both"):
                spk_rec = self._open_speaker()

            # Use context managers if opened
            if mic_rec and spk_rec:
                with mic_rec as mic, spk_rec as spk:
                    self._loop_both(mic, spk)
            elif mic_rec:
                with mic_rec as mic:
                    self._loop_single(mic)
            elif spk_rec:
                with spk_rec as spk:
                    self._loop_single(spk)
            else:
                print("[AudioCapture] No audio sources available.")
        except Exception as e:
            print(f"[AudioCapture] Error: {e}")

    def _loop_single(self, rec) -> None:
        while not self.killer.is_stopped():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            frames = rec.record(numframes=self.frame_samples)
            frames = downmix_to_mono(frames)
            if self.normalize:
                frames = normalize_audio(frames)
            self._safe_put(frames)

    def _loop_both(self, mic, spk) -> None:
        while not self.killer.is_stopped():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            mic_frames = downmix_to_mono(mic.record(numframes=self.frame_samples))
            spk_frames = downmix_to_mono(spk.record(numframes=self.frame_samples))
            # Align and mix (simple average)
            mixed_len = min(mic_frames.shape[0], spk_frames.shape[0])
            mixed = 0.5 * (mic_frames[:mixed_len] + spk_frames[:mixed_len])
            if self.normalize:
                mixed = normalize_audio(mixed)
            self._safe_put(mixed)

    def _safe_put(self, frames: np.ndarray) -> None:
        try:
            self.out_queue.put(frames, timeout=0.25)
        except queue.Full:
            # Drop oldest if queue is full to keep latency low
            try:
                _ = self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(frames)
            except queue.Full:
                pass


class VadSegmenter(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue[np.ndarray],
        out_queue: queue.Queue[AudioChunk],
        samplerate: int = 16000,
        frame_ms: int = 30,
        vad_aggressiveness: int = 2,
        min_chunk_ms: int = 700,
        max_chunk_ms: int = 2000,
        killer: Optional[GracefulKiller] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.samplerate = samplerate
        self.frame_samples = int(self.samplerate * frame_ms / 1000)
        self.min_frames = max(1, int(min_chunk_ms / frame_ms))
        self.max_frames = max(1, int(max_chunk_ms / frame_ms))
        self.killer = killer or GracefulKiller()
        if webrtcvad is None:
            raise RuntimeError("webrtcvad not available. Install with: pip install webrtcvad")
        self.vad = webrtcvad.Vad(int(vad_aggressiveness))

        self._pause = threading.Event()
        self._pause.clear()

        # State
        self._buffer: List[np.ndarray] = []
        self._buffer_start_time: Optional[float] = None

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def toggle(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
        else:
            self._pause.set()

    def run(self) -> None:
        while not self.killer.is_stopped():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            try:
                frame = self.in_queue.get(timeout=0.2)  # float32 mono
            except queue.Empty:
                continue

            # Convert to 16-bit PCM bytes for VAD decision
            pcm = float_to_int16_pcm(frame)
            is_speech = self.vad.is_speech(pcm, self.samplerate)
            now = time.time()

            if is_speech:
                if not self._buffer:
                    self._buffer_start_time = now
                self._buffer.append(frame)

                if len(self._buffer) >= self.max_frames:
                    self._flush(now)
            else:
                # Non-speech frame
                if self._buffer and len(self._buffer) >= self.min_frames:
                    self._flush(now)
                else:
                    # drop buffer if too short; reset
                    self._buffer.clear()
                    self._buffer_start_time = None

    def _flush(self, now: float) -> None:
        if not self._buffer:
            return
        samples = np.concatenate(self._buffer, axis=0)
        start = self._buffer_start_time or now
        chunk = AudioChunk(samples=samples, start_time=start, end_time=now)
        self._buffer.clear()
        self._buffer_start_time = None
        self._safe_put(chunk)

    def _safe_put(self, chunk: AudioChunk) -> None:
        try:
            self.out_queue.put(chunk, timeout=0.25)
        except queue.Full:
            try:
                _ = self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(chunk)
            except queue.Full:
                pass


@dataclass
class TranscriptPiece:
    text: str
    start_time: float
    end_time: float


class Transcriber(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue[AudioChunk],
        out_queue: queue.Queue[TranscriptPiece],
        model_size: str = "tiny",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        killer: Optional[GracefulKiller] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.word_timestamps = word_timestamps
        self.killer = killer or GracefulKiller()
        self._pause = threading.Event()
        self._pause.clear()

        if WhisperModel is None:
            raise RuntimeError("faster-whisper not available. Install with: pip install faster-whisper")

        # Resolve device
        if self.device is None:
            # Simple heuristic: try CUDA via environment hint
            self.device = os.environ.get("WHISPER_DEVICE", "cuda")
        if self.compute_type is None:
            self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16" if self.device == "cuda" else "int8")

        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def toggle(self) -> None:
        if self._pause.is_set():
            self._pause.clear()
        else:
            self._pause.set()

    def run(self) -> None:
        while not self.killer.is_stopped():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            try:
                chunk = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            audio = chunk.samples.astype(np.float32, copy=False)
            try:
                # Transcribe chunk; avoid conditioning on previous text to keep low latency
                result = self.model.transcribe(
                    audio=audio,
                    beam_size=1,
                    language=self.language,
                    vad_filter=False,
                    word_timestamps=self.word_timestamps,
                    condition_on_previous_text=False,
                )
                # Handle both generator or (segments, info)
                segments_iter = None
                info = None
                if isinstance(result, tuple) and len(result) == 2:
                    segments_iter, info = result
                else:
                    segments_iter = result

                full_text_parts: List[str] = []
                for seg in segments_iter:
                    seg_text = getattr(seg, "text", "")
                    if seg_text:
                        full_text_parts.append(seg_text)
                final_text = " ".join(part.strip() for part in full_text_parts).strip()

                if final_text:
                    piece = TranscriptPiece(text=final_text, start_time=chunk.start_time, end_time=chunk.end_time)
                    self._safe_put(piece)
            except Exception as e:
                print(f"[Transcriber] Error during transcription: {e}")

    def _safe_put(self, piece: TranscriptPiece) -> None:
        try:
            self.out_queue.put(piece, timeout=0.25)
        except queue.Full:
            try:
                _ = self.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.out_queue.put_nowait(piece)
            except queue.Full:
                pass


class TranscriptionLogger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def write(self, piece: TranscriptPiece) -> None:
        if not self.path:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(piece.end_time))
        line = f"[{ts}] {piece.text}\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)


class Summarizer:
    def __init__(self, enabled: bool = False, model: str = "gpt-4o-mini", max_chars: int = 6000) -> None:
        self.enabled = enabled
        self.model = model
        self.max_chars = max_chars
        self._client = None
        if self.enabled:
            try:
                import openai  # type: ignore
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    openai.api_key = api_key
                    self._client = openai
                else:
                    print("[Summarizer] OPENAI_API_KEY not set; summarization disabled.")
                    self.enabled = False
            except Exception as e:
                print(f"[Summarizer] OpenAI client not available: {e}")
                self.enabled = False

    def summarize(self, text: str) -> Optional[str]:
        if not self.enabled or not self._client:
            return None
        prompt = (
            "Summarize the following meeting transcript into key points, decisions, and action items "
            "with speaker-agnostic bullets. Keep it concise.\n\n" + text[-self.max_chars :]
        )
        try:
            # Basic completion using Chat Completions API if present
            # Users may need to adjust depending on openai SDK version
            completion = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return completion["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Summarizer] Error: {e}")
            return None


class FloatingUI(threading.Thread):
    def __init__(self, in_queue: queue.Queue[TranscriptPiece], killer: GracefulKiller, on_toggle_pause) -> None:
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.killer = killer
        self.on_toggle_pause = on_toggle_pause

        if tk is None or ScrolledText is None:
            raise RuntimeError("Tkinter is not available for UI. Install system Tk libraries.")

    def run(self) -> None:
        root = tk.Tk()
        root.title("Live Transcription")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.92)
        except Exception:
            pass
        root.geometry("560x320")

        text = ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 12))
        text.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X)
        pause_btn = tk.Button(controls, text="Pause/Resume", command=self.on_toggle_pause)
        pause_btn.pack(side=tk.LEFT)

        def poll_queue():
            try:
                while True:
                    piece = self.in_queue.get_nowait()
                    self._append_text(text, piece)
            except queue.Empty:
                pass
            if not self.killer.is_stopped():
                root.after(100, poll_queue)
            else:
                root.quit()

        root.after(100, poll_queue)
        root.protocol("WM_DELETE_WINDOW", self.killer.stop)
        root.mainloop()

    def _append_text(self, widget: ScrolledText, piece: TranscriptPiece) -> None:
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, piece.text + "\n")
        # transient highlight for recent text
        try:
            start_index = f"end-{len(piece.text) + 1}c"
            end_index = "end-1c"
            widget.tag_add("new", start_index, end_index)
            widget.tag_config("new", background="#e6f7ff")
            widget.see(tk.END)
            # remove highlight after a short delay
            widget.after(1200, lambda: widget.tag_delete("new"))
        except Exception:
            pass
        widget.configure(state=tk.DISABLED)


def format_console_piece(piece: TranscriptPiece) -> str:
    ts = time.strftime("%H:%M:%S", time.localtime(piece.end_time))
    return f"[{ts}] {piece.text}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Real-time streaming transcription with VAD and faster-whisper")
    parser.add_argument("--mode", choices=["mic", "system", "both"], default="mic", help="Audio source: microphone, system loopback, or both")
    parser.add_argument("--model", default="tiny", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--device", default=None, help="Device for inference: cuda/cpu")
    parser.add_argument("--compute-type", dest="compute_type", default=None, help="Compute precision (e.g., int8_float16, int8, float16, float32)")
    parser.add_argument("--language", default=None, help="Language code (auto if not provided)")
    parser.add_argument("--frame-ms", type=int, default=30, help="Frame duration in ms for VAD (10/20/30)")
    parser.add_argument("--min-chunk-ms", type=int, default=900, help="Minimum chunk length before flush")
    parser.add_argument("--max-chunk-ms", type=int, default=1800, help="Maximum chunk length before forced flush")
    parser.add_argument("--vad", type=int, default=2, help="VAD aggressiveness (0-3)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable audio normalization")
    parser.add_argument("--ui", dest="ui", action="store_true", help="Show floating window UI")
    parser.add_argument("--no-ui", dest="ui", action="store_false", help="Disable UI; console only")
    parser.set_defaults(ui=False)
    parser.add_argument("--log", default=None, help="Path to append transcript log (e.g., transcripts/meeting.txt)")
    parser.add_argument("--word-timestamps", action="store_true", help="Enable word-level timestamps (slower)")
    parser.add_argument("--mic-name", default=None, help="Substring to match microphone device name")
    parser.add_argument("--speaker-name", default=None, help="Substring to match speaker device name")
    parser.add_argument("--summarize", action="store_true", help="Enable periodic GPT summarization (requires OPENAI_API_KEY)")
    parser.add_argument("--summarize-interval", type=int, default=600, help="Summarization interval in seconds")

    args = parser.parse_args(argv)

    if sc is None:
        print("soundcard is required: pip install soundcard")
        return 2
    if webrtcvad is None:
        print("webrtcvad is required: pip install webrtcvad")
        return 2
    if WhisperModel is None:
        print("faster-whisper is required: pip install faster-whisper")
        return 2

    print(
        f"Starting streaming transcription: mode={args.mode}, model={args.model}, ui={'on' if args.ui else 'off'}"
    )

    killer = GracefulKiller()

    # Queues
    audio_frames_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    vad_chunks_q: queue.Queue[AudioChunk] = queue.Queue(maxsize=32)
    transcript_q: queue.Queue[TranscriptPiece] = queue.Queue(maxsize=64)

    # Components
    capture = AudioCapture(
        out_queue=audio_frames_q,
        mode=args.mode,
        samplerate=16000,
        frame_ms=int(args.frame_ms),
        normalize=not args.no_normalize,
        killer=killer,
        mic_name=args.mic_name,
        speaker_name=args.speaker_name,
    )

    segmenter = VadSegmenter(
        in_queue=audio_frames_q,
        out_queue=vad_chunks_q,
        samplerate=16000,
        frame_ms=int(args.frame_ms),
        vad_aggressiveness=int(args.vad),
        min_chunk_ms=int(args.min_chunk_ms),
        max_chunk_ms=int(args.max_chunk_ms),
        killer=killer,
    )

    transcriber = Transcriber(
        in_queue=vad_chunks_q,
        out_queue=transcript_q,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        word_timestamps=bool(args.word_timestamps),
        killer=killer,
    )

    logger = TranscriptionLogger(path=args.log)

    # Optional UI
    ui_thread: Optional[FloatingUI] = None
    if args.ui and tk is not None and ScrolledText is not None:
        ui_thread = FloatingUI(
            in_queue=transcript_q,
            killer=killer,
            on_toggle_pause=lambda: (capture.toggle(), segmenter.toggle(), transcriber.toggle()),
        )

    # Summarization support
    summarizer = Summarizer(enabled=bool(args.summarize))
    last_summarize_ts = time.time()
    rolling_transcript: List[str] = []

    # Start threads
    capture.start()
    segmenter.start()
    transcriber.start()
    if ui_thread is not None:
        ui_thread.start()

    try:
        while not killer.is_stopped():
            try:
                piece = transcript_q.get(timeout=0.25)
            except queue.Empty:
                # periodic summarization check
                if summarizer.enabled and (time.time() - last_summarize_ts) >= int(args.summarize_interval):
                    last_summarize_ts = time.time()
                    summary = summarizer.summarize("\n".join(rolling_transcript))
                    if summary:
                        # Print and optionally log summary
                        print("\n=== Summary ===\n" + summary + "\n=== End Summary ===\n")
                        if logger.path:
                            with open(logger.path, "a", encoding="utf-8") as f:
                                f.write("\n=== Summary ===\n" + summary + "\n=== End Summary ===\n\n")
                        if ui_thread is None:
                            # console-only: create a TranscriptPiece-like output for visual continuity
                            ts = time.strftime("%H:%M:%S", time.localtime(time.time()))
                            print(f"[{ts}] (summary updated)")
                    # reset rolling transcript after summary
                    rolling_transcript.clear()
                continue

            console_line = format_console_piece(piece)
            print(console_line)
            logger.write(piece)
            rolling_transcript.append(piece.text)

    except KeyboardInterrupt:
        pass
    finally:
        killer.stop()
        time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
