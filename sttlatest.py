import pyaudio
import wave
import numpy as np
import torch
import keyboard
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHza
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "live_input.wav"

# Initialize PyAudio
audio = pyaudio.PyAudio()

def record_audio():
    print("Recording... Hold SPACE to continue recording.")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    while keyboard.is_pressed("space"):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    print("Recording stopped.")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe():
    import soundfile as sf
    import librosa

    print("Transcribing...")
    # Load and preprocess audio
    speech, sr = librosa.load(WAVE_OUTPUT_FILENAME, sr=16000)
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Transcribe
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print("📝 Transcription:", transcription)

print("🎤 Press and hold SPACE to record. Release to transcribe.")
try:
    while True:
        if keyboard.is_pressed("space"):
            record_audio()
            transcribe()
except KeyboardInterrupt:
    print("\nStopped by user.")
    audio.terminate()
