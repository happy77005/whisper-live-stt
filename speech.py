import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import scipy.io.wavfile as wav

model = whisper.load_model("base")  # use "small" or "medium" for better accuracy

samplerate = 16000  # Whisper expects 16kHz
block_duration = 5  # seconds

print("🎤 Speak now (press Ctrl+C to stop)...")

try:
    while True:
        print("\n🔴 Listening...")
        audio = sd.rec(int(block_duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        # Save to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, samplerate, (audio * 32767).astype(np.int16))
            temp_path = f.name

        print("🔍 Transcribing...")
        result = model.transcribe(temp_path)
        print("📝 " + result["text"])

        os.remove(temp_path)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
