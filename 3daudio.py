import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
from pydub import AudioSegment
import numpy as np
import os

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to apply ping-pong effect
def create_ping_pong_effect(samples, fs, chunk_duration):
    chunk_size = int(fs * chunk_duration)  # Number of samples per chunk
    left_channel = []
    right_channel = []

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        fade = np.linspace(0, 1, len(chunk))  # Smooth transition using linear fade

        if (i // chunk_size) % 2 == 0:
            # Left channel active, right channel fades in
            left_channel.extend(chunk * fade)
            right_channel.extend(chunk * (1 - fade))
        else:
            # Right channel active, left channel fades in
            left_channel.extend(chunk * (1 - fade))
            right_channel.extend(chunk * fade)

    # Pad with zeros if channels are of unequal length
    min_length = min(len(left_channel), len(right_channel))
    left_channel = left_channel[:min_length]
    right_channel = right_channel[:min_length]

    # Combine into stereo
    stereo_audio = np.stack((left_channel, right_channel), axis=-1)
    return stereo_audio

# File processing function
def process_audio(file_path):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1)  # Convert to mono for processing
        fs = audio.frame_rate
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize audio
        samples = samples / np.max(np.abs(samples))

        # Extract vocals (250 Hz to 6000 Hz)
        vocals = bandpass_filter(samples, 250, 6000, fs)

        # Remove vocals from the original audio to isolate non-vocal parts
        non_vocals = samples - vocals

        # Set chunk duration for the ping-pong effect (e.g., 1 second)
        chunk_duration = 2.0  # Adjust this value for slower or faster alternation

        # Apply ping-pong effect to non-vocal parts
        stereo_non_vocals = create_ping_pong_effect(non_vocals, fs, chunk_duration)

        # Add vocals equally to both channels
        stereo_vocals = np.stack((vocals, vocals), axis=-1)

        # Combine vocals and non-vocals
        stereo_audio = stereo_non_vocals + stereo_vocals

        # Normalize the stereo audio to prevent clipping
        stereo_audio = (stereo_audio / np.max(np.abs(stereo_audio)) * 32767).astype(np.int16)

        # Save the output as a WAV file
        output_path = os.path.splitext(file_path)[0] + "_Smooth3D_TWS.wav"
        write(output_path, fs, stereo_audio)
        messagebox.showinfo("Success", f"Processed file saved as: {output_path}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI for file input
def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.aac;*.ogg")]
    )
    if file_path:
        process_audio(file_path)

# Main tkinter window
root = tk.Tk()
root.title("Smooth 3D TWS Audio Enhancer")

label = tk.Label(root, text="Select an audio file to process:")
label.pack(pady=10)

select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(pady=10)

root.mainloop()
