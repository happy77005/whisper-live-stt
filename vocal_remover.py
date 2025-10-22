import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import soundfile as sf

# Create a Tkinter root window (it won't be shown)
root = tk.Tk()
root.withdraw()  # Hides the root window

# Ask the user to select an audio file
file_path = filedialog.askopenfilename(
    title="Select an Audio File",
    filetypes=[("Audio Files", "*.mp3;*.wav;*.flac;*.ogg")]
)

# Check if the user selected a file
if file_path:
    # Load the audio file
    audio, sr = librosa.load(file_path, mono=False)

    # Subtract the stereo channels to remove center-panned vocals
    instrumental = audio[0] - audio[1]

    # Generate the output file path
    output_path = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("WAV Files", "*.wav")],
        title="Save the Output File"
    )

    # Save the instrumental audio if the user provided an output path
    if output_path:
        sf.write(output_path, instrumental, sr)
        print(f"Instrumental audio saved to: {output_path}")
else:
    print("No file selected.")
