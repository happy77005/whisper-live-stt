import os
import tkinter as tk
from tkinter import filedialog, messagebox
from spleeter.separator import Separator

class BGMExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("BGM Extractor")
        self.root.geometry("400x250")

        self.label = tk.Label(root, text="Drop an audio file (MP3, M4A, FLAC, WAV)", font=("Arial", 12))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Upload Audio", command=self.upload_file, font=("Arial", 12))
        self.upload_btn.pack(pady=10)

        self.process_btn = tk.Button(root, text="Extract BGM", command=self.process_audio, font=("Arial", 12), state=tk.DISABLED)
        self.process_btn.pack(pady=10)

        self.filepath = None

    def upload_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[
            ("Audio Files", "*.mp3 *.m4a *.flac *.wav"),
            ("All Files", "*.*")
        ])
        if self.filepath:
            messagebox.showinfo("File Selected", f"Loaded: {os.path.basename(self.filepath)}")
            self.process_btn.config(state=tk.NORMAL)

    def process_audio(self):
        if not self.filepath:
            messagebox.showerror("Error", "No file selected!")
            return

        try:
            # Use Spleeter for separating vocals and music
            separator = Separator('spleeter:2stems')  # 2stems separates vocals and accompaniment (BGM)
            
            # Process the audio file
            separator.separate_to_file(self.filepath, os.path.dirname(self.filepath))
            
            # Success message
            messagebox.showinfo("Success", f"BGM Extracted and saved as:\n{os.path.dirname(self.filepath)}")

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

# Run the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = BGMExtractor(root)
    root.mainloop()
