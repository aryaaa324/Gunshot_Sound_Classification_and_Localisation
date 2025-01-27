import os
import re
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory with converted .wav files
converted_dir = r"C:\Users\91971\Desktop\gunshot_data_updated"

# Extract metadata from filenames
pattern = r"(\w+)_([0-9]+)m_(\w+)\.wav"
data_info = [
    {"filename": file, "weapon": match.group(1), "distance": int(match.group(2)), "position": match.group(3)}
    for file in os.listdir(converted_dir) if (match := re.match(pattern, file))
]

# Create a DataFrame for better visualization
df = pd.DataFrame(data_info)
print("Dataset Information:\n", df)

# Function to plot Mel Spectrogram
def plot_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    S_dB = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {os.path.basename(file_path)}")
    plt.tight_layout()
    plt.show()

# Visualize a sample file
if not df.empty:
    sample_file = os.path.join(converted_dir, df.iloc[0]['filename'])
    print(f"Visualizing Mel Spectrogram for file: {sample_file}")
    plot_mel_spectrogram(sample_file)
