import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Path to your dataset
data_dir = "C:/Users/91971/Desktop/gunshot_data_updated"

# List all files in the dataset directory
audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

# Plot for each audio file
for audio_file in audio_files:
    file_path = os.path.join(data_dir, audio_file)
    
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)  # Load at native sampling rate

    # Compute time axis for amplitude
    time = np.linspace(0, len(y) / sr, len(y))
    
    # Compute frequency spectrum
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft), 1 / sr)
    magnitude = np.abs(fft)

    # Plot the amplitude and frequency
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot amplitude
    ax[0].plot(time, y, color='blue')
    ax[0].set_title(f"Amplitude Waveform: {audio_file}")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    
    # Plot frequency
    ax[1].plot(freqs[:len(freqs)//2], magnitude[:len(magnitude)//2], color='green')  # Positive frequencies
    ax[1].set_title(f"Frequency Spectrum: {audio_file}")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude")
    
    plt.tight_layout()
    plt.show()
