import os
import re
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Directory with converted .wav files
data_dir = r"C:\Users\91971\Desktop\gunshot_data_updated"

# Step 1: Extract Metadata
pattern = r"(\w+)_([0-9]+)m_(\w+)\.wav"
data_info = [
    {"filename": file, "weapon": match.group(1), "distance": int(match.group(2)), "position": match.group(3)}
    for file in os.listdir(data_dir) if (match := re.match(pattern, file))
]
df = pd.DataFrame(data_info)

# Step 2: EDA
print("Dataset Information:")
print(df.info())
print("\nClass Distribution:")
print(df['weapon'].value_counts())

# Plot class distribution
plt.figure(figsize=(12, 6))
sns.countplot(x="weapon", data=df, order=df['weapon'].value_counts().index, palette="viridis")
plt.title("Weapon Type Distribution")
plt.xticks(rotation=45)
plt.show()