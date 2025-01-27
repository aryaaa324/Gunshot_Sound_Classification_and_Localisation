import os
import re  # Added this import for regular expressions
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Data Directory
data_dir = r"C:\Users\91971\Desktop\gunshot_data_updated"

# Fixed number of frames for spectrograms
fixed_length = 128

# Function to extract features (Mel Spectrogram)
def extract_features(file_path, n_mels=128, fmax=8000):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    return librosa.power_to_db(mel_spec, ref=np.max)

# Function to pad or truncate spectrograms
def pad_or_truncate(spectrogram, max_length=fixed_length):
    if spectrogram.shape[1] > max_length:
        return spectrogram[:, :max_length]
    elif spectrogram.shape[1] < max_length:
        pad_width = max_length - spectrogram.shape[1]
        return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    return spectrogram

# Extract Metadata
pattern = r"(\w+)_([0-9]+)m_(\w+)\.wav"
data_info = [
    {"filename": file, "label": match.group(1)}
    for file in os.listdir(data_dir) if (match := re.match(pattern, file))
]
df = pd.DataFrame(data_info)

# Encode Labels
le = LabelEncoder()
df["encoded_label"] = le.fit_transform(df["label"])

# Prepare Dataset for Training
X = []
y = []

print("Extracting features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    file_path = os.path.join(data_dir, row["filename"])
    try:
        features = extract_features(file_path)
        features = pad_or_truncate(features)  # Ensure uniform shape
        X.append(features)
        y.append(row["encoded_label"])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Convert to NumPy Arrays
X = np.array(X)
y = np.array(y)

# Reshape for CNN (Add channel dimension)
X = X[..., np.newaxis]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# One-Hot Encode Labels
y_train = to_categorical(y_train, num_classes=len(le.classes_))
y_test = to_categorical(y_test, num_classes=len(le.classes_))

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, fixed_length, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate Model
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save Model
model.save("gun_type_classifier.h5")
print("Model saved as 'gun_type_classifier.h5'.")

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(le.classes_)), le.classes_, rotation=90)
plt.yticks(np.arange(len(le.classes_)), le.classes_)
plt.tight_layout()
plt.show()
