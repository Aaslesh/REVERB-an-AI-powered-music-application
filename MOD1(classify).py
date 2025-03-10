import os
# Suppress TensorFlow logs before importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only errors will be shown
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize
import shutil

# Configuration
MODEL_PATH = "model_V3.h5"
CLASSES_FILE = "classes.txt"
ROOT_FOLDER = r"../majour2"
UNCLASSIFIED_FOLDER = os.path.join(ROOT_FOLDER, "songs", "unclassified")
CLASSIFIED_FOLDER = os.path.join(ROOT_FOLDER, "songs", "classified")

# Load model and classes
model = tf.keras.models.load_model(MODEL_PATH)

try:
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    # Fallback to default classes if file not found
    classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Audio chunking parameters
    chunk_duration = 4   # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) /
                  (chunk_samples - overlap_samples)) + 1)

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        
        # If the chunk is too short, pad it
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

def model_prediction(X_test):
    y_pred = model.predict(X_test, verbose=0)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    return unique_elements[np.argmax(counts)]

def process_songs_folder():
    # Ensure the classified folder exists
    os.makedirs(CLASSIFIED_FOLDER, exist_ok=True)
    
    # Gather valid audio files
    audio_files = [f for f in os.listdir(UNCLASSIFIED_FOLDER)
                   if os.path.isfile(os.path.join(UNCLASSIFIED_FOLDER, f))
                   and f.lower().endswith(('.mp3', '.wav'))]
    print(f"\n\nTotal files loaded: {len(audio_files)}\n\n")
    
    processed_files = 0
    class_summary = {cls: 0 for cls in classes}
    
    for filename in audio_files:
        file_path = os.path.join(UNCLASSIFIED_FOLDER, filename)
        try:
            # Preprocess and predict
            X_test = load_and_preprocess_data(file_path)
            if len(X_test) == 0:
                continue  # silently skip if no valid audio chunks
                
            c_index = model_prediction(X_test)
            predicted_class = classes[c_index]
            
            # Create the target folder for the class
            class_folder = os.path.join(CLASSIFIED_FOLDER, predicted_class)
            os.makedirs(class_folder, exist_ok=True)
            
            # Move file to its class folder
            dest_path = os.path.join(class_folder, filename)
            shutil.move(file_path, dest_path)
            
            print(f"\nClassified {filename} as {predicted_class}\n")
            processed_files += 1
            class_summary[predicted_class] += 1
                
        except Exception:
            # Any error is silently skipped
            continue

    # Final summary
    if processed_files:
        print(f"\n\nProcessed {processed_files} file(s).\n\nSummary:\n")
        for cls, count in class_summary.items():
            print(f"{cls}: {count} file(s)\n")
        print("\n\n")
    else:
        print("\n\nNo audio files processed.\n\n")

# Run processing
process_songs_folder()
