import os
import shutil
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (classification_report, log_loss, 
                            roc_auc_score, confusion_matrix,
                            precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize as sk_resize
from datetime import datetime

# Configuration
MODEL_PATH = "model_V3.h5"
CLASSES_FILE = "classes.txt"
GENRES_DATA_DIR = "genres_data"
CLASSES_SOURCE_DIR = "classes"
TARGET_SHAPE = (150, 150)
EPOCHS = 30
BATCH_SIZE = 32
SAMPLE_RATE = 22050

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def print_header(text):
    print("\n" + "="*50)
    print(f" {text.upper()} ")
    print("="*50)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def move_new_data():
    print_header("data migration process")
    if not os.path.exists(CLASSES_SOURCE_DIR):
        print("No new classes directory found - skipping data migration")
        return

    total_files = 0
    for class_name in os.listdir(CLASSES_SOURCE_DIR):
        src_dir = os.path.join(CLASSES_SOURCE_DIR, class_name)
        if os.path.isdir(src_dir):
            dest_dir = os.path.join(GENRES_DATA_DIR, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            class_files = os.listdir(src_dir)
            print(f"\nProcessing class '{class_name}' with {len(class_files)} files")
            
            moved_count = 0
            for filename in class_files:
                src_path = os.path.join(src_dir, filename)
                dest_path = os.path.join(dest_dir, filename)
                
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_dir)
                    moved_count += 1
            
            print(f"  Moved {moved_count} new files to {dest_dir}")
            total_files += moved_count
            
            shutil.rmtree(src_dir)
    
    print(f"\nTotal new files added: {total_files}")

def get_current_classes():
    if not os.path.exists(GENRES_DATA_DIR):
        raise FileNotFoundError(f"Genres data directory not found: {GENRES_DATA_DIR}")
    
    classes = sorted([d for d in os.listdir(GENRES_DATA_DIR) 
                     if os.path.isdir(os.path.join(GENRES_DATA_DIR, d))])
    
    if not classes:
        raise ValueError("No classes found in genres_data directory")
    
    return classes

def build_enhanced_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Conv2D(256, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                          tf.keras.metrics.AUC(name='auc')])
    return model

def audio_to_melspectrogram(audio, sr):
    try:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel = sk_resize(mel, TARGET_SHAPE, anti_aliasing=True)
        return mel.astype(np.float32)[..., np.newaxis]
    except Exception as e:
        print(f"Audio processing error: {str(e)}")
        return None

def load_and_preprocess(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        chunks = []
        chunk_size = SAMPLE_RATE * 4
        hop_size = SAMPLE_RATE * 2
        
        for i in range(0, len(audio)-chunk_size+1, hop_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
            mel = audio_to_melspectrogram(chunk, SAMPLE_RATE)
            if mel is not None:
                chunks.append(mel)
        
        return chunks
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return []

def load_dataset():
    print_header("dataset loading")
    classes = get_current_classes()
    class_counts = {cls: 0 for cls in classes}
    
    X, y = [], []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(GENRES_DATA_DIR, class_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.wav', '.mp3'))]
        
        print(f"\nProcessing {class_name} ({len(files)} files)")
        processed_files = 0
        
        for filename in files:
            file_path = os.path.join(class_dir, filename)
            chunks = load_and_preprocess(file_path)
            if chunks:
                X.extend(chunks)
                y.extend([class_idx]*len(chunks))
                processed_files += 1
                class_counts[class_name] += len(chunks)
        
        print(f"  Successfully processed {processed_files}/{len(files)} files")
    
    print("\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} samples")
    
    return np.array(X), to_categorical(y, num_classes=len(classes))

def main():
    tf.config.run_functions_eagerly(True)
    move_new_data()
    current_classes = get_current_classes()
    num_classes = len(current_classes)
    
    print_header("model initialization")
    if os.path.exists(MODEL_PATH):
        print("Found existing model - loading for fine-tuning")
        # Load model without optimizer
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        if os.path.exists(CLASSES_FILE):
            with open(CLASSES_FILE, 'r') as f:
                old_classes = [line.strip() for line in f]
            
            if set(old_classes) != set(current_classes):
                print("New classes detected - modifying model architecture")
                # Preserve all layers except last
                x = model.layers[-2].output
                # Add new classification layer
                new_output = Dense(num_classes, activation='softmax')(x)
                # Create new model
                model = Model(inputs=model.input, outputs=new_output)
        
        # Recompile the model regardless of whether new classes are detected or not
        model.compile(optimizer=Adam(learning_rate=0.00001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'precision', 'recall', 'auc'])
        print("Model compiled")
    else:
        print("No existing model found - building new model")
        model = build_enhanced_model((*TARGET_SHAPE, 1), num_classes)
    
    # Save current class list
    with open(CLASSES_FILE, 'w') as f:
        f.write('\n'.join(current_classes))
    
    # Data loading
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training
    print_header("model training")
    print(f"Training on {len(X_train)} samples | Validating on {len(X_test)} samples")
    print(f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"model_{timestamp}.h5"
    model.save(save_path)
    print(f"\nModel saved as: {save_path}")
    
    # Evaluation
    print_header("model evaluation")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nComprehensive Evaluation Metrics:")
    print(f"Classification Accuracy: {np.mean(y_true == y_pred_classes):.2%}")
    print(f"Logarithmic Loss: {log_loss(y_test, y_pred):.4f}")
    print(f"Macro AUC Score: {roc_auc_score(y_test, y_pred, multi_class='ovr', average='macro'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred_classes, average='macro'):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_classes, average='macro'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred_classes, average='macro'):.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=current_classes))
    
    plot_confusion_matrix(y_true, y_pred_classes, current_classes)
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    print("\nTraining Summary:")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Model saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()

