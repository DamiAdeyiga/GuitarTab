#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model training module for chord recognition.
Creates a CRNN model for audio chord classification.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

# Define paths and parameters
PROCESSED_DIR = "data/processed/IDMT_CHORDS"
TRAIN_DIR = os.path.join(PROCESSED_DIR, "Training")
TEST_DIR = os.path.join(PROCESSED_DIR, "Test")
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
batch_size = 32

def get_chords():
    """Auto-detect chord classes with validation"""
    chords = []
    for d in os.listdir(TRAIN_DIR):
        dir_path = os.path.join(TRAIN_DIR, d)
        if os.path.isdir(dir_path) and len(os.listdir(dir_path)) > 0:
            chords.append(d)
    return sorted(chords)

def get_file_paths(directory, chords):
    """Get all file paths and labels for a directory"""
    files = []
    labels = []
    for chord in chords:
        chord_dir = os.path.join(directory, chord)
        if os.path.exists(chord_dir):
            for file in os.listdir(chord_dir):
                if file.endswith('.wav'):
                    files.append(os.path.join(chord_dir, file))
                    labels.append(chord)
    return files, labels

def encode_labels(labels):
    """Encode string labels to one-hot vectors"""
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    # Convert to categorical (one-hot)
    categorical_labels = tf.keras.utils.to_categorical(encoded_labels)
    return encoder, categorical_labels

def load_and_preprocess_audio(file_path, label):
    """Load audio file and compute mel spectrogram"""
    # Read the file
    audio = tf.io.read_file(file_path)
    
    # Decode WAV file to get waveform
    waveform, _ = tf.audio.decode_wav(audio, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    
    # Standardize length
    waveform = tf.cond(
        tf.shape(waveform)[0] >= 16000,
        lambda: waveform[:16000],
        lambda: tf.pad(waveform, [[0, 16000 - tf.shape(waveform)[0]]])
    )
    
    # Compute spectrogram
    spectrogram = tf.signal.stft(
        waveform, 
        frame_length=255, 
        frame_step=128
    )
    spectrogram = tf.abs(spectrogram)
    
    # Convert to mel spectrogram
    mel_spectrogram = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=tf.shape(spectrogram)[0],
        sample_rate=16000,
        lower_edge_hertz=0,
        upper_edge_hertz=8000
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_spectrogram, 1)
    
    # Convert to dB scale
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # Resize and normalize
    mel_spectrogram = tf.image.resize(
        tf.expand_dims(mel_spectrogram, -1), 
        [128, 128]
    )
    
    # Normalize
    mel_spectrogram = (mel_spectrogram - tf.reduce_mean(mel_spectrogram)) / (tf.math.reduce_std(mel_spectrogram) + 1e-5)
    
    return mel_spectrogram, label

def create_tf_datasets(train_files, train_labels, test_files, test_labels, batch_size, num_classes):
    """Create TensorFlow datasets for training and validation"""
    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_dataset = train_dataset.map(
        load_and_preprocess_audio,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    validation_dataset = validation_dataset.map(
        load_and_preprocess_audio,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, validation_dataset

def create_crnn_model(input_shape, num_classes):
    """Create a Convolutional Recurrent Neural Network model"""
    model = tf.keras.Sequential([
        # Enhanced convolutional layers
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        
        # Improved RNN section
        tf.keras.layers.Reshape((-1, 128)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
        
        # Enhanced dense layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Add precision and recall metrics
    precision = tf.keras.metrics.Precision(name='precision')
    recall = tf.keras.metrics.Recall(name='recall')
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy', precision, recall]
    )
    return model

def plot_history(history):
    """Plot the training history metrics"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    precision = history.history['precision']
    val_precision = history.history['val_precision']

    recall = history.history['recall']
    val_recall = history.history['val_recall']

    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, precision, label='Training Precision')
    plt.plot(epochs_range, val_precision, label='Validation Precision')
    plt.legend(loc='lower right')
    plt.title('Precision')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, recall, label='Training Recall')
    plt.plot(epochs_range, val_recall, label='Validation Recall')
    plt.legend(loc='lower right')
    plt.title('Recall')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

def plot_confusion_matrix(model, dataset, label_encoder):
    """Plot the confusion matrix for the model predictions"""
    y_true = []
    y_pred = []

    for spectrogram, label in dataset:
        predictions = model.predict(spectrogram)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(label.numpy(), axis=1)
        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.show()

def print_classification_report(model, dataset, label_encoder):
    """Print and save the classification report"""
    y_true = []
    y_pred = []

    for spectrogram, label in dataset:
        predictions = model.predict(spectrogram)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(label.numpy(), axis=1)
        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)

    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(report)
    
    # Also save the report to a file
    with open(os.path.join(MODEL_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    # Get available chords
    chords = get_chords()
    num_classes = len(chords)
    print(f"Training model for {num_classes} chord classes: {chords}")
    
    # Get file paths and labels
    print("Loading training and test data...")
    train_files, train_labels = get_file_paths(TRAIN_DIR, chords)
    test_files, test_labels = get_file_paths(TEST_DIR, chords)
    
    # Encode labels
    label_encoder, train_labels_cat = encode_labels(train_labels)
    _, test_labels_cat = encode_labels(test_labels)
    
    # Create datasets
    print("Creating TensorFlow datasets...")
    train_dataset, validation_dataset = create_tf_datasets(
        train_files, train_labels_cat, 
        test_files, test_labels_cat,
        batch_size=batch_size,
        num_classes=num_classes
    )
    
    # Create model
    print("Building CRNN model...")
    model = create_crnn_model(input_shape=(128, 128, 1), num_classes=num_classes)
    model.summary()
    
    # Compute class weights
    print("Computing class weights...")
    class_counts = Counter(train_labels)
    print("Class distribution:", class_counts)
    
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
    print("Class weights:", class_weight_dict)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, 'chord_model_best.h5'), save_best_only=True)
    ]
    
    # Train the model
    print("Starting model training...")
    history = model.fit(
        train_dataset,
        epochs=50,
        validation_data=validation_dataset,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save the model and label encoder
    model.save(os.path.join(MODEL_DIR, 'chord_model_final.h5'))
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Plot training history
    plot_history(history)
    
    # Evaluate the model
    print("Evaluating model performance...")
    plot_confusion_matrix(model, validation_dataset, label_encoder)
    print_classification_report(model, validation_dataset, label_encoder)
    
    print(f"Training complete. Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()