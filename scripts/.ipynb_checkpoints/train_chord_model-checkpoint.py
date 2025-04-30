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

# Import functions from our modules
from src.data_preprocessing import get_file_paths, encode_labels, create_tf_datasets

# Define paths and parameters
DATA_DIR = "data/processed/IDMT_CHORDS"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Test")
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