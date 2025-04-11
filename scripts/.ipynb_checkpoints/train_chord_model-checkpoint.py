import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

from src.data_preprocessing import get_file_paths, encode_labels, create_tf_datasets
from src.models import create_crnn_model

# Define paths and parameters
DATA_DIR = r"C:\Users\User\Documents\GitHub\GuitarTab\Guitar_Chords_V2"

TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Test')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

chords = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
batch_size = 16
num_classes = len(chords)

# Load and encode labels
train_files, train_labels = get_file_paths(TRAIN_DIR, chords)
test_files, test_labels = get_file_paths(TEST_DIR, chords)

label_encoder, train_labels_cat = encode_labels(train_labels)
_, test_labels_cat = encode_labels(test_labels)

# Create datasets
train_dataset, validation_dataset = create_tf_datasets(
    train_files, train_labels_cat, 
    test_files, test_labels_cat,
    batch_size=batch_size,
    num_classes=num_classes
)

# Create model
model = create_crnn_model(input_shape=(128, 128, 1), num_classes=num_classes)
model.summary()

# Compute class weights
train_labels_list = train_labels
class_counts = Counter(train_labels_list)
print("Class distribution:", class_counts)

class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(train_labels_list), 
    y=train_labels_list
)
class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
print("Class weights:", class_weight_dict)

# Set up callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'chord_model_best.h5'), save_best_only=True)
]

# Train the model
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
def plot_history(history):
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

plot_history(history)

# Plot confusion matrix
def plot_confusion_matrix(model, dataset, label_encoder):
    """
    Plot the confusion matrix for the model predictions.
    """
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

# Print classification report
def print_classification_report(model, dataset, label_encoder):
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

# Evaluate the model
plot_confusion_matrix(model, validation_dataset, label_encoder)
print_classification_report(model, validation_dataset, label_encoder)

print(f"Training complete. Model saved to {MODEL_DIR}")