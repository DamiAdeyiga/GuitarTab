# scripts/train_chord_model.py

import os
import tensorflow as tf
from src.data_preprocessing import load_wav_16k_mono, preprocess_audio, compute_spectrogram, get_file_paths, encode_labels, preprocess_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization, Reshape, Bidirectional, LSTM

# Define paths and parameters
DATA_DIR = '/content/drive/MyDrive/GuitarTab/data/raw/GuitarChordsV2/'
TRAIN_DIR = os.path.join(DATA_DIR, 'Training')
TEST_DIR = os.path.join(DATA_DIR, 'Test')
chords = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE
num_classes = len(chords)

# Load and encode labels
train_files, train_labels = get_file_paths(TRAIN_DIR, chords)
test_files, test_labels = get_file_paths(TEST_DIR, chords)

label_encoder, train_labels_cat = encode_labels(train_labels)
_, test_labels_cat = encode_labels(test_labels)  # Ensure same encoding

# Create TensorFlow datasets
def preprocess_function(file_path, label):
    wav = load_wav_16k_mono(file_path.numpy().decode('utf-8'))
    # Apply augmentation during training (50% probability)
    if np.random.random() < 0.5:
        wav = augment_audio(wav)
    # Standardize length
    wav = standardize_audio_length(wav)
    # Compute mel spectrogram instead of regular spectrogram
    mel_spec = compute_mel_spectrogram(wav, fixed_shape=(128, 128))
    # Add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    return mel_spec, label

def tf_preprocess(file_path, label):
    spectrogram, label = tf.py_function(preprocess_function, [file_path, label], [tf.float32, tf.float32])
    # Set fixed shape instead of None, None
    spectrogram.set_shape([128, 128, 1])
    label.set_shape([num_classes])
    return spectrogram, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels_cat))
train_dataset = train_dataset.map(tf_preprocess, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels_cat))
validation_dataset = validation_dataset.map(tf_preprocess, num_parallel_calls=AUTOTUNE)
validation_dataset = validation_dataset.batch(batch_size).prefetch(AUTOTUNE)


# Define the CRNN model
model = Sequential([
    # CNN part
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),  # 64x64
    
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),  # 32x32
    
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),  # 16x16
    
    # Reshape for RNN
    Reshape((16, 128)),  # Reshape to (16, 128) or adjust based on your dimensions
    
    # RNN part
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    
    # Classification head
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Display the model summary
model.summary()

# Set up callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('models/chord_model_best.h5', save_best_only=True)
]
# Check if classes are imbalanced
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

train_labels_list = [label for label in train_labels]
class_counts = Counter(train_labels_list)
print("Class distribution:", class_counts)

# Compute class weights if needed
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(train_labels_list), 
    y=train_labels_list
)
class_weight_dict = dict(zip(range(len(class_weights)), class_weights))
print("Class weights:", class_weight_dict)


# Train the model with class weights
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

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

    plt.show()

plot_history(history)

# Save the final model
model.save('models/chord_model_final.h5')

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Optionally, save label encoder
import pickle
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(model, dataset, label_encoder):
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    ----------
    model : tf.keras.Model
        Trained Keras model.
    dataset : tf.data.Dataset
        Validation dataset.
    label_encoder : sklearn.preprocessing.label_encoder 
        Label encoder used for encoding labels.
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
    plt.show()

# Plot confusion matrix
plot_confusion_matrix(model, validation_dataset, label_encoder)

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

    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

print_classification_report(model, validation_dataset, label_encoder)
