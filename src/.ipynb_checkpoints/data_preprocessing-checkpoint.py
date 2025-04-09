import numpy as np
import tensorflow as tf
import librosa
import os
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_wav_16k_mono(file_path):
    """Load a WAV file, resample to 16kHz and convert to mono"""
    wav, sr = librosa.load(file_path, sr=16000, mono=True)
    return wav

def standardize_audio_length(wav, sr=16000, duration=3.0):
    """Standardize audio to a fixed length"""
    target_length = int(duration * sr)
    if len(wav) > target_length:
        wav = wav[:target_length]
    else:
        wav = np.pad(wav, (0, max(0, target_length - len(wav))))
    return wav

def augment_audio(wav, sr=16000, apply_prob=0.5):
    """Apply random augmentations to audio data"""
    # Random pitch shift between -2 and 2 semitones
    if np.random.random() < apply_prob:
        n_steps = np.random.uniform(-2, 2)
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)
    
    # Random time stretching between 0.8x and 1.2x
    if np.random.random() < apply_prob:
        rate = np.random.uniform(0.8, 1.2)
        wav = librosa.effects.time_stretch(wav, rate=rate)
    
    # Add random noise
    if np.random.random() < apply_prob/2:  # Lower probability for noise
        noise_level = np.random.uniform(0, 0.005)
        noise = np.random.normal(0, noise_level, len(wav))
        wav = wav + noise
        
    # Ensure the amplitude is within [-1, 1]
    wav = np.clip(wav, -1, 1)
    
    return wav

def compute_mel_spectrogram(wav, sr=16000, n_mels=128, fixed_shape=(128, 128)):
    """Compute mel spectrogram with fixed dimensions"""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
    
    # Convert to decibels
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to fixed dimensions if needed
    if fixed_shape:
        mel_spec = librosa.util.fix_length(mel_spec, size=fixed_shape[1], axis=1)
    
    return mel_spec

def get_file_paths(directory, class_names):
    """Get file paths and corresponding labels from a directory structure"""
    files = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_files = glob.glob(os.path.join(class_dir, '*.wav'))
        files.extend(class_files)
        labels.extend([class_name] * len(class_files))
    
    return files, labels

def encode_labels(labels):
    """Encode string labels to one-hot encoded vectors"""
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)
    
    return label_encoder, categorical_labels

def preprocess_function(file_path, label, augment=True):
    """Process a single audio file and its label"""
    wav = load_wav_16k_mono(file_path)
    
    # Apply augmentation during training
    if augment and np.random.random() < 0.5:
        wav = augment_audio(wav)
        
    # Standardize length
    wav = standardize_audio_length(wav)
    
    # Compute mel spectrogram
    mel_spec = compute_mel_spectrogram(wav)
    
    # Add channel dimension
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    
    return mel_spec, label

def create_tf_datasets(train_files, train_labels_cat, test_files, test_labels_cat, 
                       batch_size=16, num_classes=8):
    """Create TensorFlow datasets for training and validation"""
    AUTOTUNE = tf.data.AUTOTUNE
    
    def tf_preprocess(file_path, label, augment=True):
        """TensorFlow wrapper for preprocessing function"""
        def _preprocess(file_path, label):
            # Convert tensors to numpy
            file_path_str = file_path.numpy().decode('utf-8')
            label_np = label.numpy()
            
            # Process audio
            spectrogram, label = preprocess_function(file_path_str, label_np, augment)
            return spectrogram.astype(np.float32), label
        
        # Use py_function to call Python code
        spectrogram, label = tf.py_function(
            _preprocess, [file_path, label], [tf.float32, tf.float32])
        
        # Set shapes
        spectrogram.set_shape([128, 128, 1])
        label.set_shape([num_classes])
        
        return spectrogram, label
    
    # Create training dataset with augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels_cat))
    train_dataset = train_dataset.map(
        lambda x, y: tf_preprocess(x, y, True), 
        num_parallel_calls=AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(AUTOTUNE)
    
    # Create validation dataset without augmentation
    validation_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels_cat))
    validation_dataset = validation_dataset.map(
        lambda x, y: tf_preprocess(x, y, False),
        num_parallel_calls=AUTOTUNE
    )
    validation_dataset = validation_dataset.batch(batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, validation_dataset