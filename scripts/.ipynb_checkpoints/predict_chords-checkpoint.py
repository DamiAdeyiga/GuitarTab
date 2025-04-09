import os
import numpy as np
import tensorflow as tf
import librosa
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_wav_16k_mono, standardize_audio_length, compute_mel_spectrogram

def predict_chord(audio_file, model_path, encoder_path):
    """
    Predict chord from audio file
    """
    # Load model and label encoder
    model = load_model(model_path)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Process audio
    wav = load_wav_16k_mono(audio_file)
    wav = standardize_audio_length(wav)
    mel_spec = compute_mel_spectrogram(wav)
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    mel_spec = np.expand_dims(mel_spec, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(mel_spec)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]
    predicted_chord = label_encoder.classes_[predicted_index]
    
    # Display results
    print(f"Predicted chord: {predicted_chord} (confidence: {confidence:.2f})")
    
    # Plot spectrogram with prediction
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spec[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Mel Spectrogram - Predicted: {predicted_chord} ({confidence:.2f})")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency Bin")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
    return predicted_chord, confidence

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict guitar chord from audio file')
    parser.add_argument('audio_file', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/chord_model_best.h5',
                        help='Path to trained model')
    parser.add_argument('--encoder', type=str, default='models/label_encoder.pkl',
                        help='Path to label encoder')
    
    args = parser.parse_args()
    
    predict_chord(args.audio_file, args.model, args.encoder)