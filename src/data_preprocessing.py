# src/data_preprocessing.py

import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

def load_wav_16k_mono(file_path: str) -> np.ndarray:
    """
    Load an audio file, resample to 16kHz, and convert to mono.

    Parameters:
    ----------
    file_path : str
        Path to the audio file.

    Returns:
    -------
    np.ndarray
        Audio waveform as a 1D numpy array.
    """
    try:
        wav, sr = librosa.load(file_path, sr=16000, mono=True)
        return wav
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([])

def preprocess_audio(wav: np.ndarray, max_length: int = 48000) -> np.ndarray:
    """
    Truncate or pad the audio waveform to a fixed length.

    Parameters:
    ----------
    wav : np.ndarray
        Audio waveform.
    max_length : int, optional
        Desired length in samples (default is 48000, i.e., 3 seconds at 16kHz).

    Returns:
    -------
    np.ndarray
        Truncated or padded audio waveform.
    """
    if len(wav) > max_length:
        return wav[:max_length]
    elif len(wav) < max_length:
        padding = np.zeros(max_length - len(wav))
        return np.concatenate([wav, padding])
    else:
        return wav

def compute_spectrogram(wav: np.ndarray, frame_length: int = 320, frame_step: int = 32) -> np.ndarray:
    """
    Compute the spectrogram of the audio waveform.

    Parameters:
    ----------
    wav : np.ndarray
        Audio waveform.
    frame_length : int, optional
        Number of samples per frame (default is 320).
    frame_step : int, optional
        Number of samples to step (default is 32).

    Returns:
    -------
    np.ndarray
        Spectrogram as a 2D numpy array.
    """
    stft = librosa.stft(wav, n_fft=frame_length, hop_length=frame_step)
    spectrogram = np.abs(stft)
    return spectrogram

def plot_waveform(wav: np.ndarray, title: str = "Waveform"):
    """
    Plot the audio waveform.

    Parameters:
    ----------
    wav : np.ndarray
        Audio waveform.
    title : str, optional
        Title of the plot (default is "Waveform").
    """
    plt.figure(figsize=(14, 5))
    plt.plot(wav)
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.show()

def plot_spectrogram(spectrogram: np.ndarray, title: str = "Spectrogram"):
    """
    Plot the spectrogram.

    Parameters:
    ----------
    spectrogram : np.ndarray
        Spectrogram.
    title : str, optional
        Title of the plot (default is "Spectrogram").
    """
    plt.figure(figsize=(14, 5))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")
    plt.colorbar(format='%+2.0f dB')
    plt.show()