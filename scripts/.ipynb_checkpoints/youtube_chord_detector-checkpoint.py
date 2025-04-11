import os
import librosa
import numpy as np
from src.data_preprocessing import standardize_audio_length, compute_mel_spectrogram
from scripts.predict_chords import predict_chords  # Reuse your existing function
import yt_dlp
import soundfile as sf
def process_youtube_video(url, model_path, encoder_path, chunk_size=3):
    """Process YouTube video using yt-dlp"""
    # Download audio with yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace('.webm', '.wav').replace('.mp4', '.wav')
    
    # Load audio
    y, sr = librosa.load(filename, sr=16000)
    os.remove(filename)
    
    # Process chunks (same as before)
    chunks = []
    for i in range(0, len(y), sr*chunk_size):
        chunk = y[i:i+sr*chunk_size]
        if len(chunk) < sr*chunk_size:
            chunk = librosa.util.fix_length(chunk, size=sr*chunk_size)
        chunks.append(chunk)
    
    results = []
    for i, chunk in enumerate(chunks):
        temp_file = f"temp_chunk_{i}.wav"
        sf.write(temp_file, chunk, sr)
        
        chord, conf = predict_chords(temp_file, model_path, encoder_path)
        results.append({
            "start": i*chunk_size,
            "end": (i+1)*chunk_size,
            "chord": chord,
            "confidence": conf
        })
        os.remove(temp_file)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect chords in YouTube video')
    parser.add_argument('url', type=str, help='YouTube URL')
    parser.add_argument('--model', type=str, default='models/chord_model_best.h5')
    parser.add_argument('--encoder', type=str, default='models/label_encoder.pkl')
    
    args = parser.parse_args()
    
    results = process_youtube_video(args.url, args.model, args.encoder)
    
    # Print results
    print("\nChord Timeline:")
    for entry in results:
        print(f"{entry['start']:03d}-{entry['end']:03d}s: {entry['chord']} ({entry['confidence']:.2f})")