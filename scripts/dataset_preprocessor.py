import os
import librosa
import soundfile as sf
from src.data_preprocessing import parse_idmt_annotations
from sklearn.model_selection import train_test_split
import shutil

def split_dataset():
    """Split combined dataset into train/test"""
    all_files = []
    for chord in os.listdir("data/processed/Combined"):
        chord_dir = os.path.join("data/processed/Combined", chord)
        files = [os.path.join(chord_dir, f) for f in os.listdir(chord_dir)]
        all_files.extend(files)
    
    train_files, test_files = train_test_split(
        all_files, 
        test_size=0.2,
        stratify=[os.path.basename(os.path.dirname(f)) for f in all_files]
    )
    
    # Move files to train/test directories
    for f in train_files:
        dest = f.replace("Combined", "Combined/Training")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(f, dest)
        
    for f in test_files:
        dest = f.replace("Combined", "Combined/Test")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.rename(f, dest)
        
def process_idmt_dataset():
    """Process IDMT dataset into individual chord files"""
    raw_idmt = "data/raw/IDMT-SMT-CHORDS"
    processed_dir = "data/processed/Combined"
    
    # Process each guitar WAV
    for wav_file in os.listdir(f"{raw_idmt}/guitar"):
        if not wav_file.endswith(".wav"): continue
        
        # Load audio and annotations
        audio_path = f"{raw_idmt}/guitar/{wav_file}"
        ann_path = f"{raw_idmt}/guitar/guitar_annotation.lab"
        annotations = parse_idmt_annotations(ann_path)
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Save individual chords
        for start, end, chord in annotations:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            chunk = y[start_sample:end_sample]
            
            chord_dir = f"{processed_dir}/{chord}"
            os.makedirs(chord_dir, exist_ok=True)
            sf.write(f"{chord_dir}/{chord}_{start}.wav", chunk, sr)

def merge_datasets():
    """Combine IDMT and Guitar_Chords_V2"""
    src = "data/raw/Guitar_Chords_V2"
    dst = "data/processed/Combined"
    
    for root, dirs, files in os.walk(src):
        for file in files:
            src_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_path, src)
            dst_path = os.path.join(dst, relative_path)
            
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            
if __name__ == "__main__":
    process_idmt_dataset()
    merge_datasets()
    print("Datasets processed and merged!")