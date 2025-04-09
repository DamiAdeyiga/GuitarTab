# GuitarTab

GuitarTab is a project aimed at making guitar learning more accessible by converting raw audio files into chord predictions and, eventually, detailed guitar tablature. 

## Goals
- **Phase 1:** Develop a chord prediction model trained on labeled chord data.
- **Phase 2:** Build a tablature prediction model that leverages chord information to produce detailed guitar tabs.
- **Phase 3:** Integrate both models and explore extracting audio from YouTube videos.

## Project Structure
- `data/`: Contains raw and processed audio datasets.
- `src/`: Will hold Python modules for data preprocessing, model definitions, and utilities.
- `scripts/`: Future home of training and inference scripts.
- `notebooks/`: Jupyter notebooks for exploration, experimentation, and demonstration.

## Setup Instructions

**Prerequisites**:
- Python 3.8+ recommended
- Git
- Virtual environment tool (such as `venv` or `conda`)

**Installation**:
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/GuitarTab.git
   cd GuitarTab


## Day 3: Chord Prediction Model

### Data Preparation

- **Loading Audio Files**: Converted audio files into 16kHz mono WAV files.
- **Preprocessing**: Truncated or padded audio to 3 seconds (48000 samples).
- **Spectrograms**: Computed spectrograms from audio waveforms for model input.
- **Label Encoding**: Encoded chord labels numerically and categorically.

### Model Training

- **Architecture**: Built a CNN with multiple convolutional and pooling layers.
- **Training**: Trained the model with early stopping and model checkpointing.
- **Evaluation**: Achieved [Insert your metrics here] on the validation set.

### Usage

To train the model, run:

```bash
python scripts/train_chord_model.py