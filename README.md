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