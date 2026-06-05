# Exploring Brain Signals and Affective Computing for Neuromarketing

## Overview

This project explores the intersection of EEG signal processing, machine learning, and deep learning to understand emotional responses and affective states through brain signals. The project focuses on emotion classification using the SEED (Similarity Emotion Database) dataset, employing both classical machine learning algorithms and advanced deep learning architectures.

The goal is to develop robust models for emotion recognition from EEG signals, with applications in neuromarketing research, user experience optimization, and affective computing.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Features](#features)
- [Usage](#usage)
- [Models](#models)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)

## Project Structure

```
.
├── data/                          # Dataset storage
│   ├── DEAP/                      # DEAP dataset (optional)
│   └── SEED_EEG/                  # SEED dataset
│       └── SEED_EEG/
│           ├── ExtractedFeatures_1s/      # Pre-extracted features (1-second windows)
│           ├── ExtractedFeatures_4s/      # Pre-extracted features (4-second windows)
│           ├── Preprocessed_EEG/          # Preprocessed raw EEG data
│           ├── SEED_RAW_EEG/              # Raw EEG signals
│           └── subject-id-gender-seed.txt # Subject metadata
├── model/                         # Trained model checkpoints and results
│   ├── deep_experiment/           # Deep learning model weights
│   │   ├── cnn_attention_best.pt
│   │   ├── cnn_lstm_best.pt
│   │   ├── deep4net_best.pt
│   │   ├── eegconformer_best.pt
│   │   ├── eegnet_best.pt
│   │   ├── lstm_best.pt
│   │   ├── shallowconv_best.pt
│   │   └── tcn_best.pt
│   ├── extra_trees/               # Ensemble model results
│   ├── logistic_regression/       # Linear model results
│   ├── random_forest/             # Random forest model results
│   ├── sgd_clf/                   # SGD classifier results
│   ├── xgboost/                   # XGBoost model results
│   ├── benchmark_summary.json     # Overall benchmark results
│   └── deep_results.json          # Deep learning results summary
├── notebooks/                     # Jupyter notebooks for exploration and analysis
│   ├── Autoencoder-Transformers-Models.ipynb
│   ├── BiLSTM-model.ipynb
│   ├── Classical_ML_Algorithms.ipynb
│   ├── CNN-Model.ipynb
│   ├── EEG_Query.ipynb
│   ├── EEG-XAI.ipynb
│   ├── GRU-Model.ipynb
│   ├── Models-Evaluation.ipynb
│   └── test.ipynb
├── output/                        # Benchmark and inference results
│   └── benchmark_inference_*.json
├── src/                           # Source code modules
│   ├── benchmark.py               # Model benchmarking
│   ├── config.py                  # Configuration settings
│   ├── feature_extraction.py      # EEG feature extraction
│   ├── llm_inference.py           # LLM-based inference
│   ├── llm_training.py            # LLM model training
│   ├── model_registry.py          # Model registry and factory
│   ├── preprocessing.py           # EEG signal preprocessing
│   ├── seed_loader.py             # SEED dataset loader
│   ├── tokenization.py            # Sequence tokenization
│   ├── train_baselines.py         # Baseline model training
│   └── train_deep_models.py       # Deep learning model training
├── tests/                         # Unit tests
│   ├── test_feature_extraction.py
│   ├── test_llm_inference.py
│   ├── test_preprocessing.py
│   └── test_seed_loader.py
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git for version control

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Exploring-Brain-Signals-and-Affective-Computing-for-Neuromarketing
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

- **torch**: Deep learning framework
- **scipy**: Scientific computing
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **sklearn**: Machine learning algorithms
- **braindecode**: EEG-specific deep learning toolkit

## Datasets

### SEED Dataset

The project primarily uses the SEED (Similarity Emotion Database) dataset, which contains:

- **15 subjects** with multiple sessions
- **3 emotion classes**: Positive, Negative, Neutral
- **15 trials** per subject per session
- **62 EEG channels** (10-20 International System)
- **Sampling rate**: 200 Hz
- **Pre-processed and raw data** available

#### Data Organization:

- `Preprocessed_EEG/`: Cleaned and preprocessed EEG signals
- `ExtractedFeatures_1s/`: Features extracted with 1-second windows
- `ExtractedFeatures_4s/`: Features extracted with 4-second windows
- `SEED_RAW_EEG/`: Original raw EEG recordings

### Additional Datasets

- **DEAP Dataset** (optional): For multimodal emotion recognition research

## Features

### Preprocessing
- Signal filtering and noise removal
- Artifact detection and handling
- Normalization and standardization
- Windowing and feature extraction

### Feature Extraction
- Statistical features (mean, variance, skewness, kurtosis)
- Frequency domain features (power spectral density, bandpower)
- Time-frequency features (wavelet transforms)
- Deep learning embeddings

### Models Implemented

#### Classical Machine Learning
- Logistic Regression
- Random Forest
- Extra Trees
- XGBoost
- SGD Classifier

#### Deep Learning
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- Gated Recurrent Unit (GRU)
- Attention-based CNN
- EEG-specific architectures (EEGNet, Deep4Net, EEGConformer)
- Temporal Convolutional Networks (TCN)
- Autoencoders and Transformers

## Usage

### Configuration

Edit `src/config.py` to customize:
- Dataset paths
- Preprocessing window sizes
- Model hyperparameters
- Training parameters
- Emotion labels and mappings

### Training Models

#### Train Classical ML Models
```bash
python src/train_baselines.py
```

#### Train Deep Learning Models
```bash
python src/train_deep_models.py
```

#### Benchmark All Models
```bash
python src/benchmark.py
```

### Running Notebooks

Open and run Jupyter notebooks for exploratory analysis:

```bash
jupyter notebook notebooks/
```

Key notebooks:
- `Classical_ML_Algorithms.ipynb`: Baseline model comparison
- `CNN-Model.ipynb`: CNN architecture exploration
- `EEG-XAI.ipynb`: Model explainability analysis
- `Models-Evaluation.ipynb`: Comprehensive model evaluation

### Feature Extraction

Extract features from EEG signals:
```python
from src.feature_extraction import extract_features
features = extract_features(eeg_signal)
```

### EEG Data Loading

Load SEED dataset:
```python
from src.seed_loader import SEEDLoader
loader = SEEDLoader()
data = loader.load_subject(subject_id=1)
```

### LLM Inference

Perform inference with LLM models:
```bash
python src/llm_inference.py
```

## Models

### Model Registry

All models are registered in `src/model_registry.py` for easy access and standardized training/evaluation.

### Trained Weights

Pre-trained model weights are available in `model/deep_experiment/`:
- `cnn_attention_best.pt`: CNN with attention mechanism
- `eegnet_best.pt`: EEGNet architecture
- `deep4net_best.pt`: Deep4Net architecture
- `lstm_best.pt`: Standard LSTM
- `cnn_lstm_best.pt`: CNN-LSTM hybrid
- `tcn_best.pt`: Temporal Convolutional Network
- And more...

## Configuration

### Key Configuration Parameters (src/config.py)

```python
# Preprocessing
WINDOW_SIZE = 450           # Primary window size
ML_WINDOW_SIZE = 800        # ML model window
LLM_WINDOW_SIZE = 1000      # LLM model window

# Dataset
N_TRIALS = 15               # Trials per subject
LABEL_FILE = "label.mat"    # Label file name

# Training
RANDOM_SEED = 42            # For reproducibility
BATCH_SIZE = 32             # Training batch size
EPOCHS = 100                # Training epochs
```

For complete configuration options, see `src/config.py`.

## Results

### Benchmark Results

Comprehensive benchmark results for all models are stored in:
- `model/benchmark_summary.json`: Overall performance summary
- `model/deep_results.json`: Deep learning model results
- `model/*/metrics.json`: Per-model metrics (precision, recall, F1-score, accuracy)
- `model/*/confusion.npy`: Confusion matrices

### Inference Results

Latest inference benchmarks:
- `output/benchmark_inference_*.json`: Time-stamped inference results

### Performance Summary

(To be filled in after running complete evaluation)

---

## Contributing

### Development Workflow

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/`
4. Format code: `black src/ tests/`
5. Submit a pull request

### Testing

Run unit tests:
```bash
pytest tests/ -v
```

## License

[Specify your license here]

## Authors

- [Add contributors here]

## Acknowledgments

- SEED Dataset creators and maintainers
- Braindecode community for EEG deep learning utilities
- References and inspirations in affective computing research

## References

- Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks", Wei-Long Zheng, and Bao-Liang Lu, IEEE Transactions on Autonomous Mental Development (IEEE TAMD), 2015.

## Questions & Issues

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: June 6, 2026
