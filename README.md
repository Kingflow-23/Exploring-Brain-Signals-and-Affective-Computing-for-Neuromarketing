# Exploring Brain Signals and Affective Computing for Neuromarketing

## Overview

This project explores the intersection of EEG signal processing, machine learning, and deep learning to understand emotional responses and affective states through brain signals. The project focuses on emotion classification using the SEED (Similarity Emotion Database) dataset, employing both classical machine learning algorithms and advanced deep learning architectures.

The goal is to develop robust models for emotion recognition from EEG signals, with applications in neuromarketing research, user experience optimization, and affective computing. And as a part of the project, it was a goal to assess for llm perfromance in emotion classification.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Features](#features)
- [Usage](#usage)
- [Models](#models)
- [Configuration](#configuration)
- [Results](#results)
- [References](#references)

## Project Structure

```
.
├── data/                          # Dataset storage
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
├── output/                        # Benchmark and inference results
│   └── benchmark_inference_*.json
├── prompt/                        # Prompt used for llm inference 
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
   git clone https://github.com/Kingflow-23/Exploring-Brain-Signals-and-Affective-Computing-for-Neuromarketing.git
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

- **DEAP Dataset** : For a more marketing approach. It was not used for this project but in another interesting one that can be found **[here](https://github.com/vsx23733/AI-CLINIC)**

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

#### LLM
As a part of our project, we had to assess the llm performance in emotion prediction. To do that we used a local qwen model: "qwen/qwen3.6-35b-a3b" 

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

After configuring the test dataset, run:

```bash
python src/benchmark.py
```

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

All DL models are registered in `src/model_registry.py` for easy access and standardized training/evaluation.

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

### Benchmark Artifacts

All experimental outputs are stored in the repository:

- `model/benchmark_summary.json`: global performance summary across models
- `model/deep_results.json`: deep learning evaluation results
- `model/*/metrics.json`: per-model metrics (accuracy, F1-score, precision, recall)
- `model/*/confusion.npy`: confusion matrices for error analysis
- `output/benchmark_inference_*.json`: timestamped inference logs

---

### Overall Performance Summary

The benchmark evaluates **classical ML models**, **deep learning architectures**, and a **Large Language Model (LLM)** on the SEED EEG emotion recognition dataset using a **3-class classification task**:

- Negative
- Neutral
- Positive

---

### Classical Machine Learning Results

| Model | Window Accuracy | Trial Accuracy |
|------|------|------|
| Logistic Regression | **62.83%** | **66.67%** |
| XGBoost | 60.88% | 60.00% |
| Random Forest | 60.05% | 60.00% |
| Extra Trees | 59.05% | 57.78% |
| SGD Classifier | 58.81% | 55.56% |

**Key insight:**
- Logistic Regression is the strongest baseline
- Indicates that EEG features are partially **linearly separable after preprocessing**
- Tree-based methods show class bias toward **Positive emotion**

---

### Deep Learning Results

| Model | Window Accuracy | Trial Accuracy |
|------|------|------|
| LSTM | **63.70%** | **71.11%** |
| TCN | 60.40% | **71.11%** |
| EEGNet | 56.78% | 53.33% |
| Deep4Net | 55.07% | 55.56% |
| ShallowConvNet | 56.65% | 60.00% |
| EEGConformer | 56.57% | 55.56% |
| CNN-Attention | 55.12% | 55.56% |
| CNN-LSTM | 51.28% | 51.11% |

**Key insight:**
- Temporal architectures (**LSTM, TCN**) outperform convolution-only models
- Performance saturates around **~71% trial accuracy**
- No architecture significantly breaks this ceiling

---

### LLM-Based Classification (Qwen3.6-35B-A3B)

| Model | Window Accuracy | Trial Accuracy |
|------|------|------|
| Qwen3.6-35B-A3B | **38.11%** | **44.44%** |

**Key insight:**
- Strong underperformance compared to supervised models
- Indicates that **LLMs are not suitable for raw EEG feature reasoning without task-specific training or fine-tuning**

---

## Performance Analysis

### 1. Class Structure and Separability

Across all models, a consistent pattern emerges:

- **Positive emotion → highest recall**
- **Negative emotion → moderate confusion**
- **Neutral emotion → highest ambiguity**

This suggests:

> The Neutral class behaves as a **transition manifold** between Positive and Negative states rather than a strictly separable class.

---

### 2. Why Logistic Regression Performs Surprisingly Well

Logistic Regression remains competitive because:

- EEG features are heavily engineered (spectral + asymmetry + entropy)
- Decision boundaries are approximately linear in transformed space
- Noise is reduced through preprocessing and window averaging

---

### 3. Deep Learning Performance Ceiling (~70%)

Despite increased model complexity, deep architectures converge around:

- **~63–64% window accuracy**
- **~71% trial accuracy**

This ceiling is primarily due to:

- strong inter-subject variability
- limited dataset size (15 subjects)
- loss of spatial EEG structure after feature extraction
- redundancy across temporal windows

> Result: models learn temporal smoothing rather than richer discriminative structure.

---

### 4. LLM Failure Mode

The LLM fails due to structural mismatch:

- input = normalized scalar EEG features
- no spatial topology
- no temporal continuity
- no domain-specific pretraining

Consequently:

> The model treats EEG signals as generic numerical tokens, lacking inductive bias for neurophysiological patterns.

---

## Key Takeaway

Across all model families:

> Performance is primarily constrained by **feature separability**, not model capacity.

This explains why:

- classical ML is competitive
- deep learning does not significantly outperform baselines
- LLMs fail entirely in zero-shot setting

---

## Authors

- **Florian HOUNKPATIN** — https://www.linkedin.com/in/florian-hounkpatin/
- **Noémi DOMBOU** — https://www.linkedin.com/in/noemi-dombou/
- **Axel ONOBIONO** — https://www.linkedin.com/in/axel-onobiono/
- **Ephraim KOSSONOU** — https://www.linkedin.com/in/ephraïm-kossonou/

---

## References

- Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks", Wei-Long Zheng, and Bao-Liang Lu, IEEE Transactions on Autonomous Mental Development (IEEE TAMD), 2015.


---

**Last Updated**: June 7, 2026
