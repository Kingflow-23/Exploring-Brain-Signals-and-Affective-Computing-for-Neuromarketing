"""
EEG-Based Emotion Classification Module.

Core package for exploring brain signals and affective computing applied to neuromarketing.

Submodules:
    - config: Central configuration for all tasks
    - seed_loader: SEED dataset loading and preprocessing
    - preprocessing: EEG signal windowing and normalization
    - feature_extraction: Deterministic feature extraction for classical ML
    - train_baselines: Classical machine learning model training
    - train_deep_models: Deep learning model training and architectures
    - model_registry: Central registry for all EEG models
    - benchmark: Comprehensive model evaluation and benchmarking
    - llm_inference: EEG-to-LLM inference pipeline

Datasets Supported:
    - SEED: Similarity Emotion Database (primary)

Models Supported:
    Classical ML: Logistic Regression, Random Forest, Extra Trees, XGBoost, SGD
    Deep Learning: CNN, LSTM, GRU, TCN, Attention mechanisms, EEGNet, Deep4Net, EEGConformer

Quick Start:
    >>> from src.seed_loader import build_seed_dataset
    >>> from src.preprocessing import preprocess_dataset
    >>> from src.train_baselines import train_all_models
"""
