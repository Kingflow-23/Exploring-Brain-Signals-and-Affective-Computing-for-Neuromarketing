"""
Model Registry Module.

Central registry for all EEG deep learning models used in the project.

Unified interface for model instantiation, ensuring consistency across training and inference pipelines.

Supported Models:

    Custom Architectures (3D input):
        - lstm: LSTM (batch, seq_len, channels=62)
        - tcn: Temporal Convolutional Network (batch, seq_len, channels=62)
        - cnn_attention: CNN with attention mechanism (batch, 1, channels=62, seq_len)
        - cnn_lstm: CNN-LSTM hybrid (batch, 1, channels=62, seq_len)

    Braindecode Models (4D input):
        - eegnet: Lightweight EEG-specific CNN
        - deep4net: Deep convolutional network for EEG
        - shallowconv: Shallow convolutional network (ShallowFBCSPNet)
        - eegconformer: Vision Transformer-inspired EEG model

Input Shapes:
    - 3D input (LSTM, TCN): (batch_size, sequence_length, 62 channels)
    - 4D input (CNN-based): (batch_size, 1 channel, 62 electrodes, sequence_length)

Model Output:
    All models output (batch_size, 3) logits for 3-class emotion classification

Configuration:
    - Window size: Configured in config.py (affects sequence length)
    - Number of channels: 62 (SEED electrode montage)
    - Number of classes: 3 (positive, neutral, negative)

Usage:
    model = get_model('eegnet', input_shape=(1, 62, 450), num_classes=3)
"""

import torch.nn as nn

from train_deep_models import (
    EEG_CNN_Attention,
    EEG_LSTM,
    EEG_TCN,
    EEG_CNN_LSTM,
)

from braindecode.models import (
    EEGNet,
    Deep4Net,
    ShallowFBCSPNet,
    EEGConformer,
)

from config import WINDOW_SIZE


def get_model(model_name, n_chans=62, n_classes=3, sfreq=200):
    """
    Retrieve a model from the registry by name.

    Parameters
    ----------
    model_name : str
        Name of the model to retrieve.
    n_chans : int, optional
        Number of EEG channels, by default 62
    n_classes : int, optional
        Number of classification classes, by default 3
    sfreq : int, optional
        Sampling frequency in Hz, by default 200

    Returns
    -------
    dict
        Dictionary with keys:
            - "model": instantiated model
            - "input_mode": expected input shape format ("eeg_3d" or "eeg_4d")

    Raises
    ------
    ValueError
        If model_name is not in registry.
    """

    input_window_seconds = WINDOW_SIZE / sfreq

    registry = {
        # ==================================================
        # CUSTOM 4D MODELS
        # ==================================================
        "cnn_attention": {
            "model": EEG_CNN_Attention(
                n_chans=n_chans,
                n_classes=n_classes,
            ),
            "input_mode": "eeg_4d",
        },
        "cnn_lstm": {
            "model": EEG_CNN_LSTM(
                n_chans=n_chans,
                n_classes=n_classes,
            ),
            "input_mode": "eeg_4d",
        },
        # ==================================================
        # CUSTOM 3D MODELS
        # ==================================================
        "lstm": {
            "model": EEG_LSTM(
                n_chans=n_chans,
                n_classes=n_classes,
            ),
            "input_mode": "eeg_3d",
        },
        "tcn": {
            "model": EEG_TCN(
                n_chans=n_chans,
                n_classes=n_classes,
            ),
            "input_mode": "eeg_3d",
        },
        # ==================================================
        # BRAINDCODE MODELS
        # ==================================================
        "eegnet": {
            "model": EEGNet(
                n_chans=n_chans,
                n_outputs=n_classes,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
                final_conv_length="auto",
            ),
            "input_mode": "eeg_3d",
        },
        "deep4net": {
            "model": Deep4Net(
                n_chans=n_chans,
                n_outputs=n_classes,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
                final_conv_length="auto",
            ),
            "input_mode": "eeg_3d",
        },
        "shallowconv": {
            "model": ShallowFBCSPNet(
                n_chans=n_chans,
                n_outputs=n_classes,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            ),
            "input_mode": "eeg_3d",
        },
        "eegconformer": {
            "model": EEGConformer(
                n_chans=n_chans,
                n_outputs=n_classes,
                input_window_seconds=input_window_seconds,
                sfreq=sfreq,
            ),
            "input_mode": "eeg_3d",
        },
    }

    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    return registry[model_name]
