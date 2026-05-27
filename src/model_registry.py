"""
Model Registry Module.

Central registry for all EEG deep learning models used in the project.

Supported Models:
    Custom Models:
        - cnn_attention: CNN with Attention mechanism (4D input)
        - cnn_lstm: CNN-LSTM hybrid (4D input)
        - lstm: LSTM network (3D input)
        - tcn: Temporal Convolutional Network (3D input)

    Braindecode Models:
        - eegnet: Lightweight EEG-specific CNN
        - deep4net: Deep convolutional network for EEG
        - shallowconv: Shallow convolutional network for EEG
        - eegconformer: Transformer-based EEG model

Input Modes:
    - eeg_3d: (batch, channels, time)
    - eeg_4d: (batch, 1, channels, time)
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
