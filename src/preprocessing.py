"""
EEG Preprocessing Module for SEED Dataset.

Transforms raw EEG trials into fixed-length windows for machine learning models.

Key Operations:
    - Splits variable-length EEG trials into overlapping windows
    - Applies optional z-score normalization
    - Maintains metadata (subject, trial, label) for each window
    - Outputs structured dataset format compatible with ML/DL pipelines

Design Notes:
    - SEED data is already filtered (0-75 Hz) and downsampled (200 Hz)
    - No additional artifact removal is applied
    - Windows can overlap (controlled by step_size parameter)
    - Normalization is applied per-sample to maintain signal structure
"""

import numpy as np
import logging

from tqdm import tqdm
from config import WINDOW_SIZE, STEP_SIZE, LABELS_MAP

logger = logging.getLogger("EEG_PREPROCESSOR")


# =============================================================================
# NORMALIZATION
# =============================================================================


def zscore_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply channel-wise z-score normalization.

    Why:
    ----
    EEG amplitudes differ across subjects and electrodes.
    Normalization stabilizes training and prevents dominance of high-amplitude channels.

    Input:
        x: (62, T)

    Output:
        normalized x: (62, T)
    """
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)

    return (x - mean) / (std + eps)


# =============================================================================
# WINDOWING FUNCTION
# =============================================================================


def sliding_window(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> np.ndarray:
    """
    Convert continuous EEG trial into overlapping fixed-size windows.

    WHY WINDOWING?
    -------------
    EEG is non-stationary → statistical properties change over time.
    Windowing allows the model to:
        - capture local temporal dynamics
        - increase dataset size
        - stabilize learning

    INPUT:
        signal: (62, T)

    OUTPUT:
        windows: (N_windows, 62, window_size)
    """

    channels, T = signal.shape
    windows = []

    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        window = signal[:, start:end]
        windows.append(window)

    return np.array(windows)


# =============================================================================
# FULL PREPROCESSING PIPELINE
# =============================================================================


def preprocess_trial(
    signal: np.ndarray,
    normalize: bool = True,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> np.ndarray:
    """
    Full preprocessing pipeline for one EEG trial.

    Steps:
    ------
    1. Optional normalization (z-score)
    2. Sliding window segmentation

    OUTPUT:
        (N_windows, 62, window_size)
    """

    if normalize:
        signal = zscore_normalize(signal)

    windows = sliding_window(signal, window_size=window_size, step_size=step_size)

    return windows


# =============================================================================
# DATASET-LEVEL PROCESSOR
# =============================================================================


def preprocess_dataset(
    dataset: list, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE
) -> list:
    """
    Convert full SEED dataset into windowed ML-ready format.

    INPUT FORMAT:
        [
            {
                "signal": (62, T),
                "label": int,
                "subject": int,
                "trial": int
            }
        ]

    OUTPUT FORMAT:
        [
            {
                "windows": (N, 62, W),
                "label": int,
                "subject": int,
                "trial": int
            }
        ]
    """

    logger.info("Starting EEG preprocessing pipeline...")

    processed = []

    for idx, sample in enumerate(tqdm(dataset, desc="Preprocessing EEG")):

        signal = sample["signal"]

        windows = preprocess_trial(signal, window_size=window_size, step_size=step_size)

        processed.append(
            {
                "windows": windows,
                "label": LABELS_MAP[sample["label"]],
                "subject": sample["subject"],
                "trial": sample["trial"],
            }
        )

        logger.info(
            f"Processed sample {idx+1}/{len(dataset)} | " f"windows: {windows.shape}"
        )

    logger.info("EEG preprocessing completed. ✅")

    return processed
