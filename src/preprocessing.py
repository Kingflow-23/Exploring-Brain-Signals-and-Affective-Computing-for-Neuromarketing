"""
===============================================================================
EEG PREPROCESSING PIPELINE — SEED DATASET
===============================================================================

🧠 WHY DO WE NEED EEG PREPROCESSING?
-------------------------------------------------------------------------------
Raw EEG signals are not directly usable by machine learning models because:

1. EEG is continuous time-series data
   → each trial has variable length (not fixed-size input)

2. EEG signals are noisy and non-stationary
   → amplitude varies across:
        - subjects
        - sessions
        - electrodes

3. ML models require structured input
   → fixed-size tensors (not raw variable-length signals)

4. Deep models (CNN / Transformers / LLM pipelines) require:
   → consistent shape
   → normalized distribution
   → temporal segmentation

-------------------------------------------------------------------------------
WHAT WE ALREADY HAVE (SEED PREPROCESSED DATA)
-------------------------------------------------------------------------------
The SEED dataset already provides:

✔ Downsampled EEG → 200 Hz
✔ Bandpass filtered → 0–75 Hz
✔ Cleaned and segmented into trials
✔ 62 EEG channels per sample

So we DO NOT need:
✘ Raw signal filtering
✘ Artifact removal (ICA, EOG cleaning)
✘ Downsampling
✘ Basic segmentation

-------------------------------------------------------------------------------
WHAT IS STILL REQUIRED (THIS FILE DOES THIS PART)
-------------------------------------------------------------------------------

We still must transform raw trial signals:

    (62, T variable length)

into ML-ready tensors:

    (N_windows, 62, window_size)

This is required because:

✔ Neural networks need fixed-size input
✔ Temporal patterns must be captured locally
✔ Long EEG trials must be decomposed into segments

-------------------------------------------------------------------------------
WHAT THIS PIPELINE DOES
-------------------------------------------------------------------------------

1. Takes EEG trial: (62, T)
2. Applies optional normalization (z-score)
3. Splits signal into overlapping windows
4. Outputs structured dataset:

    {
        "windows": (N, 62, W),
        "label": int,
        "subject": int,
        "trial": int
    }

-------------------------------------------------------------------------------
DESIGN CHOICES
-------------------------------------------------------------------------------

✔ Windowing instead of full-trial modeling
    → reduces noise
    → increases training samples
    → stabilizes learning

✔ No filtering or artifact removal
    → already done in SEED preprocessing

✔ Z-score normalization optional
    → improves convergence in deep models

✔ Overlap supported (default 50%)
    → improves temporal smoothness

===============================================================================
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
