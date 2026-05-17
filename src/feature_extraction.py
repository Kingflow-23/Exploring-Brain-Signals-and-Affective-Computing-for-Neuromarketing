"""
===============================================================================
EEG FEATURE EXTRACTION PIPELINE (SEED - PRODUCTION GRADE)
===============================================================================

PURPOSE
-------
Transforms EEG windows into fixed feature vectors for classical ML models.

This module is STRICTLY deterministic:
    same input window → same feature vector

It MUST be shared across:
    - training
    - validation
    - inference

===============================================================================
INPUT CONTRACT
===============================================================================

window: np.ndarray of shape (62, W)

Where:
    62 = EEG channels (fixed SEED montage order)
    W  = time samples

IMPORTANT:
----------
We assume channel ORDER is fixed and corresponds to SEED montage.

We DO NOT depend on raw signal metadata at inference time,
but we DO define a canonical channel map internally.

===============================================================================
FEATURE SET
===============================================================================

1. Bandpower (theta, alpha, beta, gamma)
2. Differential Entropy (DE)
3. Band-compressed PSD
4. Asymmetry features (DASM / RASM using anatomical pairs)

===============================================================================
"""

import logging
import numpy as np

from scipy.signal import butter, filtfilt, welch

logger = logging.getLogger("EEG_FEATURES")

# =============================================================================
# SAMPLING CONFIG
# =============================================================================

FS = 200
NYQ = FS / 2
ORDER = 5

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# =============================================================================
# CANONICAL SEED CHANNEL MONTAGE
# =============================================================================

CHANNELS = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]

CHANNEL_INDEX = {ch: i for i, ch in enumerate(CHANNELS)}

# =============================================================================
# LEFT-RIGHT PAIRS (ANATOMICAL SYMMETRY)
# =============================================================================

LEFT_RIGHT_PAIRS = [
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F5", "F6"),
    ("F3", "F4"),
    ("F1", "F2"),
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("FT7", "FT8"),
    ("T7", "T8"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("TP7", "TP8"),
    ("P7", "P8"),
    ("P5", "P6"),
    ("P3", "P4"),
    ("P1", "P2"),
    ("PO7", "PO8"),
    ("PO5", "PO6"),
    ("PO3", "PO4"),
    ("CB1", "CB2"),
    ("O1", "O2"),
]

PAIR_INDICES = [(CHANNEL_INDEX[l], CHANNEL_INDEX[r]) for l, r in LEFT_RIGHT_PAIRS]

# =============================================================================
# FILTERS
# =============================================================================


def create_band_filters():
    filters = {}

    for band, (low, high) in BANDS.items():
        b, a = butter(ORDER, [low / NYQ, high / NYQ], btype="band")
        filters[band] = (b, a)

    return filters


FILTERS = create_band_filters()

# =============================================================================
# FEATURE COMPONENTS
# =============================================================================


def bandpass_filter(signal: np.ndarray, band: str) -> np.ndarray:
    b, a = FILTERS[band]
    return filtfilt(b, a, signal, axis=1)


def compute_de(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Differential entropy per channel."""
    var = np.var(signal, axis=1)
    return (0.5 * np.log(2 * np.pi * np.e * (var + eps))).astype(np.float32)


def compute_psd(signal: np.ndarray) -> np.ndarray:
    """
    Band-compressed PSD (robust global spectral descriptor).
    Output: (4,)
    """
    per_channel = []

    for ch in signal:
        freqs, psd = welch(ch, fs=FS, nperseg=min(256, len(ch)))

        per_channel.append(
            [
                np.mean(psd[(freqs >= 4) & (freqs < 8)]),
                np.mean(psd[(freqs >= 8) & (freqs < 13)]),
                np.mean(psd[(freqs >= 13) & (freqs < 30)]),
                np.mean(psd[(freqs >= 30) & (freqs < 45)]),
            ]
        )

    return np.mean(per_channel, axis=0).astype(np.float32)


def compute_bandpower(window: np.ndarray) -> np.ndarray:
    """(62 × 4) bandpower features flattened."""
    feats = []

    for band in BANDS:
        filtered = bandpass_filter(window, band)
        power = np.mean(filtered**2, axis=1)
        feats.append(np.log(power + 1e-8))

    return np.concatenate(feats).astype(np.float32)


def compute_dasm(de: np.ndarray) -> np.ndarray:
    """Left-right difference asymmetry."""
    return np.asarray([de[l] - de[r] for l, r in PAIR_INDICES], dtype=np.float32)


def compute_rasm(de: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Left-right ratio asymmetry."""
    return np.asarray(
        [de[l] / (de[r] + eps) for l, r in PAIR_INDICES], dtype=np.float32
    )


# =============================================================================
# MASTER FEATURE EXTRACTOR
# =============================================================================


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Convert EEG window → feature vector.

    Input:
        (62, W)

    Output:
        (F,)
    """

    features = []

    # 1. bandpower
    features.append(compute_bandpower(window))

    # 2. DE
    de = compute_de(window)
    features.append(de)

    # 3. PSD
    features.append(compute_psd(window))

    # 4. asymmetry
    features.append(compute_dasm(de))
    features.append(compute_rasm(de))

    return np.concatenate(features).astype(np.float32)


# =============================================================================
# DATASET CONVERSION
# =============================================================================


def extract_dataset_features(processed_dataset):
    """
    Convert full dataset into ML-ready arrays.

    Each window becomes one training sample.
    """

    logger.info("Extracting EEG features...")

    X, y, groups = [], [], []

    for sample in processed_dataset:
        label = sample["label"]
        subject = sample["subject"]

        for window in sample["windows"]:
            X.append(extract_features(window))
            y.append(label)
            groups.append(subject)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y, groups
