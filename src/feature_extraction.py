"""
EEG Feature Extraction Module for SEED Dataset.

Extracts deterministic feature vectors from EEG windows for classical ML models.

Key Features Extracted:
    - Bandpower in frequency bands (theta, alpha, beta, gamma)
    - Differential Entropy (DE) per channel
    - Band-compressed Power Spectral Density (PSD)
    - Asymmetry features (DASM/RASM) using anatomical channel pairs

Design:
    - Deterministic: same input window always produces same features
    - Canonical channel order: fixed to SEED montage (62 channels)
    - Compatible across training, validation, and inference
    - No external metadata required at inference time
"""

import logging
import numpy as np

from scipy.signal import butter, filtfilt, welch
from config import FS, NYQ, ORDER, BANDS, PAIR_INDICES

logger = logging.getLogger("EEG_FEATURES")

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
    """
    Apply bandpass filter to EEG signal for a specific frequency band.

    Parameters
    ----------
    signal : np.ndarray
        EEG data with shape (n_channels, n_samples)
    band : str
        Frequency band name: 'theta', 'alpha', 'beta', or 'gamma'

    Returns
    -------
    np.ndarray
        Filtered signal with same shape as input
    """
    b, a = FILTERS[band]
    return filtfilt(b, a, signal, axis=1)


def compute_de(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Differential Entropy (DE) per EEG channel.

    Differential entropy is a measure of signal complexity and is correlated
    with emotional states in EEG signals.

    Parameters
    ----------
    signal : np.ndarray
        EEG data with shape (n_channels, n_samples)
    eps : float, optional
        Small constant for numerical stability, by default 1e-8

    Returns
    -------
    np.ndarray
        DE values per channel, shape (n_channels,)
    """
    var = np.var(signal, axis=1)
    return (0.5 * np.log(2 * np.pi * np.e * (var + eps))).astype(np.float32)


def compute_psd(signal: np.ndarray) -> np.ndarray:
    """
    Compute band-compressed Power Spectral Density (PSD).

    Computes average PSD across all channels within each frequency band
    (theta, alpha, beta, gamma) to create a 4-dimensional global spectral descriptor.

    Parameters
    ----------
    signal : np.ndarray
        EEG data with shape (n_channels, n_samples)

    Returns
    -------
    np.ndarray
        PSD values per band, shape (4,) corresponding to
        [theta_psd, alpha_psd, beta_psd, gamma_psd]
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
    """
    Compute log bandpower features across all EEG channels and bands.

    Extracts bandpower (log-transformed) in theta, alpha, beta, and gamma bands
    for all 62 EEG channels, resulting in 248 features (62 channels × 4 bands).

    Parameters
    ----------
    window : np.ndarray
        EEG window with shape (62, window_size)

    Returns
    -------
    np.ndarray
        Bandpower features, shape (248,) representing all channel-band pairs
    """
    feats = []

    for band in BANDS:
        filtered = bandpass_filter(window, band)
        power = np.mean(filtered**2, axis=1)
        feats.append(np.log(power + 1e-8))

    return np.concatenate(feats).astype(np.float32)


def compute_dasm(de: np.ndarray) -> np.ndarray:
    """
    Compute Differential Asymmetry (DASM) features.

    DASM measures left-right hemisphere differences in differential entropy.
    Used as a biomarker for emotional valence assessment.

    Parameters
    ----------
    de : np.ndarray
        Differential entropy per channel, shape (62,)

    Returns
    -------
    np.ndarray
        DASM values for each anatomical pair, shape (30,)
    """
    return np.asarray([de[l] - de[r] for l, r in PAIR_INDICES], dtype=np.float32)


def compute_rasm(de: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Relative Asymmetry (RASM) features.

    RASM is a normalized version of DASM that is more robust to baseline shifts.
    Measures the relative balance between left and right hemisphere activity.

    Parameters
    ----------
    de : np.ndarray
        Differential entropy per channel, shape (62,)
    eps : float, optional
        Small constant for numerical stability, by default 1e-8

    Returns
    -------
    np.ndarray
        RASM values for each anatomical pair, shape (30,)
    """
    return np.asarray(
        [de[l] / (de[r] + eps) for l, r in PAIR_INDICES], dtype=np.float32
    )


# =============================================================================
# MASTER FEATURE EXTRACTOR
# =============================================================================


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive handcrafted feature vector from EEG window.

    Combines multiple feature types for classical ML models:
    1. Bandpower: Log-transformed power in 4 bands × 62 channels (248 dims)
    2. Differential Entropy (DE): Signal complexity per channel (62 dims)
    3. Power Spectral Density (PSD): Global spectral descriptor (4 dims)
    4. Differential Asymmetry (DASM): Left-right difference (30 dims)
    5. Relative Asymmetry (RASM): Left-right ratio (30 dims)

    Total: ~374 features per window

    Parameters
    ----------
    window : np.ndarray
        EEG window with shape (62, window_size)

    Returns
    -------
    np.ndarray
        Feature vector with shape (~374,)
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
    Convert preprocessed EEG dataset into ML-ready feature matrix.

    Processes all windows in the dataset through the feature extraction pipeline
    to create feature vectors compatible with classical ML models.

    Parameters
    ----------
    processed_dataset : list
        List of preprocessed samples, each containing:
            {
                "windows": list of np.ndarray,
                "label": int,
                "subject": int,
                "trial": int
            }

    Returns
    -------
    tuple
        (X, y, groups) where:
            - X: Feature matrix, shape (n_samples, ~374)
            - y: Labels, shape (n_samples,)
            - groups: Subject IDs for group-aware cross-validation
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
