"""
SEED Dataset Loader Module.

Loads raw EEG recordings from the SEED (Salient Emotion Emotion Database) dataset
and reconstructs the original data structure (subjects, trials, labels).

Key Features:
    - Loads emotion labels from .mat files
    - Reconstructs EEG trial structure with metadata
    - Handles subject-level data organization
    - Logs progress for large file operations
"""

import os
import logging
import numpy as np

from scipy.io import loadmat

from config import N_TRIALS, LABEL_FILE, EEG_KEY_PATTERN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SEED_LOADER")



def load_labels(label_path):
    """
    Load emotion labels from SEED dataset.

    The SEED dataset stores emotion labels in a .mat file with 15 labels per subject:
        - 1 = positive emotion
        - 0 = neutral emotion
        - -1 = negative emotion

    Parameters
    ----------
    label_path : str
        Path to MATLAB .mat file containing emotion labels.

    Returns
    -------
    np.ndarray
        Normalized label array, shape (1, n_trials) or (n_trials, n_trials).

    Raises
    ------
    ValueError
        If no valid label variable found in the .mat file.
    """

    logger.info(f"Loading labels from: {label_path}")

    mat = loadmat(label_path, simplify_cells=True)

    for k, v in mat.items():
        if not k.startswith("__"):
            labels = np.array(v)

            # Normalize to ensure consistent indexing
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)

            logger.info(f"Label shape normalized: {labels.shape}")
            return labels

    raise ValueError("No valid label found in label.mat")




def detect_prefix(mat):
    """
    Detect EEG data prefix from MATLAB structure keys.

    Parameters
    ----------
    mat : dict
        MATLAB structure loaded as dictionary.

    Returns
    -------
    str or None
        The detected EEG key prefix, or None if not found.
    """
    """
    Automatically detect subject-specific EEG prefix.

    -------------------------------------------------------------------------
    Purpose
    -------------------------------------------------------------------------
    SEED MAT files use subject-dependent prefixes such as:
        djc_eeg1
        jl_eeg1

    This function extracts the prefix dynamically.

    -------------------------------------------------------------------------
    Logic
    -------------------------------------------------------------------------
    - Search for any key containing "_eeg1"
    - Extract substring before it
    - Return prefix

    -------------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------------
    str: EEG prefix (e.g., "djc")
    """

    for k in mat.keys():
        if f"{EEG_KEY_PATTERN}1" in k:
            prefix = k.split(f"{EEG_KEY_PATTERN}1")[0]
            logger.debug(f"Detected prefix: {prefix}")
            return prefix

    raise ValueError("Cannot detect EEG prefix in file")


# =============================================================================
# SUBJECT LOADER
# =============================================================================


def load_subject(file_path, labels, subject_id):
    """
    Load all EEG trials for a single subject.

    -------------------------------------------------------------------------
    Purpose
    -------------------------------------------------------------------------
    Each subject has 15 EEG recordings corresponding to 15 emotional stimuli.

    This function:
    - extracts all EEG trials
    - assigns correct labels
    - structures data into ML-ready format

    -------------------------------------------------------------------------
    EEG Signal Format
    -------------------------------------------------------------------------
    Each trial:
        signal shape = (62, T)

    Where:
        62 = EEG channels
        T  = time samples (variable length)

    -------------------------------------------------------------------------
    Output Format
    -------------------------------------------------------------------------
    List of dictionaries:
    {
        "signal": np.ndarray (62, T),
        "label": int,
        "subject": int,
        "trial": int
    }
    """

    logger.info(f"Processing subject {subject_id}: {file_path}")

    mat = loadmat(file_path, simplify_cells=True)
    prefix = detect_prefix(mat)

    samples = []

    # SEED dataset: labels are shared across subjects
    labels_flat = labels.flatten()

    for i in range(1, N_TRIALS + 1):
        key = f"{prefix}{EEG_KEY_PATTERN}{i}"

        # Ensure EEG trial exists
        if key not in mat:
            raise KeyError(f"Missing EEG key: {key} in {file_path}")

        signal = np.array(mat[key])

        label = int(labels_flat[i - 1])

        if signal.ndim != 2:
            raise ValueError(f"Invalid signal shape: {signal.shape}")

        samples.append(
            {
                "signal": signal,
                "label": label,
                "subject": subject_id,
                "trial": i,
            }
        )

    logger.info(f"Loaded {len(samples)} trials for subject {subject_id}")

    return samples


# =============================================================================
# FULL DATASET BUILDER
# =============================================================================


def build_seed_dataset(folder_path):
    """
    Build full SEED EEG dataset from directory.

    -------------------------------------------------------------------------
    Purpose
    -------------------------------------------------------------------------
    Aggregates all subjects into a unified dataset for ML training.

    -------------------------------------------------------------------------
    Pipeline Overview
    -------------------------------------------------------------------------
    1. Load labels
    2. Detect all subject .mat files
    3. Iterate subject-by-subject
    4. Extract EEG trials
    5. Combine into single dataset

    -------------------------------------------------------------------------
    Output Format
    -------------------------------------------------------------------------
    List of dictionaries:
    {
        "signal": (62, T),
        "label": int,
        "subject": int,
        "trial": int
    }

    -------------------------------------------------------------------------
    Design Choice
    -------------------------------------------------------------------------
    - Failures in one subject do NOT stop the pipeline
    - Ensures robustness for large-scale EEG datasets
    """

    logger.info(f"Starting dataset build from: {folder_path}")

    label_path = os.path.join(folder_path, LABEL_FILE)
    labels = load_labels(label_path)

    dataset = []

    files = [
        f for f in os.listdir(folder_path) if f.endswith(".mat") and f != LABEL_FILE
    ]

    logger.info(f"Found {len(files)} subject files")

    for idx, file in enumerate(sorted(files), 1):

        subject_id = int(file.split("_")[0])
        file_path = os.path.join(folder_path, file)

        logger.info(f"[{idx}/{len(files)}] Loading subject {subject_id}")

        try:
            samples = load_subject(file_path, labels, subject_id)
            dataset.extend(samples)

        except Exception as e:
            logger.error(f"Failed subject {subject_id}: {e}")

    logger.info(f"Dataset built successfully ✅: {len(dataset)} samples")

    return dataset
