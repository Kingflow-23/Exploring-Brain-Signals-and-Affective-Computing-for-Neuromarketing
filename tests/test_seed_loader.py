"""
SEED Dataset Loader Tests.

Validates core loader functionality:
    - Label loading from .mat files
    - EEG key prefix detection
    - Subject data loading
    - Complete dataset building

Focus: Structural integrity and basic correctness.
"""

import numpy as np
from scipy.io import savemat

from seed_loader import load_labels, detect_prefix, load_subject, build_seed_dataset

# -----------------------------
# TEST 1: LABEL LOADING
# -----------------------------


def test_load_labels(tmp_path):
    """
    Check that label.mat is loaded correctly.
    """
    file_path = tmp_path / "label.mat"

    savemat(file_path, {"label": np.array([[-1, 0, 1]])})

    labels = load_labels(str(file_path))

    assert labels.shape == (1, 3)
    assert labels[0, 0] == -1


# -----------------------------
# TEST 2: PREFIX DETECTION
# -----------------------------


def test_detect_prefix():
    """
    Check automatic EEG key prefix detection.
    """
    mat = {
        "djc_eeg1": np.random.randn(62, 1000),
        "djc_eeg2": np.random.randn(62, 1000),
    }

    prefix = detect_prefix(mat)

    assert prefix == "djc"


# -----------------------------
# TEST 3: SUBJECT LOADING (core logic)
# -----------------------------


def test_load_subject(tmp_path):
    """
    Check that one subject returns 15 EEG trials.
    """

    file_path = tmp_path / "1_test.mat"

    mat = {}
    for i in range(1, 16):
        mat[f"djc_eeg{i}"] = np.random.randn(62, 1000)

    savemat(file_path, mat)

    labels = np.array([[-1] * 15])

    samples = load_subject(file_path=str(file_path), labels=labels, subject_id=1)

    assert len(samples) == 15
    assert samples[0]["signal"].shape[0] == 62


# -----------------------------
# TEST 4: FULL PIPELINE (minimal)
# -----------------------------


def test_full_pipeline(tmp_path):
    """
    End-to-end test:
    label + subject → dataset
    """

    folder = tmp_path

    # label file
    savemat(folder / "label.mat", {"label": np.array([[-1] * 15])})

    # subject file
    mat = {}
    for i in range(1, 16):
        mat[f"djc_eeg{i}"] = np.random.randn(62, 1000)

    savemat(folder / "1_test.mat", mat)

    dataset = build_seed_dataset(str(folder))

    assert len(dataset) == 15
    assert all("signal" in d for d in dataset)
    assert all(d["signal"].shape[0] == 62 for d in dataset)
