"""
===============================================================================
EEG PREPROCESSING TESTS (SEED PIPELINE)
===============================================================================

Goal:
-----
Ensure preprocessing pipeline correctly:
1. Normalizes EEG signals
2. Splits into valid sliding windows
3. Produces ML-ready tensor shapes

We avoid complex mocking and focus on structural correctness only.
===============================================================================
"""

import numpy as np

from preprocessing import (
    zscore_normalize,
    sliding_window,
    preprocess_trial,
    preprocess_dataset,
)

# =============================================================================
# TEST 1 — Z-SCORE NORMALIZATION
# =============================================================================


def test_zscore_normalization():
    """
    Verify:
    - output shape unchanged
    - mean approx 0 per channel after normalization
    """

    signal = np.random.randn(62, 1000)

    norm = zscore_normalize(signal)

    assert norm.shape == signal.shape

    # mean should be close to 0 per channel
    assert np.allclose(np.mean(norm, axis=1), 0, atol=1e-5)


# =============================================================================
# TEST 2 — SLIDING WINDOW
# =============================================================================


def test_sliding_window_shape():
    """
    Verify:
    - correct number of windows
    - correct window shape
    """

    signal = np.random.randn(62, 1000)

    window_size = 400
    step_size = 200

    windows = sliding_window(signal, window_size, step_size)

    expected_n = (1000 - window_size) // step_size + 1

    assert windows.shape[1:] == (62, window_size)
    assert windows.shape[0] == expected_n


# =============================================================================
# TEST 3 — FULL TRIAL PREPROCESSING
# =============================================================================


def test_preprocess_trial():
    """
    Verify full pipeline:
    signal -> normalized -> windowed
    """

    signal = np.random.randn(62, 1000)

    windows = preprocess_trial(signal, normalize=True)

    assert windows.ndim == 3
    assert windows.shape[1] == 62
    assert windows.shape[2] == 400  # default window size


# =============================================================================
# TEST 4 — DATASET LEVEL PIPELINE
# =============================================================================


def test_preprocess_dataset():
    """
    Verify dataset transformation preserves structure.
    """

    fake_dataset = [
        {"signal": np.random.randn(62, 1000), "label": 1, "subject": 1, "trial": 1},
        {"signal": np.random.randn(62, 1200), "label": 0, "subject": 1, "trial": 2},
    ]

    processed = preprocess_dataset(fake_dataset)

    assert len(processed) == 2

    for item in processed:
        assert "windows" in item
        assert item["windows"].ndim == 3
        assert item["windows"].shape[1] == 62
        assert item["label"] in [-1, 0, 1]
