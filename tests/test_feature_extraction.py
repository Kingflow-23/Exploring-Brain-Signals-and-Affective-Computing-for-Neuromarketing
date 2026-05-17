import numpy as np

from feature_extraction import extract_features


def create_dummy_window(n_channels=62, length=1000):
    """
    Create synthetic EEG-like signal.

    Shape:
        (62, W)
    """
    rng = np.random.RandomState(42)

    # simulate EEG: low-frequency + noise
    signal = rng.randn(n_channels, length) * 10e-6

    # add slow oscillation
    t = np.linspace(0, 1, length)
    for ch in range(n_channels):
        signal[ch] += 10e-6 * np.sin(2 * np.pi * 10 * t)

    return signal


def test_feature_shape():
    window = create_dummy_window()

    feats = extract_features(window)

    print("Feature shape:", feats.shape)

    assert feats.ndim == 1, "Features must be 1D vector"


def test_no_nan_inf():
    window = create_dummy_window()

    feats = extract_features(window)

    assert np.all(np.isfinite(feats)), "NaN or Inf detected"


def test_determinism():
    window = create_dummy_window()

    f1 = extract_features(window)
    f2 = extract_features(window)

    assert np.allclose(f1, f2), "Feature extraction is not deterministic"


def test_multiple_runs():
    for _ in range(5):
        window = create_dummy_window()
        feats = extract_features(window)

        assert feats.shape[0] > 0


def test_minimal_signal():
    # edge case: very short signal
    window = np.random.randn(62, 300)

    feats = extract_features(window)

    assert feats.shape[0] > 0
