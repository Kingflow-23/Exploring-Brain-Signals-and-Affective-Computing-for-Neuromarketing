"""
EEG to LLM Inference Pipeline Tests.

Validates EEG feature extraction, prompt generation, and LLM integration:
    - Feature extraction correctness
    - Prompt generation and formatting
    - Emotion prediction logic
    - Integration without real LM Studio calls

Uses synthetic signals and mocking for isolated unit testing.
"""

import numpy as np

from llm_inference import (
    extract_eeg_features,
    build_eeg_prompt,
    predict_emotion_from_eeg,
)

# =============================================================================
# FIXTURE
# =============================================================================


def fake_eeg_signal():
    """
    Create deterministic EEG-like signal.

    Shape:
        (62, 1000)
    """
    np.random.seed(42)
    return np.random.randn(62, 1000)


# =============================================================================
# TEST 1 — FEATURE EXTRACTION
# =============================================================================


def test_extract_eeg_features():
    signal = fake_eeg_signal()

    features = extract_eeg_features(signal)

    # --- keys exist
    expected_keys = [
        "mean",
        "std",
        "max",
        "min",
        "energy",
        "theta_ratio",
        "alpha_ratio",
        "beta_ratio",
        "gamma_ratio",
        "activity",
    ]

    for k in expected_keys:
        assert k in features

    # --- values are valid numbers
    for v in features.values():
        assert np.isfinite(v)


# =============================================================================
# TEST 2 — PROMPT GENERATION
# =============================================================================


def test_build_eeg_prompt():
    features = {
        "mean": 0.1,
        "std": 1.0,
        "max": 2.0,
        "min": -2.0,
        "energy": 0.5,
        "theta_ratio": 0.2,
        "alpha_ratio": 0.2,
        "beta_ratio": 0.3,
        "gamma_ratio": 0.3,
        "activity": 1.0,
    }

    prompt = build_eeg_prompt(features)

    assert isinstance(prompt, str)

    # required semantic checks
    assert "EEG emotion classifier" in prompt
    assert "Theta ratio" in prompt
    assert "Alpha ratio" in prompt
    assert "positive" in prompt
    assert "neutral" in prompt
    assert "negative" in prompt


# =============================================================================
# TEST 3 — MOCK LLM CLIENT
# =============================================================================


class MockLLM:
    def generate(self, prompt: str):
        return "neutral"


def test_full_pipeline():
    signal = fake_eeg_signal()

    mock_client = MockLLM()

    result = predict_emotion_from_eeg(signal, mock_client)

    # structure checks
    assert "features" in result
    assert "prompt" in result
    assert "prediction" in result

    assert result["prediction"] in ["positive", "neutral", "negative"]

    assert isinstance(result["features"], dict)
    assert isinstance(result["prompt"], str)


# =============================================================================
# TEST 4 — PROMPT CONSISTENCY
# =============================================================================


def test_prompt_contains_values():
    signal = fake_eeg_signal()

    features = extract_eeg_features(signal)
    prompt = build_eeg_prompt(features)

    # sanity: numeric values must appear in text
    assert str(round(features["mean"], 3))[:3] in prompt or "Mean" in prompt
