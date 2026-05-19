"""
===============================================================================
EEG → LLM EMOTION INFERENCE PIPELINE (LM STUDIO READY)
===============================================================================

AUTHOR PURPOSE
------------------------------------------------------------------------------
This module converts EEG signals into interpretable features, transforms them
into natural-language prompts, and asks a Large Language Model (LLM) to predict
emotion labels.

This is designed for benchmarking against traditional EEG ML models.

Target dataset:
    SEED Dataset

Emotion labels:
    - positive
    - neutral
    - negative

===============================================================================
WHY THIS FILE EXISTS
------------------------------------------------------------------------------

Raw EEG data cannot be directly given to an LLM because:

1. EEG is numeric time-series data
   Shape example:
        (62 channels, 47001 samples)

2. LLMs expect text / semantic structure

3. Context window limitations make raw EEG impossible to feed directly

Therefore we build a bridge:

    EEG Signal
        ↓
    Signal Features
        ↓
    Human-readable Summary
        ↓
    LLM Prediction

===============================================================================
WHY THIS IS SCIENTIFICALLY BETTER THAN BASIC STATS ONLY
------------------------------------------------------------------------------

Instead of only:

- mean
- std
- max
- min

We include EEG-relevant neuroscience features:

1. Band Power
    - theta  (4–8 Hz)
    - alpha  (8–13 Hz)
    - beta   (13–30 Hz)
    - gamma  (30–45 Hz)

2. Relative Band Ratios

3. Signal Variability

4. Hjorth Activity (basic mobility proxy)

These are much more meaningful for emotion recognition than plain mean values.

===============================================================================
EXPECTED INPUT
------------------------------------------------------------------------------

signal: np.ndarray

Shape:
    (62, T)

Where:
    62 = EEG electrodes
    T  = time samples

Sampling rate expected:
    200 Hz (SEED preprocessed version)

===============================================================================
OUTPUT
------------------------------------------------------------------------------

{
    "features": dict,
    "prompt": str,
    "prediction": str
}

===============================================================================
"""

import numpy as np
import requests

from scipy.signal import welch
from config import DEFAULT_FS, ALLOWED_LABELS, DEFAULT_MODEL

# =============================================================================
# EEG FEATURE EXTRACTION
# =============================================================================


def compute_bandpower(
    signal_1d: np.ndarray,
    fs: int,
    low: float,
    high: float,
) -> float:
    """
    Compute spectral power inside a frequency band.

    PARAMETERS
    ----------
    signal_1d : np.ndarray
        Single EEG channel shape (T,)

    fs : int
        Sampling rate

    low : float
        Lower frequency bound

    high : float
        Upper frequency bound

    RETURNS
    -------
    float
        Integrated power in selected band

    WHY THIS MATTERS
    ----------------
    Emotion-related EEG states often differ in band activity:

    - alpha: calm / relaxation
    - beta: attention / arousal
    - theta: memory / engagement
    - gamma: complex cognition
    """

    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.trapezoid(psd[mask], freqs[mask]))


def extract_eeg_features(
    signal: np.ndarray,
    fs: int = DEFAULT_FS,
) -> dict:
    """
    Extract robust EEG summary features.

    INPUT
    -----
    signal : (62, T)

    RETURNS
    -------
    dict
    """

    signal = signal.astype(np.float32)

    # -------------------------------------------------------------------------
    # Global statistics
    # -------------------------------------------------------------------------
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    max_val = float(np.max(signal))
    min_val = float(np.min(signal))
    energy = float(np.mean(signal**2))

    # -------------------------------------------------------------------------
    # Compute average bandpower across channels
    # -------------------------------------------------------------------------
    theta_vals = []
    alpha_vals = []
    beta_vals = []
    gamma_vals = []

    for ch in signal:
        theta_vals.append(compute_bandpower(ch, fs, 4, 8))
        alpha_vals.append(compute_bandpower(ch, fs, 8, 13))
        beta_vals.append(compute_bandpower(ch, fs, 13, 30))
        gamma_vals.append(compute_bandpower(ch, fs, 30, 45))

    theta = float(np.mean(theta_vals))
    alpha = float(np.mean(alpha_vals))
    beta = float(np.mean(beta_vals))
    gamma = float(np.mean(gamma_vals))

    total = theta + alpha + beta + gamma + 1e-8

    # -------------------------------------------------------------------------
    # Relative powers
    # -------------------------------------------------------------------------
    theta_ratio = theta / total
    alpha_ratio = alpha / total
    beta_ratio = beta / total
    gamma_ratio = gamma / total

    # -------------------------------------------------------------------------
    # Hjorth activity proxy
    # -------------------------------------------------------------------------
    activity = float(np.var(signal))

    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "energy": energy,
        "theta_power": theta,
        "alpha_power": alpha,
        "beta_power": beta,
        "gamma_power": gamma,
        "theta_ratio": theta_ratio,
        "alpha_ratio": alpha_ratio,
        "beta_ratio": beta_ratio,
        "gamma_ratio": gamma_ratio,
        "activity": activity,
    }


# =============================================================================
# FEATURE → PROMPT
# =============================================================================


def build_eeg_prompt(features: dict) -> str:
    """
    Convert EEG features into an LLM-readable prompt.

    WHY THIS WORKS
    --------------
    LLMs reason better from structured descriptions than raw vectors.
    """

    return f"""
You are an EEG emotion classifier.

A subject watched a movie clip while EEG was recorded.

Extracted EEG indicators:

Global Signal:
- Mean amplitude: {features["mean"]:.6f}
- Standard deviation: {features["std"]:.6f}
- Maximum amplitude: {features["max"]:.6f}
- Minimum amplitude: {features["min"]:.6f}
- Signal energy: {features["energy"]:.6f}

Frequency Features:
- Theta ratio (4-8 Hz): {features["theta_ratio"]:.6f}
- Alpha ratio (8-13 Hz): {features["alpha_ratio"]:.6f}
- Beta ratio (13-30 Hz): {features["beta_ratio"]:.6f}
- Gamma ratio (30-45 Hz): {features["gamma_ratio"]:.6f}

Brain Dynamics:
- Activity variance: {features["activity"]:.6f}

Task:
Predict emotional state.

Allowed outputs ONLY:
positive
neutral
negative

Return ONLY one label.
""".strip()


# =============================================================================
# LM STUDIO CLIENT
# =============================================================================


class LMStudioClient:
    """
    Client for LM Studio local server.

    Default endpoint:
        http://localhost:1234/v1/chat/completions
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1/chat/completions",
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        """
        Send prompt to LM Studio.
        """

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are a strict neuroscience classifier.\n"
                        "Return only one label: positive, neutral, or negative.\n\n"
                        + prompt
                    ),
                }
            ],
            "temperature": self.temperature,
        }

        response = requests.post(self.base_url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()

        text = data["choices"][0]["message"]["content"].strip().lower()

        # Safety normalize
        for label in ALLOWED_LABELS:
            if label in text:
                return label

        return text


# =============================================================================
# FULL PIPELINE
# =============================================================================


def predict_emotion_from_eeg(
    signal: np.ndarray,
    llm_client: LMStudioClient,
    fs: int = DEFAULT_FS,
) -> dict:
    """
    Full EEG → LLM inference pipeline.

    STEPS
    -----
    1. Extract EEG features
    2. Build prompt
    3. Query LM Studio
    4. Return structured result
    """

    features = extract_eeg_features(signal, fs=fs)

    prompt = build_eeg_prompt(features)

    prediction = llm_client.generate(prompt)

    return {
        "features": features,
        "prompt": prompt,
        "prediction": prediction,
    }
