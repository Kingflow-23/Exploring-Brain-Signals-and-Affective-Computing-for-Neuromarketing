"""
EEG to LLM Emotion Inference Pipeline.

Bridges EEG signals to Large Language Models for emotion classification.

Pipeline:
    EEG Signal → Feature Extraction → Natural Language Prompt → LLM → Emotion Label

Key Features:
    - Extracts neuroscience-relevant EEG features (bandpower, ratios, entropy)
    - Converts features into human-readable text prompts
    - Queries local LM Studio LLM for emotion prediction
    - Supports emotion labels: positive, neutral, negative

Design:
    - Uses EEG-specific features rather than raw statistics
    - Compatible with local LLM inference (LM Studio)
    - Deterministic feature extraction for reproducibility
    - Semantic understanding of EEG patterns through language models

Input:
    EEG signal of shape (62, T) where:
        62 = EEG channels (SEED montage)
        T  = time samples
"""

import numpy as np
import requests

from scipy.signal import welch
from config import (
    DEFAULT_FS,
    ALLOWED_LABELS,
    DEFAULT_MODEL,
    PAIR_INDICES,
    FRONTAL_PAIRS_IDX,
)

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


def compute_faa(signal, fs, pair_indices):
    """
    Frontal Alpha Asymmetry (FAA)

    Idea:
    FAA = log(left_alpha_power) - log(right_alpha_power)

    Positive FAA → more left activity → positive valence
    Negative FAA → right dominance → negative valence
    """

    faa_values = []

    for left_i, right_i in pair_indices:

        left = signal[left_i]
        right = signal[right_i]

        left_alpha = compute_bandpower(left, fs, 8, 13)
        right_alpha = compute_bandpower(right, fs, 8, 13)

        faa = np.log(left_alpha + 1e-8) - np.log(right_alpha + 1e-8)
        faa_values.append(faa)

    return float(np.mean(faa_values))


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

    # =====================================================
    # BANDPOWER
    # =====================================================
    theta_vals, alpha_vals, beta_vals, gamma_vals = [], [], [], []

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

    theta_ratio = theta / total
    alpha_ratio = alpha / total
    beta_ratio = beta / total
    gamma_ratio = gamma / total

    # =====================================================
    # SIMPLE GLOBAL ACTIVITY
    # =====================================================
    activity = float(np.std(signal))

    # =====================================================
    # ENTROPY (COMPLEXITY)
    # =====================================================
    prob = np.abs(signal)
    prob = prob / (np.sum(prob) + 1e-8)
    entropy = float(-np.sum(prob * np.log(prob + 1e-8)))

    # =====================================================
    # FAA (ONLY FINAL VALENCE SCORE)
    # =====================================================
    faa_valence_index = compute_faa(signal, fs, PAIR_INDICES)

    return {
        # spectral
        "theta_ratio": theta_ratio,
        "alpha_ratio": alpha_ratio,
        "beta_ratio": beta_ratio,
        "gamma_ratio": gamma_ratio,
        # simple dynamics
        "activity": activity,
        "entropy": entropy,
        # valence
        "faa": faa_valence_index,
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
You are an EEG emotion classification system.

Your task is to classify emotional state from EEG spectral features.

You MUST choose exactly one label:
positive, neutral, negative

---

FEATURES:

Spectral:
- theta: {features["theta_ratio"]:.4f}
- alpha: {features["alpha_ratio"]:.4f}
- beta: {features["beta_ratio"]:.4f}
- gamma: {features["gamma_ratio"]:.4f}

Brain state:
- activity: {features["activity"]:.4f}
- entropy: {features["entropy"]:.4f}

Valence:
- FAA (frontal asymmetry): {features["faa"]:.4f}

---

EEG BAND INTERPRETATION (context only):

- theta: drowsiness / internal cognition
- alpha: relaxed wakefulness / calm state
- beta: active cognition / attention / alertness
- gamma: high cognitive integration / load

These describe brain state only and do NOT determine emotion directly.

---

STEP 1 — VALENCE ESTIMATION (FAA ONLY)

FAA defines emotional direction:

- FAA > +0.3 → positive valence
- FAA < -0.3 → negative valence
- otherwise → undecided

Store result as:
VALENCE = positive / negative / undecided

---

STEP 2 — AROUSAL (CONTINUOUS CONFIDENCE SCALE)

Compute:

arousal_score =
    (activity - 0.75) +
    (entropy - 10.0)

Clamp interpretation:

- arousal_score < -0.3 → low arousal
- -0.3 to 0.3 → medium arousal
- > 0.3 → high arousal

---

STEP 3 — SPECTRAL CONTEXT (MODULATOR ONLY)

Compute qualitative context:

- beta/gamma dominant → engagement context
- alpha dominant → relaxed context
- theta dominant → disengaged context

This ONLY adjusts confidence:

- strong engagement → reinforces FAA decision
- strong relaxation → pushes toward neutral if FAA is weak

DO NOT convert spectral into votes.

---

STEP 4 — FINAL DECISION RULE

CASE A:
If |FAA| ≥ 0.15 AND arousal is medium or high:
    → follow FAA direction (positive/negative)

CASE B:
If FAA is weak AND arousal is low:
    → neutral

CASE C:
If FAA is weak AND spectral shows strong engagement:
    → neutral (no clear valence signal)

CASE D:
If FAA and spectral strongly conflict:
    → neutral

CASE E:
Otherwise:
    → follow FAA direction with reduced confidence bias
    
---

IMPORTANT RULES:

- FAA is direction (valence axis)
- arousal is confidence, NOT a gate
- spectral is context only (never voting system)
- neutral = uncertainty OR low emotional drive
- no single feature dominates always

---

OUTPUT FORMAT:
Return ONLY one word:
positive
neutral
negative
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
        temperature: float = 0.0,
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
                    "content": prompt,
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

    raw = llm_client.generate(prompt).strip().lower()

    if raw in ALLOWED_LABELS:
        prediction = raw
    else:
        prediction = "neutral"  # fallback safe default

    return {
        "features": features,
        "prompt": prompt,
        "prediction": prediction,
    }
