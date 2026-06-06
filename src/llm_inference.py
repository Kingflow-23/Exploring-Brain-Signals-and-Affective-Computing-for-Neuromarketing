"""
EEG to LLM Emotion Inference Pipeline.

Bridges EEG signals to Large Language Models for emotion classification through feature-based prompting.

Pipeline:
    EEG Signal → Feature Extraction → Natural Language Prompt → LLM → Emotion Label

Key Features:
    - Extracts neuroscience-relevant EEG features (bandpower, entropy, asymmetry)
    - Converts features into structured natural language prompts
    - Queries local LLM services for emotion prediction via REST API
    - Supports emotion labels: positive, neutral, negative
    - Deterministic and reproducible feature extraction

Design Philosophy:
    - NO rule-based decision logic inside prompt
    - LLM is responsible for multimodal integration
    - Features are descriptive, not prescriptive
    - Emphasis on neuroscience interpretability

Input:
    EEG signal of shape (62, T) where:
        62 = EEG channels (SEED 10-20 montage)
        T  = time samples at 200 Hz sampling rate

Output:
    Predicted emotion label: 'positive', 'neutral', or 'negative'
"""

import numpy as np
import requests

from scipy.signal import welch

from config import (
    DEFAULT_FS,
    ALLOWED_LABELS,
    DEFAULT_MODEL,
    FRONTAL_PAIRS_IDX,
)

# =============================================================================
# BANDPOWER
# =============================================================================


def compute_bandpower(signal_1d: np.ndarray, fs: int, low: float, high: float) -> float:

    freqs, psd = welch(signal_1d, fs=fs, nperseg=min(256, len(signal_1d)))

    mask = (freqs >= low) & (freqs <= high)

    if not np.any(mask):
        return 0.0

    return float(np.trapezoid(psd[mask], freqs[mask]))


# =============================================================================
# FAA (3 frontal asymmetry features)
# =============================================================================


def compute_faa(signal, fs, frontal_pairs_idx):
    """
    Compute 3 explicit frontal FAA features.
    Order is now guaranteed by explicit mapping.
    """

    eps = 1e-8

    FAA_KEYS = [
        "faa_f3_f4",
        "faa_f7_f8",
        "faa_fp1_fp2",
    ]

    faa_vals = {}

    for key, (left_i, right_i) in zip(FAA_KEYS, frontal_pairs_idx):

        left = signal[left_i]
        right = signal[right_i]

        left_alpha = compute_bandpower(left, fs, 8, 13)
        right_alpha = compute_bandpower(right, fs, 8, 13)

        faa_vals[key] = np.log(left_alpha + eps) - np.log(right_alpha + eps)

    return faa_vals


# =============================================================================
# FEATURE EXTRACTION (18-feature model)
# =============================================================================


def extract_eeg_features(signal: np.ndarray, fs: int = DEFAULT_FS) -> dict:
    signal = signal.astype(np.float32)

    # -----------------------
    # GLOBAL SPECTRAL STATE
    # -----------------------
    theta_vals, alpha_vals, beta_vals, gamma_vals = [], [], [], []

    for ch in signal:
        theta_vals.append(compute_bandpower(ch, fs, 4, 8))
        alpha_vals.append(compute_bandpower(ch, fs, 8, 13))
        beta_vals.append(compute_bandpower(ch, fs, 13, 30))
        gamma_vals.append(compute_bandpower(ch, fs, 30, 45))

    theta = np.mean(theta_vals)
    alpha = np.mean(alpha_vals)
    beta = np.mean(beta_vals)
    gamma = np.mean(gamma_vals)

    eps = 1e-8
    total = theta + alpha + beta + gamma + eps

    theta_ratio = theta / total
    alpha_ratio = alpha / total
    beta_ratio = beta / total
    gamma_ratio = gamma / total

    # -----------------------
    # REGIONAL FEATURES
    # -----------------------
    frontal = signal[:20]
    temporal = signal[20:40]
    occipital = signal[40:]

    def region_band(region, low, high):
        return np.mean([compute_bandpower(ch, fs, low, high) for ch in region])

    frontal_alpha = region_band(frontal, 8, 13)
    frontal_beta = region_band(frontal, 13, 30)
    frontal_gamma = region_band(frontal, 30, 45)

    temporal_alpha = region_band(temporal, 8, 13)
    temporal_beta = region_band(temporal, 13, 30)

    occipital_alpha = region_band(occipital, 8, 13)

    # -----------------------
    # FAA (3 features)
    # -----------------------
    faa_vals = compute_faa(signal, fs, FRONTAL_PAIRS_IDX)

    # -----------------------
    # COMPLEXITY
    # -----------------------
    activity = float(np.std(signal))

    power = signal**2
    prob = power / (np.sum(power) + eps)
    entropy = -np.sum(prob * np.log(prob + eps))

    # -----------------------
    # DERIVED RATIOS
    # -----------------------
    beta_alpha_ratio = beta / (alpha + eps)
    gamma_beta_ratio = gamma / (beta + eps)
    frontal_occipital_alpha_ratio = frontal_alpha / (occipital_alpha + eps)

    return {
        # global
        "theta_ratio": theta_ratio,
        "alpha_ratio": alpha_ratio,
        "beta_ratio": beta_ratio,
        "gamma_ratio": gamma_ratio,
        # FAA (3)
        **faa_vals,
        # regional
        "frontal_alpha": frontal_alpha,
        "frontal_beta": frontal_beta,
        "frontal_gamma": frontal_gamma,
        "temporal_alpha": temporal_alpha,
        "temporal_beta": temporal_beta,
        "occipital_alpha": occipital_alpha,
        # complexity
        "activity": activity,
        "entropy": entropy,
        # ratios
        "beta_alpha_ratio": beta_alpha_ratio,
        "gamma_beta_ratio": gamma_beta_ratio,
        "frontal_occipital_alpha_ratio": frontal_occipital_alpha_ratio,
    }


# =============================================================================
# PROMPT (NO RULE ENGINE — PURE NEUROSCIENCE CONTEXT)
# =============================================================================


def build_eeg_prompt(features: dict) -> str:
    return f"""
You are an EEG emotion classification expert.

Your task is to classify emotional state into one of:

positive
neutral
negative

You must use all features jointly.

-----------------------
EEG FEATURES
-----------------------

Global spectral state:
- theta_ratio = {features["theta_ratio"]:.4f}
- alpha_ratio = {features["alpha_ratio"]:.4f}
- beta_ratio  = {features["beta_ratio"]:.4f}
- gamma_ratio = {features["gamma_ratio"]:.4f}

Frontal asymmetry:
- FAA(Fp1-Fp2) = {features["faa_fp1_fp2"]:.4f}
- FAA(F3-F4)   = {features["faa_f3_f4"]:.4f}
- FAA(F7-F8)   = {features["faa_f7_f8"]:.4f}

Regional activity:
- frontal_alpha = {features["frontal_alpha"]:.4f}
- frontal_beta  = {features["frontal_beta"]:.4f}
- frontal_gamma = {features["frontal_gamma"]:.4f}

- temporal_alpha = {features["temporal_alpha"]:.4f}
- temporal_beta  = {features["temporal_beta"]:.4f}

- occipital_alpha = {features["occipital_alpha"]:.4f}

Complexity:
- activity = {features["activity"]:.4f}
- entropy  = {features["entropy"]:.4f}

Derived ratios:
- beta_alpha_ratio = {features["beta_alpha_ratio"]:.4f}
- gamma_beta_ratio = {features["gamma_beta_ratio"]:.4f}
- frontal_occipital_alpha_ratio = {features["frontal_occipital_alpha_ratio"]:.4f}

-----------------------
NEUROSCIENCE CONTEXT
-----------------------

Frontal asymmetry (FAA):
- reflects hemispheric imbalance in affective processing
- positive → left dominance (approach-related affect tendency)
- negative → right dominance (withdrawal-related affect tendency)
- magnitude reflects strength of lateralization

Spectral bands:
- theta → internal processing / low vigilance
- alpha → inhibition / relaxed wakefulness
- beta → cognitive engagement / attention
- gamma → integrative high-level processing

Regional activity:
- frontal → emotional regulation and executive control
- temporal → affective and semantic processing
- occipital → visual baseline and relaxation state

Derived ratios:
- beta/alpha → cognitive engagement vs relaxation balance
- gamma/beta → integrative processing load
- frontal/occipital alpha → affective control vs baseline relaxation

-----------------------
INSTRUCTIONS
-----------------------

- Integrate all features holistically
- Do NOT apply fixed thresholds or rules
- No feature alone determines the label
- Allow uncertainty when signals conflict
- Use FAA as important but not exclusive evidence

-----------------------
OUTPUT
-----------------------

Return ONLY one label:
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
