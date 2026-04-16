"""
===============================================================================
EEG → LLM Emotion Inference Pipeline (LM Studio Compatible)
===============================================================================

PROJECT CONTEXT
---------------
This module is part of a neuromarketing / affective computing pipeline using
the SEED EEG dataset.

Goal:
-----
Transform raw EEG signals (62 channels × time) into a structured representation
that can be interpreted by a Large Language Model (LLM) for emotion
classification.

WHY THIS APPROACH EXISTS
------------------------
LLMs cannot directly process raw EEG signals because:
- EEG is high-dimensional continuous time-series data
- No semantic structure exists in raw signals
- Context window limitations make raw ingestion impossible

Therefore, we use a 3-stage abstraction:

    EEG Signal
        ↓
    Numerical Feature Extraction
        ↓
    Natural Language Summary
        ↓
    LLM Classification (LM Studio)

CURRENT LIMITATION
------------------
This is a BASELINE system:
- Uses global statistical features only
- Does NOT yet use frequency bands (alpha/beta/theta)
- Does NOT preserve spatial channel structure

It is designed for:
- feasibility testing
- LLM benchmarking
- pipeline validation

Future upgrade path:
- channel-wise features
- DE (differential entropy) features (SEED standard)
- transformer-based EEG encoders
===============================================================================
"""

import numpy as np
import requests


# =============================================================================
# 1. EEG FEATURE EXTRACTION (BASELINE STATISTICAL MODEL)
# =============================================================================

def extract_basic_features(signal: np.ndarray) -> dict:
    """
    Convert raw EEG trial into compact statistical descriptors.

    INPUT
    -----
    signal : np.ndarray
        EEG signal of shape (62, T)

    OUTPUT
    ------
    dict of aggregated metrics

    WHY THIS STEP EXISTS
    ---------------------
    Raw EEG is too large and unstructured for LLMs.
    We compress it into interpretable global statistics.
    """

    signal = signal.astype(np.float32)

    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "max": float(np.max(signal)),
        "min": float(np.min(signal)),
        "energy": float(np.mean(signal ** 2)),
    }


# =============================================================================
# 2. EEG → NATURAL LANGUAGE REPRESENTATION
# =============================================================================

def build_eeg_summary(features: dict) -> str:
    """
    Convert numerical EEG features into structured natural language prompt.

    WHY THIS STEP EXISTS
    --------------------
    LLMs perform significantly better on structured semantic input than raw
    numeric vectors.

    This acts as a "neural-to-language interface".
    """

    return f"""
You are analyzing EEG signals from a neuroscience experiment.

Context:
- The EEG was recorded while a subject watched a film clip.
- The goal is to infer the emotional state.

Extracted EEG Features:
- Mean amplitude: {features['mean']:.6f}
- Signal variability (std): {features['std']:.6f}
- Maximum amplitude: {features['max']:.6f}
- Minimum amplitude: {features['min']:.6f}
- Signal energy: {features['energy']:.6f}

Task:
Classify the emotional state of the subject.

Allowed labels ONLY:
- positive
- neutral
- negative

Return ONLY the label.
""".strip()


# =============================================================================
# 3. LM STUDIO CLIENT (OPENAI-COMPATIBLE API)
# =============================================================================

class LMStudioClient:
    """
    Minimal client for LM Studio local inference server.

    LM Studio exposes an OpenAI-compatible API:

        http://localhost:1234/v1/chat/completions

    This class wraps that API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1/chat/completions",
        model: str = "local-model"
    ):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Send prompt to LM Studio and return model response.
        """

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict EEG emotion classification model. "
                        "You only output one label: positive, neutral, or negative."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        }

        response = requests.post(self.base_url, json=payload)

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()


# =============================================================================
# 4. LLM INFERENCE WRAPPER
# =============================================================================

def llm_predict(prompt: str, llm_client: LMStudioClient) -> str:
    """
    Send prompt to LLM and return predicted emotion label.
    """

    return llm_client.generate(prompt)


# =============================================================================
# 5. FULL PIPELINE
# =============================================================================

def predict_emotion_from_eeg(signal: np.ndarray, llm_client: LMStudioClient) -> dict:
    """
    FULL EEG → LLM PIPELINE

    Steps:
    ------
    1. Extract statistical EEG features
    2. Convert features into structured text prompt
    3. Send prompt to LM Studio model
    4. Return prediction

    OUTPUT
    ------
    dict containing:
        - features (numeric EEG summary)
        - prompt (LLM input)
        - prediction (LLM output)
    """

    features = extract_basic_features(signal)
    prompt = build_eeg_summary(features)
    prediction = llm_predict(prompt, llm_client)

    return {
        "features": features,
        "prompt": prompt,
        "prediction": prediction
    }


# =============================================================================
# 6. QUICK TEST EXAMPLE
# =============================================================================

if __name__ == "__main__":
    """
    Simple sanity test using random EEG-like data.
    """

    fake_signal = np.random.randn(62, 1000)

    client = LMStudioClient(model="llama-3")

    result = predict_emotion_from_eeg(fake_signal, client)

    print("\n================ RESULT ================\n")
    print("PREDICTION:", result["prediction"])