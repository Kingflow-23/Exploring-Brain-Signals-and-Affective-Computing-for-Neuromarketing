"""
Statistical Window Size Analysis for EEG Feature Stability.

Goal:
    Analyze how EEG feature distributions change across window sizes
    WITHOUT any LLM inference or classification.

Pipeline:
    EEG Signal → Windowing → Feature Extraction → Statistical Analysis

Outputs:
    - Feature variance vs window size
    - Feature stability scores
    - Entropy trends
    - Cross-window similarity
    - Subject-wise variability
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from typing import Dict, List

from seed_loader import load_subject, load_labels
from llm_inference import extract_eeg_features
from config import DATASET_DIR, OUTPUT_DIR, LABEL_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WINDOW_ANALYSIS")


# =========================================================
# CONFIG
# =========================================================

WINDOW_SIZES = [200, 400, 600, 800, 1000, 1200]


# =========================================================
# DATA LOADING
# =========================================================


def load_subject_data(subject_id: int):
    label_path = os.path.join(DATASET_DIR, LABEL_FILE)
    labels = load_labels(label_path)

    subject_files = [
        f for f in os.listdir(DATASET_DIR) if f.startswith(f"{subject_id}_")
    ]

    if not subject_files:
        return []

    file_path = os.path.join(DATASET_DIR, subject_files[0])
    return load_subject(file_path, labels, subject_id)


# =========================================================
# WINDOWING
# =========================================================


def create_windows(signal: np.ndarray, window_size: int, step=None):
    if step is None:
        step = window_size // 2

    windows = []
    T = signal.shape[1]

    for start in range(0, T - window_size + 1, step):
        windows.append(signal[:, start : start + window_size])

    return np.array(windows) if windows else np.empty((0, *signal.shape))


# =========================================================
# FEATURE EXTRACTION (UNCHANGED)
# =========================================================


def extract_features_per_window(windows: np.ndarray):
    features = []

    for w in windows:
        try:
            f = extract_eeg_features(w)
            features.append(list(f.values()))
        except:
            features.append(np.zeros(18))

    return np.array(features)


# =========================================================
# STATISTICS CORE
# =========================================================


def feature_statistics(feature_matrix: np.ndarray) -> Dict:
    """
    Compute statistical descriptors of features across windows.
    """

    if len(feature_matrix) == 0:
        return {}

    return {
        "mean": np.mean(feature_matrix, axis=0),
        "std": np.std(feature_matrix, axis=0),
        "var": np.var(feature_matrix, axis=0),
        "entropy": np.mean(
            -np.sum(
                np.abs(feature_matrix)
                / (np.sum(np.abs(feature_matrix), axis=1, keepdims=True) + 1e-8)
                * np.log(
                    np.abs(feature_matrix)
                    / (np.sum(np.abs(feature_matrix), axis=1, keepdims=True) + 1e-8)
                ),
                axis=1,
            )
        ),
    }


def stability_score(feature_matrix: np.ndarray) -> float:
    """
    Lower variance across windows = higher stability.
    """

    if len(feature_matrix) < 2:
        return 0.0

    return float(1.0 / (np.mean(np.var(feature_matrix, axis=0)) + 1e-8))


def window_similarity(feature_matrix: np.ndarray) -> float:
    """
    Average cosine similarity between consecutive windows.
    """

    if len(feature_matrix) < 2:
        return 0.0

    sims = []

    for i in range(len(feature_matrix) - 1):
        a = feature_matrix[i]
        b = feature_matrix[i + 1]

        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
        sims.append(np.dot(a, b) / denom)

    return float(np.mean(sims))


# =========================================================
# MAIN ANALYSIS
# =========================================================


def analyze_subject(subject_id: int) -> pd.DataFrame:
    logger.info(f"Analyzing subject {subject_id}")

    trials = load_subject_data(subject_id)
    results = []

    for trial_idx, trial in enumerate(trials):

        signal = trial["signal"]

        for ws in WINDOW_SIZES:

            logger.info(f"Subject {subject_id} - Trial {trial_idx} - Window Size {ws}")

            windows = create_windows(signal, ws)

            if len(windows) == 0:
                continue

            features = extract_features_per_window(windows)

            stats = feature_statistics(features)

            results.append(
                {
                    "subject": subject_id,
                    "trial": trial_idx,
                    "window_size": ws,
                    "n_windows": len(windows),
                    # CORE METRICS
                    "stability": stability_score(features),
                    "similarity": window_similarity(features),
                    # FEATURE VARIABILITY
                    "feature_variance_mean": np.mean(np.var(features, axis=0)),
                    "feature_variance_max": np.max(np.var(features, axis=0)),
                    # ENTROPY STRUCTURE
                    "entropy_mean": stats.get("entropy", 0.0),
                }
            )

    return pd.DataFrame(results)


# =========================================================
# MULTI SUBJECT
# =========================================================


def run(subject_ids=None):
    if subject_ids is None:
        subject_ids = list(range(1, 16))

    all_data = []

    for sid in subject_ids:
        df = analyze_subject(sid)
        all_data.append(df)

    results = pd.concat(all_data, ignore_index=True)

    out_dir = os.path.join(OUTPUT_DIR, "window_analysis")
    os.makedirs(out_dir, exist_ok=True)

    results.to_csv(os.path.join(out_dir, "feature_window_stats.csv"), index=False)

    # Summary plot
    summary = results.groupby("window_size").mean(numeric_only=True)

    plt.figure()
    plt.plot(summary.index, summary["stability"], marker="o")
    plt.title("Feature Stability vs Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Stability Score")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "stability.png"))

    plt.figure()
    plt.plot(summary.index, summary["similarity"], marker="o")
    plt.title("Window Similarity vs Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Cosine Similarity")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "similarity.png"))

    logger.info("DONE ✅")


if __name__ == "__main__":
    run(subject_ids=[1])
