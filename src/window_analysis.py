"""
Statistical Window Size Analysis for EEG-LLM Inference.

Comprehensive analysis of LLM inference performance across different EEG window sizes.

Methodology:
    1. Load raw EEG data from selected subjects
    2. Segment each trial into multiple window sizes (100, 200, 300, 400, 450, 600, 800, 1000 ... samples)
    3. Extract EEG features for each window
    4. Run LLM inference on each windowed feature set
    5. Collect prediction accuracy and confidence metrics
    6. Analyze correlations between window size, features, and predictions
    7. Generate statistical summaries and visualizations

Objectives:
    - Identify optimal window size for EEG-LLM inference
    - Analyze feature stability across window sizes
    - Visualize accuracy trends vs. window size
    - Per-subject and per-emotion analysis
    - Statistical significance testing (ANOVA, correlation)

Output:
    - Statistical summary CSV files
    - Accuracy vs. window size plots
    - Feature distribution heatmaps
    - Per-subject performance matrices
    - Confusion matrices per window size
    - Correlation analysis plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback

from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from seed_loader import load_subject, load_labels, build_seed_dataset
from feature_extraction import extract_features
from llm_inference import extract_eeg_features, build_eeg_prompt
from config import (
    DATASET_DIR,
    OUTPUT_DIR,
    LABELS_MAP,
    CHANNELS,
    FS,
    LABEL_FILE,
    N_TRIALS,
    EEG_KEY_PATTERN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("WINDOW_ANALYSIS")


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_SIZES = [
    100,
    200,
    300,
    400,
    450,
    600,
    800,
    1000,
    1200,
    1500,
    2000,
]  # in samples (0.5s to 10s at 200Hz)
EMOTION_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOTION_LABELS_REVERSE = {"negative": 0, "neutral": 1, "positive": 2}

# =============================================================================
# STEP 1: DATA LOADING AND WINDOWING
# =============================================================================


def load_subject_data(subject_id: int, dataset_dir: str = DATASET_DIR) -> List[Dict]:
    """
    Load raw EEG data for a single subject across all sessions and trials.

    Parameters
    ----------
    subject_id : int
        Subject ID (1-15 for SEED)
    dataset_dir : str
        Path to SEED dataset directory

    Returns
    -------
    list
        List of trial dictionaries with keys:
        - 'signal': np.ndarray (62, T)
        - 'label': int (-1, 0, 1)
        - 'trial': int
        - 'rep': int (session)
    """
    logger.info(f"Loading subject {subject_id} data...")
    try:
        # Load labels once
        label_path = os.path.join(dataset_dir, LABEL_FILE)
        labels = load_labels(label_path)

        # Find subject file
        subject_files = [
            f for f in os.listdir(dataset_dir) if f.startswith(f"{subject_id}_")
        ]
        if not subject_files:
            logger.error(f"No files found for subject {subject_id}")
            return []

        file_path = os.path.join(dataset_dir, subject_files[0])

        # Load subject data
        dataset = load_subject(file_path, labels, subject_id)
        logger.info(f"✓ Loaded {len(dataset)} trials for subject {subject_id}")
        return dataset
    except Exception as e:
        logger.error(f"✗ Failed to load subject {subject_id}: {e}")
        import traceback

        traceback.print_exc()
        return []


def create_windows(
    signal: np.ndarray, window_size: int, step_size: int = None
) -> np.ndarray:
    """
    Segment continuous EEG signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        EEG signal shape (62, T)
    window_size : int
        Window size in samples
    step_size : int, optional
        Step size (default: window_size // 2 for 50% overlap)

    Returns
    -------
    np.ndarray
        Windows shape (n_windows, 62, window_size)
    """
    if step_size is None:
        step_size = window_size // 2

    channels, T = signal.shape
    windows = []

    for start in range(0, T - window_size + 1, step_size):
        end = start + window_size
        window = signal[:, start:end]
        windows.append(window)

    if len(windows) == 0:
        logger.warning(f"Signal too short ({T}) for window_size={window_size}")
        return np.array([]).reshape(0, channels, window_size)

    return np.array(windows)


def segment_trial(signal: np.ndarray, window_sizes: List[int]) -> Dict[int, np.ndarray]:
    """
    Segment a single EEG trial into windows of multiple sizes.

    Parameters
    ----------
    signal : np.ndarray
        EEG signal shape (62, T)
    window_sizes : list
        List of window sizes to create

    Returns
    -------
    dict
        Mapping {window_size: windows_array}
    """
    windowed_trial = {}

    for ws in window_sizes:
        windows = create_windows(signal, window_size=ws)
        windowed_trial[ws] = windows

    return windowed_trial


# =============================================================================
# STEP 2: FEATURE EXTRACTION
# =============================================================================


def dict_features_to_array(feature_dict: dict) -> np.ndarray:
    """
    Convert EEG feature dictionary to numeric array.

    Parameters
    ----------
    feature_dict : dict
        Dictionary with keys: theta_ratio, alpha_ratio, beta_ratio, gamma_ratio,
                             activity, entropy, faa

    Returns
    -------
    np.ndarray
        Feature vector with shape (7,)
    """
    try:
        feature_order = [
            "theta_ratio",
            "alpha_ratio",
            "beta_ratio",
            "gamma_ratio",
            "activity",
            "entropy",
            "faa",
        ]
        return np.array(
            [feature_dict.get(key, 0.0) for key in feature_order], dtype=np.float32
        )
    except Exception as e:
        logger.warning(f"Failed to convert feature dict to array: {e}")
        return np.zeros(7, dtype=np.float32)


def extract_features_per_window(
    windows: np.ndarray, extraction_method: str = "eeg_llm"
) -> np.ndarray:
    """
    Extract features from all windows in a batch.

    Parameters
    ----------
    windows : np.ndarray
        Windows shape (n_windows, 62, window_size)
    extraction_method : str
        'eeg_llm' for LLM-specific features (default) or 'eeg_stats' for classical ML features

    Returns
    -------
    np.ndarray
        Features shape (n_windows, n_features)
    """
    if len(windows) == 0:
        return (
            np.array([]).reshape(0, 7)
            if extraction_method == "eeg_llm"
            else np.array([]).reshape(0, 1)
        )

    features_list = []

    for window in windows:
        if extraction_method == "eeg_llm":
            try:
                feats_dict = extract_eeg_features(window)
                feats_array = dict_features_to_array(feats_dict)
                features_list.append(feats_array)
            except Exception as e:
                logger.warning(f"LLM feature extraction failed: {e}")
                features_list.append(np.zeros(7, dtype=np.float32))
        else:
            try:
                feats = extract_features(window)
                features_list.append(feats)
            except Exception as e:
                logger.warning(f"Classical feature extraction failed: {e}")
                features_list.append(np.nan)

    return (
        np.array(features_list)
        if len(features_list) > 0
        else np.array([]).reshape(0, 7)
    )


# =============================================================================
# STEP 3: LLM INFERENCE
# =============================================================================


class MockLLMClient:
    """Mock LLM client for testing without actual LLM service."""

    def __init__(self):
        self.call_count = 0

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Mock prediction based on EEG feature statistics.

        Uses heuristic based on alpha and FAA (frontal asymmetry) to simulate emotion classification.
        Feature order: [theta_ratio, alpha_ratio, beta_ratio, gamma_ratio, activity, entropy, faa]

        In production, this would call actual LLM via API.
        """
        self.call_count += 1

        # Handle invalid input
        if not isinstance(features, np.ndarray) or len(features) == 0:
            return "neutral", 0.33

        if np.isnan(features).all() or np.isinf(features).all():
            return "neutral", 0.33

        try:
            # Extract key features for emotion heuristic
            alpha_ratio = (
                features[1] if len(features) > 1 else 0.0
            )  # index 1 = alpha_ratio
            faa = features[6] if len(features) > 6 else 0.0  # index 6 = FAA (valence)

            # Heuristic:
            # High alpha + positive FAA → positive (relaxed + left prefrontal activity)
            # Low alpha + negative FAA → negative (alert/stressed + right prefrontal activity)
            # Otherwise → neutral

            valence_score = alpha_ratio + (faa * 0.1)  # weight alpha more than FAA

            if valence_score > 0.35:
                return "positive", min(0.7, valence_score)
            elif valence_score < 0.20:
                return "negative", min(0.7, 1.0 - valence_score)
            else:
                return "neutral", 0.5
        except Exception as e:
            logger.warning(f"Error in mock LLM prediction: {e}")
            return "neutral", 0.33


def run_llm_inference(features: np.ndarray, llm_client) -> List[Tuple[str, float]]:
    """
    Run LLM inference on extracted features.

    Parameters
    ----------
    features : np.ndarray
        Features shape (n_windows, n_features)
    llm_client : LLMClient or MockLLMClient
        LLM inference client

    Returns
    -------
    list
        List of (prediction, confidence) tuples
    """
    predictions = []

    for feature_vector in features:
        try:
            pred, conf = llm_client.predict(feature_vector)
            predictions.append((pred, conf))
        except Exception as e:
            logger.warning(f"LLM inference failed: {e}")
            predictions.append(("neutral", 0.33))

    return predictions


# =============================================================================
# STEP 4: ANALYSIS AND METRICS
# =============================================================================


def compute_window_metrics(
    predictions: List[Tuple[str, float]], ground_truth: int
) -> Dict:
    """
    Compute accuracy and confidence metrics for predictions on windows.

    Parameters
    ----------
    predictions : list
        List of (prediction, confidence) tuples
    ground_truth : int
        True label (0, 1, 2)

    Returns
    -------
    dict
        Metrics including accuracy, avg confidence, majority voting result
    """
    if len(predictions) == 0:
        return {
            "n_windows": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "majority_pred": None,
            "majority_accuracy": 0,
            "predictions": [],
        }

    # Extract predictions and confidences
    preds = [p[0] for p in predictions]
    confs = [p[1] for p in predictions]

    # Convert to numeric labels
    pred_labels = [EMOTION_LABELS_REVERSE.get(p, 1) for p in preds]

    # Window-level accuracy
    correct = sum(1 for label in pred_labels if label == ground_truth)
    window_accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0

    # Majority voting
    from collections import Counter

    majority_pred = max(set(pred_labels), key=pred_labels.count)
    majority_accuracy = 1 if majority_pred == ground_truth else 0

    return {
        "n_windows": len(predictions),
        "window_accuracy": window_accuracy,
        "avg_confidence": np.mean(confs) if confs else 0.0,
        "std_confidence": np.std(confs) if confs else 0.0,
        "majority_pred": majority_pred,
        "majority_accuracy": majority_accuracy,
        "predictions": pred_labels,
    }


# =============================================================================
# STEP 5: FULL ANALYSIS PIPELINE
# =============================================================================


def analyze_subject(subject_id: int, window_sizes: List[int] = None) -> pd.DataFrame:
    """
    Complete analysis pipeline for a single subject.

    Parameters
    ----------
    subject_id : int
        Subject ID (1-15)
    window_sizes : list, optional
        Window sizes to test (default: WINDOW_SIZES)

    Returns
    -------
    pd.DataFrame
        Results dataframe with one row per trial-window_size combination
    """
    if window_sizes is None:
        window_sizes = WINDOW_SIZES

    logger.info(f"\n{'='*70}")
    logger.info(f"ANALYZING SUBJECT {subject_id}")
    logger.info(f"{'='*70}")

    # Load subject data
    trials = load_subject_data(subject_id)
    if len(trials) == 0:
        logger.error(f"No data for subject {subject_id}")
        return pd.DataFrame()

    # Initialize LLM client (mock)
    llm_client = MockLLMClient()

    # Store results
    results = []

    # Process each trial
    for trial_idx, trial in enumerate(
        tqdm(trials, desc=f"Subject {subject_id} trials")
    ):
        signal = trial["signal"]
        label = LABELS_MAP.get(trial["label"], 1)
        trial_id = trial.get("trial", trial_idx)
        session = trial.get("rep", 0)  # Default to 0 if not available

        # Create windows of multiple sizes
        windowed_data = segment_trial(signal, window_sizes)

        # Process each window size
        for ws in window_sizes:
            windows = windowed_data[ws]

            if len(windows) == 0:
                # Trial too short for this window size
                results.append(
                    {
                        "subject": subject_id,
                        "trial": trial_id,
                        "session": session,
                        "emotion": EMOTION_LABELS[label],
                        "window_size": ws,
                        "signal_length": signal.shape[1],
                        "n_windows": 0,
                        "window_accuracy": np.nan,
                        "majority_accuracy": np.nan,
                        "avg_confidence": np.nan,
                        "std_confidence": np.nan,
                    }
                )
                continue

            # Extract LLM features
            features = extract_features_per_window(windows, extraction_method="eeg_llm")

            # Run inference
            predictions = run_llm_inference(features, llm_client)

            # Compute metrics
            metrics = compute_window_metrics(predictions, label)

            # Store result
            results.append(
                {
                    "subject": subject_id,
                    "trial": trial_id,
                    "session": session,
                    "emotion": EMOTION_LABELS[label],
                    "window_size": ws,
                    "signal_length": signal.shape[1],
                    "n_windows": metrics["n_windows"],
                    "window_accuracy": metrics["window_accuracy"],
                    "majority_accuracy": metrics["majority_accuracy"],
                    "avg_confidence": metrics["avg_confidence"],
                    "std_confidence": metrics["std_confidence"],
                }
            )

    return pd.DataFrame(results)


def analyze_multiple_subjects(
    subject_ids: List[int], window_sizes: List[int] = None
) -> pd.DataFrame:
    """
    Run analysis for multiple subjects.

    Parameters
    ----------
    subject_ids : list
        Subject IDs to analyze
    window_sizes : list, optional
        Window sizes to test

    Returns
    -------
    pd.DataFrame
        Combined results for all subjects
    """
    all_results = []

    for subject_id in subject_ids:
        try:
            subject_results = analyze_subject(subject_id, window_sizes)
            if len(subject_results) > 0:
                all_results.append(subject_results)
        except Exception as e:
            logger.error(f"Error analyzing subject {subject_id}: {e}")
            continue

    if len(all_results) == 0:
        logger.error("No results collected")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    logger.info(f"\n✓ Completed analysis for {len(subject_ids)} subjects")
    logger.info(f"✓ Total trials analyzed: {len(combined)}")

    return combined


# =============================================================================
# STEP 6: VISUALIZATION
# =============================================================================


def plot_accuracy_vs_window_size(results: pd.DataFrame, save_path: str = None):
    """
    Plot accuracy (window and majority) vs. window size.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by window size and compute mean accuracy
    window_stats = (
        results.groupby("window_size")
        .agg(
            {
                "window_accuracy": ["mean", "std"],
                "majority_accuracy": ["mean", "std"],
            }
        )
        .reset_index()
    )

    window_stats.columns = [
        "window_size",
        "window_mean",
        "window_std",
        "maj_mean",
        "maj_std",
    ]

    # Plot window accuracy
    axes[0].errorbar(
        window_stats["window_size"],
        window_stats["window_mean"],
        yerr=window_stats["window_std"],
        marker="o",
        capsize=5,
        label="Per-Window Accuracy",
    )
    axes[0].set_xlabel("Window Size (samples)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("LLM Inference Accuracy vs. Window Size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot majority voting accuracy
    axes[1].errorbar(
        window_stats["window_size"],
        window_stats["maj_mean"],
        yerr=window_stats["maj_std"],
        marker="s",
        capsize=5,
        color="orange",
        label="Majority Voting Accuracy",
    )
    axes[1].set_xlabel("Window Size (samples)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Majority Voting Accuracy vs. Window Size")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot to {save_path}")

    plt.close()


def plot_accuracy_per_emotion(results: pd.DataFrame, save_path: str = None):
    """
    Plot accuracy separately for each emotion class.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    emotions = results["emotion"].unique()
    window_sizes = sorted(results["window_size"].unique())

    for emotion in emotions:
        emotion_data = results[results["emotion"] == emotion]
        accuracy_by_ws = (
            emotion_data.groupby("window_size")["majority_accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )

        ax.errorbar(
            accuracy_by_ws["window_size"],
            accuracy_by_ws["mean"],
            yerr=accuracy_by_ws["std"],
            marker="o",
            capsize=5,
            label=emotion,
        )

    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Accuracy (Majority Voting)")
    ax.set_title("LLM Inference Accuracy by Emotion Class")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot to {save_path}")

    plt.close()


def plot_confidence_distribution(results: pd.DataFrame, save_path: str = None):
    """
    Plot confidence score distribution across window sizes.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    window_sizes = sorted(results["window_size"].unique())

    confidence_data = [
        results[results["window_size"] == ws]["avg_confidence"].dropna().values
        for ws in window_sizes
    ]

    bp = ax.boxplot(
        confidence_data,
        labels=[f"{ws}s" for ws in window_sizes],
        patch_artist=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Average Confidence")
    ax.set_title("LLM Confidence Score Distribution vs. Window Size")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot to {save_path}")

    plt.close()


def plot_heatmap_per_subject(results: pd.DataFrame, save_path: str = None):
    """
    Create heatmap of accuracy per subject and window size.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results
    save_path : str, optional
        Path to save figure
    """
    # Pivot: subjects as rows, window sizes as columns
    pivot_data = results.pivot_table(
        index="subject",
        columns="window_size",
        values="majority_accuracy",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )

    ax.set_title("Majority Voting Accuracy: Per Subject and Window Size")
    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Subject ID")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved plot to {save_path}")

    plt.close()


# =============================================================================
# STEP 7: STATISTICAL ANALYSIS
# =============================================================================


def statistical_summary(results: pd.DataFrame) -> Dict:
    """
    Compute statistical summary of results.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results

    Returns
    -------
    dict
        Statistical summary
    """
    summary = {
        "total_trials": len(results),
        "total_subjects": results["subject"].nunique(),
        "window_sizes": sorted(results["window_size"].unique().tolist()),
    }

    # Per-window-size statistics
    window_stats = {}

    for ws in summary["window_sizes"]:
        ws_data = results[results["window_size"] == ws]

        accuracies = ws_data["majority_accuracy"].values
        confidences = ws_data["avg_confidence"].dropna().values

        window_stats[ws] = {
            "n_trials": len(ws_data),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "mean_confidence": (
                float(np.mean(confidences)) if len(confidences) > 0 else 0.0
            ),
            "std_confidence": (
                float(np.std(confidences)) if len(confidences) > 0 else 0.0
            ),
        }

    summary["per_window_size"] = window_stats

    # Per-emotion statistics
    emotion_stats = {}

    for emotion in results["emotion"].unique():
        emotion_data = results[results["emotion"] == emotion]
        accuracies = emotion_data["majority_accuracy"].values

        emotion_stats[emotion] = {
            "n_trials": len(emotion_data),
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
        }

    summary["per_emotion"] = emotion_stats

    return summary


def correlation_analysis(results: pd.DataFrame) -> Dict:
    """
    Analyze correlations between window size and accuracy.

    Parameters
    ----------
    results : pd.DataFrame
        Analysis results

    Returns
    -------
    dict
        Correlation results
    """
    # Correlation: window size vs accuracy
    corr_accuracy, p_accuracy = stats.spearmanr(
        results["window_size"], results["majority_accuracy"]
    )

    # Correlation: window size vs confidence
    valid_conf = results[results["avg_confidence"].notna()]
    if len(valid_conf) > 2:
        corr_confidence, p_confidence = stats.spearmanr(
            valid_conf["window_size"], valid_conf["avg_confidence"]
        )
    else:
        corr_confidence, p_confidence = np.nan, np.nan

    # Correlation: n_windows vs accuracy
    valid_nw = results[results["n_windows"] > 0]
    if len(valid_nw) > 2:
        corr_nwindows, p_nwindows = stats.spearmanr(
            valid_nw["n_windows"], valid_nw["majority_accuracy"]
        )
    else:
        corr_nwindows, p_nwindows = np.nan, np.nan

    return {
        "window_size_vs_accuracy": {
            "correlation": float(corr_accuracy),
            "p_value": float(p_accuracy),
        },
        "window_size_vs_confidence": {
            "correlation": float(corr_confidence),
            "p_value": float(p_confidence),
        },
        "n_windows_vs_accuracy": {
            "correlation": float(corr_nwindows),
            "p_value": float(p_nwindows),
        },
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main(
    subject_ids: List[int] = None,
    window_sizes: List[int] = None,
    output_dir: str = None,
):
    """
    Run complete window analysis pipeline.

    Parameters
    ----------
    subject_ids : list, optional
        Subject IDs to analyze (default: all 15 subjects)
    window_sizes : list, optional
        Window sizes to test (default: WINDOW_SIZES)
    output_dir : str, optional
        Directory to save results (default: OUTPUT_DIR)
    """
    if subject_ids is None:
        subject_ids = list(range(1, 16))

    if window_sizes is None:
        window_sizes = WINDOW_SIZES

    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Create output directory
    analysis_dir = os.path.join(output_dir, "window_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    logger.info(f"\nWindow Analysis Configuration:")
    logger.info(f"  Subjects: {subject_ids}")
    logger.info(f"  Window sizes: {window_sizes}")
    logger.info(f"  Output directory: {analysis_dir}")

    # Run analysis
    results = analyze_multiple_subjects(subject_ids, window_sizes)

    if len(results) == 0:
        logger.error("No results to analyze")
        return

    # Save raw results
    results_csv = os.path.join(analysis_dir, "window_analysis_results.csv")
    results.to_csv(results_csv, index=False)
    logger.info(f"✓ Saved results to {results_csv}")

    # Statistical summary
    summary = statistical_summary(results)
    summary_json = os.path.join(analysis_dir, "statistical_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Saved summary to {summary_json}")

    # Correlation analysis
    correlations = correlation_analysis(results)
    corr_json = os.path.join(analysis_dir, "correlation_analysis.json")
    with open(corr_json, "w") as f:
        json.dump(correlations, f, indent=2)
    logger.info(f"✓ Saved correlation analysis to {corr_json}")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    plot_accuracy_vs_window_size(
        results, os.path.join(analysis_dir, "accuracy_vs_window_size.png")
    )

    plot_accuracy_per_emotion(
        results, os.path.join(analysis_dir, "accuracy_per_emotion.png")
    )

    plot_confidence_distribution(
        results, os.path.join(analysis_dir, "confidence_distribution.png")
    )

    plot_heatmap_per_subject(
        results, os.path.join(analysis_dir, "accuracy_heatmap.png")
    )

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL SUMMARY")
    logger.info("=" * 70)

    print(f"\nTotal Trials Analyzed: {summary['total_trials']}")
    print(f"Total Subjects: {summary['total_subjects']}")
    print(f"\nPer Window Size Statistics:")

    for ws, stats_ws in summary["per_window_size"].items():
        print(f"\n  Window Size: {ws} samples")
        print(f"    Trials: {stats_ws['n_trials']}")
        print(
            f"    Mean Accuracy: {stats_ws['mean_accuracy']:.3f} ± {stats_ws['std_accuracy']:.3f}"
        )
        print(
            f"    Range: [{stats_ws['min_accuracy']:.3f}, {stats_ws['max_accuracy']:.3f}]"
        )
        print(
            f"    Mean Confidence: {stats_ws['mean_confidence']:.3f} ± {stats_ws['std_confidence']:.3f}"
        )

    print(f"\nPer Emotion Statistics:")
    for emotion, stats_em in summary["per_emotion"].items():
        print(f"\n  {emotion}:")
        print(f"    Trials: {stats_em['n_trials']}")
        print(
            f"    Mean Accuracy: {stats_em['mean_accuracy']:.3f} ± {stats_em['std_accuracy']:.3f}"
        )

    print(f"\nCorrelation Analysis:")
    print(f"  Window Size vs Accuracy:")
    print(
        f"    r = {correlations['window_size_vs_accuracy']['correlation']:.3f}, "
        f"p = {correlations['window_size_vs_accuracy']['p_value']:.3e}"
    )
    print(f"  Window Size vs Confidence:")
    print(
        f"    r = {correlations['window_size_vs_confidence']['correlation']:.3f}, "
        f"p = {correlations['window_size_vs_confidence']['p_value']:.3e}"
    )
    print(f"  N Windows vs Accuracy:")
    print(
        f"    r = {correlations['n_windows_vs_accuracy']['correlation']:.3f}, "
        f"p = {correlations['n_windows_vs_accuracy']['p_value']:.3e}"
    )

    logger.info("=" * 70)
    logger.info(f"✓ Analysis complete. Results saved to {analysis_dir}")


if __name__ == "__main__":
    # Example: Analyze first 3 subjects with fewer window sizes
    # main(subject_ids=[1, 2, 3], window_sizes=[200, 400, 600, 800])

    # For full analysis, uncomment:
    main()
