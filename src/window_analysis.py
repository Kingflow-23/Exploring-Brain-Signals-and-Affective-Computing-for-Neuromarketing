import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seed_loader import load_subject, load_labels
from llm_inference import extract_eeg_features
from config import DATASET_DIR, OUTPUT_DIR, LABEL_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WINDOW_ANALYSIS")


# =========================================================
# CONFIG
# =========================================================

WINDOW_SIZES = [200, 400, 600, 800, 1000, 1200]
STEP_RATIO = 0.5  # 50% overlap by default

# Your actual feature schema, grouped semantically.
FEATURE_SCHEMA = {
    "global_spectral_state": [
        "theta_ratio",
        "alpha_ratio",
        "beta_ratio",
        "gamma_ratio",
    ],
    "frontal_asymmetry": [
        "faa_fp1_fp2",
        "faa_f3_f4",
        "faa_f7_f8",
    ],
    "regional_activity": [
        "frontal_alpha",
        "frontal_beta",
        "frontal_gamma",
        "temporal_alpha",
        "temporal_beta",
        "occipital_alpha",
    ],
    "complexity": [
        "activity",
        "entropy",
    ],
    "derived_ratios": [
        "beta_alpha_ratio",
        "gamma_beta_ratio",
        "frontal_occipital_alpha_ratio",
    ],
}

FEATURE_ORDER = [f for group in FEATURE_SCHEMA.values() for f in group]


# =========================================================
# UTILITIES
# =========================================================


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def robust_zscore(x: np.ndarray, axis=0, eps=1e-8) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    """
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1.4826 * mad + eps)


def softmax(x: np.ndarray, axis=-1, eps=1e-12) -> np.ndarray:
    x = x - np.nanmax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.nansum(exp_x, axis=axis, keepdims=True) + eps)


def shannon_entropy_from_distribution(p: np.ndarray, axis=-1, eps=1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def pairwise_upper(values: List[np.ndarray], metric="cosine") -> List[float]:
    out = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if metric == "cosine":
                out.append(cosine_similarity(values[i], values[j]))
            elif metric == "spearman":
                # rank correlation via pandas
                a = pd.Series(values[i]).rank().to_numpy()
                b = pd.Series(values[j]).rank().to_numpy()
                out.append(cosine_similarity(a, b))
            else:
                raise ValueError(f"Unknown metric: {metric}")
    return out


def summarize_vector(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "var": float(np.nanvar(x)),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "median": float(np.nanmedian(x)),
        "iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
        "mad": float(np.nanmedian(np.abs(x - np.nanmedian(x)))),
        "cv": float(np.nanstd(x) / (np.abs(np.nanmean(x)) + 1e-8)),
        "skew": float(pd.Series(x).skew()),
        "kurtosis": float(pd.Series(x).kurtosis()),
    }


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

    # If you have several sessions/files per subject, extend this to load all of them.
    file_path = os.path.join(DATASET_DIR, subject_files[0])
    return load_subject(file_path, labels, subject_id)


# =========================================================
# WINDOWING
# =========================================================


def create_windows(signal: np.ndarray, window_size: int, step: int = None):
    """
    signal: shape (n_channels, n_samples)
    returns: shape (n_windows, n_channels, window_size)
    """
    if step is None:
        step = max(1, window_size // 2)

    windows = []
    T = signal.shape[1]

    for start in range(0, T - window_size + 1, step):
        windows.append(signal[:, start : start + window_size])

    if not windows:
        return np.empty((0, signal.shape[0], window_size))

    return np.asarray(windows)


# =========================================================
# FEATURE EXTRACTION
# =========================================================


def extract_features_per_window(windows: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features in a fixed order.
    Missing features are filled with NaN.
    """
    features = []

    for w in windows:
        try:
            f = extract_eeg_features(w)
            row = [safe_float(f.get(name, np.nan)) for name in FEATURE_ORDER]
            features.append(row)
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features.append([np.nan] * len(FEATURE_ORDER))

    return np.asarray(features, dtype=float), FEATURE_ORDER


# =========================================================
# FEATURE QUALITY METRICS
# =========================================================


def feature_matrix_quality(feature_matrix: np.ndarray) -> Dict[str, float]:
    """
    Metrics that describe the quality of the feature matrix itself.
    """
    if feature_matrix.size == 0:
        return {
            "missing_rate": 1.0,
            "row_valid_rate": 0.0,
            "feature_valid_rate": 0.0,
            "sparsity_rate": 0.0,
            "matrix_entropy": 0.0,
            "mean_abs_corr": 0.0,
            "effective_rank_proxy": 0.0,
        }

    X = np.asarray(feature_matrix, dtype=float)

    missing_rate = float(np.isnan(X).mean())
    row_valid_rate = float(np.mean(np.sum(~np.isnan(X), axis=1) == X.shape[1]))
    feature_valid_rate = float(np.mean(np.sum(~np.isnan(X), axis=0) > 0))
    sparsity_rate = float(np.mean(np.isclose(X, 0.0, equal_nan=False)))

    # Robustly standardize before matrix-level entropy and similarity.
    Xf = X.copy()
    col_med = np.nanmedian(Xf, axis=0)
    inds = np.where(np.isnan(Xf))
    Xf[inds] = np.take(col_med, inds[1])
    Z = robust_zscore(Xf)

    # Convert each window into a positive distribution over features.
    P = softmax(np.abs(Z), axis=1)
    ent = shannon_entropy_from_distribution(P, axis=1)
    matrix_entropy = float(np.nanmean(ent))

    # Correlation structure
    corr = pd.DataFrame(Z).corr().to_numpy()
    if np.isnan(corr).all():
        mean_abs_corr = 0.0
    else:
        mean_abs_corr = float(np.nanmean(np.abs(corr[np.triu_indices_from(corr, k=1)])))

    # A rough effective-rank proxy from singular values.
    try:
        s = np.linalg.svd(np.nan_to_num(Z), compute_uv=False)
        p = s / (np.sum(s) + 1e-12)
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    except Exception:
        eff_rank = 0.0

    return {
        "missing_rate": missing_rate,
        "row_valid_rate": row_valid_rate,
        "feature_valid_rate": feature_valid_rate,
        "sparsity_rate": sparsity_rate,
        "matrix_entropy": matrix_entropy,
        "mean_abs_corr": mean_abs_corr,
        "effective_rank_proxy": eff_rank,
    }


def per_feature_statistics(
    feature_matrix: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    """
    Per-feature statistics across windows.
    """
    if feature_matrix.size == 0:
        return pd.DataFrame()

    X = np.asarray(feature_matrix, dtype=float)
    rows = []

    for i, name in enumerate(feature_names):
        x = X[:, i]
        valid = x[~np.isnan(x)]
        if len(valid) == 0:
            rows.append(
                {
                    "feature": name,
                    "n": 0,
                    "missing_rate": 1.0,
                    "mean": np.nan,
                    "std": np.nan,
                    "var": np.nan,
                    "median": np.nan,
                    "iqr": np.nan,
                    "mad": np.nan,
                    "cv": np.nan,
                    "skew": np.nan,
                    "kurtosis": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "p05": np.nan,
                    "p95": np.nan,
                    "stability_inverse_var": np.nan,
                }
            )
            continue

        s = pd.Series(valid)
        var = float(np.nanvar(valid))
        rows.append(
            {
                "feature": name,
                "n": int(len(valid)),
                "missing_rate": float(np.isnan(x).mean()),
                "mean": float(np.nanmean(valid)),
                "std": float(np.nanstd(valid)),
                "var": var,
                "median": float(np.nanmedian(valid)),
                "iqr": float(np.nanpercentile(valid, 75) - np.nanpercentile(valid, 25)),
                "mad": float(np.nanmedian(np.abs(valid - np.nanmedian(valid)))),
                "cv": float(np.nanstd(valid) / (np.abs(np.nanmean(valid)) + 1e-8)),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
                "min": float(np.nanmin(valid)),
                "max": float(np.nanmax(valid)),
                "p05": float(np.nanpercentile(valid, 5)),
                "p95": float(np.nanpercentile(valid, 95)),
                "stability_inverse_var": float(1.0 / (var + 1e-8)),
            }
        )

    return pd.DataFrame(rows)


def window_similarity_stats(feature_matrix: np.ndarray) -> Dict[str, float]:
    """
    Similarity across consecutive windows after robust normalization.
    """
    if len(feature_matrix) < 2:
        return {"similarity_mean": 0.0, "similarity_std": 0.0}

    X = np.asarray(feature_matrix, dtype=float)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])
    Z = robust_zscore(X)

    sims = []
    for i in range(len(Z) - 1):
        sims.append(cosine_similarity(Z[i], Z[i + 1]))

    return {
        "similarity_mean": float(np.mean(sims)),
        "similarity_std": float(np.std(sims)),
    }


def feature_window_sensitivity(
    feature_matrix: np.ndarray, window_size: int, feature_names: List[str]
) -> pd.DataFrame:
    """
    Sensitivity summary for the current window size.
    This is later aggregated across all window sizes.
    """
    if feature_matrix.size == 0:
        return pd.DataFrame()

    feat_stats = per_feature_statistics(feature_matrix, feature_names)
    quality = feature_matrix_quality(feature_matrix)
    sim = window_similarity_stats(feature_matrix)

    feat_stats["window_size"] = window_size
    feat_stats["matrix_entropy"] = quality["matrix_entropy"]
    feat_stats["mean_abs_corr"] = quality["mean_abs_corr"]
    feat_stats["effective_rank_proxy"] = quality["effective_rank_proxy"]
    feat_stats["missing_rate_matrix"] = quality["missing_rate"]
    feat_stats["similarity_mean"] = sim["similarity_mean"]
    feat_stats["similarity_std"] = sim["similarity_std"]

    return feat_stats


# =========================================================
# MAIN ANALYSIS
# =========================================================


def analyze_subject(subject_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - trial/window-size level summary
      - per-feature summary records
    """
    logger.info(f"Analyzing subject {subject_id}")
    trials = load_subject_data(subject_id)

    summary_rows = []
    feature_rows = []

    for trial_idx, trial in enumerate(trials):
        signal = trial["signal"]

        # Keep the trial ID if available
        trial_label = trial.get("label", None)

        for ws in WINDOW_SIZES:
            logger.info(f"Subject {subject_id} - Trial {trial_idx} - Window Size {ws}")

            windows = create_windows(signal, ws)

            if len(windows) == 0:
                continue

            features, feature_names = extract_features_per_window(windows)

            # Robust trial-level summary vector:
            # median over windows is usually better than mean for noisy EEG features.
            col_med = np.nanmedian(features, axis=0)
            col_mean = np.nanmean(features, axis=0)
            col_std = np.nanstd(features, axis=0)

            quality = feature_matrix_quality(features)
            sim = window_similarity_stats(features)

            # Trial-level aggregated metrics
            summary_rows.append(
                {
                    "subject": subject_id,
                    "trial": trial_idx,
                    "label": trial_label,
                    "window_size": ws,
                    "n_windows": len(windows),
                    "n_features": len(feature_names),
                    # quality of the matrix passed to LLM
                    "missing_rate": quality["missing_rate"],
                    "row_valid_rate": quality["row_valid_rate"],
                    "feature_valid_rate": quality["feature_valid_rate"],
                    "sparsity_rate": quality["sparsity_rate"],
                    "matrix_entropy": quality["matrix_entropy"],
                    "mean_abs_corr": quality["mean_abs_corr"],
                    "effective_rank_proxy": quality["effective_rank_proxy"],
                    # temporal coherence
                    "similarity_mean": sim["similarity_mean"],
                    "similarity_std": sim["similarity_std"],
                    # compact stability measures
                    "feature_mean_of_means": float(np.nanmean(col_mean)),
                    "feature_mean_of_stds": float(np.nanmean(col_std)),
                    "feature_mean_of_vars": float(
                        np.nanmean(np.nanvar(features, axis=0))
                    ),
                    # overall stability score: high when variance is low
                    "stability_score": float(
                        1.0 / (np.nanmean(np.nanvar(features, axis=0)) + 1e-8)
                    ),
                }
            )

            # Per-feature stats for this subject/trial/window size
            feat_df = feature_window_sensitivity(features, ws, feature_names)
            feat_df["subject"] = subject_id
            feat_df["trial"] = trial_idx
            feat_df["label"] = trial_label
            feature_rows.append(feat_df)

    summary_df = pd.DataFrame(summary_rows)
    feature_df = (
        pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    )

    return summary_df, feature_df


# =========================================================
# CROSS-WINDOW COMPARISON
# =========================================================


def compare_window_sizes(
    summary_df: pd.DataFrame, feature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare the same subject/trial across different window sizes.
    Measures whether the feature representation is semantically stable.
    """
    # This uses aggregated per-trial/per-window-size info already in summary_df.
    # For a stronger comparison, you can also store the raw trial-level feature vectors
    # and compare them directly. Here we compare stability trends at the summary level.
    out_rows = []

    for (subject, trial), g in summary_df.groupby(["subject", "trial"]):
        g = g.sort_values("window_size")
        if len(g) < 2:
            continue

        for i in range(len(g) - 1):
            a = g.iloc[i]
            b = g.iloc[i + 1]
            out_rows.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "window_a": int(a["window_size"]),
                    "window_b": int(b["window_size"]),
                    "delta_stability": float(
                        b["stability_score"] - a["stability_score"]
                    ),
                    "delta_similarity": float(
                        b["similarity_mean"] - a["similarity_mean"]
                    ),
                    "delta_entropy": float(b["matrix_entropy"] - a["matrix_entropy"]),
                    "delta_missing_rate": float(b["missing_rate"] - a["missing_rate"]),
                    "delta_mean_abs_corr": float(
                        b["mean_abs_corr"] - a["mean_abs_corr"]
                    ),
                    "delta_effective_rank": float(
                        b["effective_rank_proxy"] - a["effective_rank_proxy"]
                    ),
                }
            )

    return pd.DataFrame(out_rows)


def aggregate_feature_trends(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-feature trends across all subjects/trials/window sizes.
    Measures whether a feature changes systematically with window size.
    """
    if feature_df.empty:
        return pd.DataFrame()

    rows = []
    for feature, g in feature_df.groupby("feature"):
        g = g.sort_values("window_size")
        # Use feature median across windows as a robust summary of the feature value
        # within each trial/window-size block.
        # Since the current table is already aggregated, we use mean of stats.
        trend_x = np.log(g["window_size"].to_numpy())
        trend_y = g["median"].to_numpy()

        if len(np.unique(trend_x)) < 2 or np.all(np.isnan(trend_y)):
            slope = np.nan
            corr = np.nan
        else:
            valid = ~np.isnan(trend_y)
            if valid.sum() < 2:
                slope = np.nan
                corr = np.nan
            else:
                slope = float(np.polyfit(trend_x[valid], trend_y[valid], 1)[0])
                corr = float(
                    pd.Series(trend_x[valid]).corr(
                        pd.Series(trend_y[valid]), method="spearman"
                    )
                )

        rows.append(
            {
                "feature": feature,
                "trend_slope_log_window": slope,
                "spearman_window_correlation": corr,
                "mean_cv": float(np.nanmean(g["cv"])),
                "mean_iqr": float(np.nanmean(g["iqr"])),
                "mean_missing_rate": float(np.nanmean(g["missing_rate"])),
                "mean_stability_inverse_var": float(
                    np.nanmean(g["stability_inverse_var"])
                ),
                "mean_skew": float(np.nanmean(g["skew"])),
                "mean_kurtosis": float(np.nanmean(g["kurtosis"])),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# REPORTING
# =========================================================


def plot_summary(summary_df: pd.DataFrame, out_dir: str):
    if summary_df.empty:
        return

    g = summary_df.groupby("window_size").mean(numeric_only=True)

    plt.figure()
    plt.plot(g.index, g["stability_score"], marker="o")
    plt.title("Feature Stability vs Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Stability Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stability_vs_window.png"))
    plt.close()

    plt.figure()
    plt.plot(g.index, g["similarity_mean"], marker="o")
    plt.title("Consecutive Window Similarity vs Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "similarity_vs_window.png"))
    plt.close()

    plt.figure()
    plt.plot(g.index, g["matrix_entropy"], marker="o")
    plt.title("Feature Matrix Entropy vs Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_vs_window.png"))
    plt.close()

    plt.figure()
    plt.plot(g.index, g["mean_abs_corr"], marker="o")
    plt.title("Mean Absolute Feature Correlation vs Window Size")
    plt.xlabel("Window Size")
    plt.ylabel("Mean |corr|")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_vs_window.png"))
    plt.close()


def build_prompt_audit(feature_trends_df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic audit table to help refine the LLM prompt.
    """
    if feature_trends_df.empty:
        return pd.DataFrame()

    df = feature_trends_df.copy()

    # Lower CV = more stable. Higher |trend_slope| = more window-sensitive.
    df["stability_rank"] = df["mean_cv"].rank(pct=True, ascending=True)
    df["sensitivity_rank"] = (
        df["trend_slope_log_window"].abs().rank(pct=True, ascending=True)
    )

    # Simple interpretability score:
    # higher when stable and less window-sensitive
    df["prompt_relevance_score"] = (
        1.0 - 0.5 * df["stability_rank"] - 0.5 * df["sensitivity_rank"]
    )
    df["prompt_relevance_score"] = df["prompt_relevance_score"].clip(0.0, 1.0)

    return df.sort_values("prompt_relevance_score", ascending=False)


def run(subject_ids=None):
    if subject_ids is None:
        subject_ids = list(range(1, 16))

    all_summary = []
    all_features = []

    for sid in subject_ids:
        summary_df, feature_df = analyze_subject(sid)
        if not summary_df.empty:
            all_summary.append(summary_df)
        if not feature_df.empty:
            all_features.append(feature_df)

    if not all_summary:
        logger.warning("No data analyzed.")
        return

    results = pd.concat(all_summary, ignore_index=True)
    feature_results = (
        pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    )

    out_dir = os.path.join(OUTPUT_DIR, "window_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Save core outputs
    results.to_csv(os.path.join(out_dir, "trial_window_stats.csv"), index=False)
    if not feature_results.empty:
        feature_results.to_csv(
            os.path.join(out_dir, "feature_window_stats.csv"), index=False
        )

    # Aggregations
    window_summary = (
        results.groupby("window_size").mean(numeric_only=True).reset_index()
    )
    window_summary.to_csv(os.path.join(out_dir, "window_summary.csv"), index=False)

    compare_df = compare_window_sizes(results, feature_results)
    if not compare_df.empty:
        compare_df.to_csv(os.path.join(out_dir, "window_comparison.csv"), index=False)

    if not feature_results.empty:
        trend_df = aggregate_feature_trends(feature_results)
        trend_df.to_csv(os.path.join(out_dir, "feature_trends.csv"), index=False)

        prompt_audit_df = build_prompt_audit(trend_df)
        prompt_audit_df.to_csv(os.path.join(out_dir, "prompt_audit.csv"), index=False)

        # Also store a feature-level summary pivot by semantic group
        feature_group_map = []
        for group_name, feats in FEATURE_SCHEMA.items():
            for feat in feats:
                feature_group_map.append({"feature": feat, "group": group_name})
        feature_group_df = pd.DataFrame(feature_group_map)

        trend_with_group = trend_df.merge(feature_group_df, on="feature", how="left")
        trend_with_group.to_csv(
            os.path.join(out_dir, "feature_trends_with_group.csv"), index=False
        )

    # Plots
    plot_summary(results, out_dir)

    # Correlation heatmap of averaged feature means across all rows
    if not feature_results.empty:
        # Pivot one row per (subject, trial, window_size), feature -> median
        pivot = feature_results.pivot_table(
            index=["subject", "trial", "window_size"],
            columns="feature",
            values="median",
            aggfunc="mean",
        )
        corr = pivot.corr(method="spearman")

        plt.figure(figsize=(12, 10))
        plt.imshow(corr, interpolation="nearest")
        plt.title("Feature Correlation (Spearman)")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_correlation_heatmap.png"))
        plt.close()

        corr.to_csv(os.path.join(out_dir, "feature_correlation.csv"))

    logger.info(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    run(subject_ids=[1])
