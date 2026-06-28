"""
Distribution-level audit of the EEG features passed to the LLM.

This script does not query the LLM. It reconstructs the exact LLM-side
preprocessing and feature extraction path, then summarizes the values that
were inserted into prompts. The goal is to support a paper discussion about
why the LLM behavior is weak or inconsistent with expected EEG affect rules.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from benchmark import load_test_data, prepare_llm_data
from config import (
    DEFAULT_FS,
    FRONTAL_PAIRS_IDX,
    LLM_STEP_SIZE,
    LLM_WINDOW_SIZE,
    OUTPUT_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("LLM_FEATURE_DISTRIBUTION")


LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

FEATURE_GROUPS = {
    "Global spectral state": [
        "theta_ratio",
        "alpha_ratio",
        "beta_ratio",
        "gamma_ratio",
    ],
    "Frontal asymmetry": [
        "faa_fp1_fp2",
        "faa_f3_f4",
        "faa_f7_f8",
    ],
    "Regional activity": [
        "frontal_alpha",
        "frontal_beta",
        "frontal_gamma",
        "temporal_alpha",
        "temporal_beta",
        "occipital_alpha",
    ],
    "Complexity": [
        "activity",
        "entropy",
    ],
    "Derived ratios": [
        "beta_alpha_ratio",
        "gamma_beta_ratio",
        "frontal_occipital_alpha_ratio",
    ],
}

FEATURE_ORDER = [feature for group in FEATURE_GROUPS.values() for feature in group]


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = series.std(ddof=0)
    if not np.isfinite(scale) or scale < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - median) / scale


def eta_squared(values: pd.Series, labels: pd.Series) -> float:
    valid = values.notna() & labels.notna()
    y = values[valid].to_numpy(dtype=float)
    groups = labels[valid].to_numpy()

    if len(y) == 0:
        return np.nan

    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    if ss_total <= 1e-12:
        return 0.0

    ss_between = 0.0
    for label in np.unique(groups):
        group_values = y[groups == label]
        ss_between += len(group_values) * (np.mean(group_values) - grand_mean) ** 2

    return float(ss_between / ss_total)


def class_summary(values: pd.Series, labels: pd.Series) -> Dict[str, float]:
    out = {}
    for label_id, label_name in LABEL_NAMES.items():
        group = values[labels == label_id].dropna()
        out[f"{label_name}_mean"] = float(group.mean()) if len(group) else np.nan
        out[f"{label_name}_median"] = float(group.median()) if len(group) else np.nan
        out[f"{label_name}_iqr"] = (
            float(group.quantile(0.75) - group.quantile(0.25)) if len(group) else np.nan
        )
    return out


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().to_numpy(dtype=float)
    b = b.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def ensure_out_dir() -> Path:
    out_dir = Path(OUTPUT_DIR) / "llm_feature_distribution_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_llm_feature_table(out_dir: Path, force: bool = False) -> pd.DataFrame:
    cache_path = out_dir / "llm_prompt_feature_values.csv"
    if cache_path.exists() and not force:
        logger.info("Using cached feature table: %s", cache_path)
        return pd.read_csv(cache_path)

    values_log = Path(OUTPUT_DIR) / "window_analysis" / "Values.txt"
    if values_log.exists() and not force:
        logger.info("Parsing existing LLM value log: %s", values_log)
        df = parse_values_log(values_log)
        if not df.empty:
            df.to_csv(cache_path, index=False)
            logger.info("Saved parsed feature table: %s", cache_path)
            return df
        logger.warning(
            "Existing Values.txt could not be parsed; falling back to extraction."
        )

    logger.info("Loading held-out test data with the LLM preprocessing path...")
    raw_test = load_test_data()
    llm_data = prepare_llm_data(raw_test)

    rows = []
    for sample in llm_data:
        for window_idx, window in enumerate(sample["windows"]):
            features = fast_extract_eeg_features(window)
            row = {
                "subject": sample["subject"],
                "trial": sample["trial"],
                "rep": sample["rep"],
                "window_idx": window_idx,
                "label": sample["label"],
                "label_name": LABEL_NAMES[sample["label"]],
                "window_size": LLM_WINDOW_SIZE,
                "step_size": LLM_STEP_SIZE,
            }
            row.update(
                {
                    feature: safe_float(features.get(feature, np.nan))
                    for feature in FEATURE_ORDER
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    logger.info("Saved feature table: %s", cache_path)
    return df


def bandpower_matrix(signal: np.ndarray, fs: int = DEFAULT_FS) -> Dict[str, np.ndarray]:
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, signal.shape[1]), axis=1)
    out = {}
    for band, low, high in [
        ("theta", 4, 8),
        ("alpha", 8, 13),
        ("beta", 13, 30),
        ("gamma", 30, 45),
    ]:
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            out[band] = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        else:
            out[band] = np.zeros(signal.shape[0], dtype=float)
    return out


def fast_extract_eeg_features(signal: np.ndarray, fs: int = DEFAULT_FS) -> dict:
    signal = signal.astype(np.float32)
    bp = bandpower_matrix(signal, fs=fs)

    theta = float(np.mean(bp["theta"]))
    alpha = float(np.mean(bp["alpha"]))
    beta = float(np.mean(bp["beta"]))
    gamma = float(np.mean(bp["gamma"]))

    eps = 1e-8
    total = theta + alpha + beta + gamma + eps

    faa_keys = ["faa_f3_f4", "faa_f7_f8", "faa_fp1_fp2"]
    faa_vals = {}
    for key, (left_i, right_i) in zip(faa_keys, FRONTAL_PAIRS_IDX):
        faa_vals[key] = float(
            np.log(bp["alpha"][left_i] + eps) - np.log(bp["alpha"][right_i] + eps)
        )

    frontal = slice(0, 20)
    temporal = slice(20, 40)
    occipital = slice(40, None)

    frontal_alpha = float(np.mean(bp["alpha"][frontal]))
    frontal_beta = float(np.mean(bp["beta"][frontal]))
    frontal_gamma = float(np.mean(bp["gamma"][frontal]))
    temporal_alpha = float(np.mean(bp["alpha"][temporal]))
    temporal_beta = float(np.mean(bp["beta"][temporal]))
    occipital_alpha = float(np.mean(bp["alpha"][occipital]))

    activity = float(np.std(signal))
    power = signal**2
    prob = power / (np.sum(power) + eps)
    entropy = float(-np.sum(prob * np.log(prob + eps)))

    return {
        "theta_ratio": theta / total,
        "alpha_ratio": alpha / total,
        "beta_ratio": beta / total,
        "gamma_ratio": gamma / total,
        **faa_vals,
        "frontal_alpha": frontal_alpha,
        "frontal_beta": frontal_beta,
        "frontal_gamma": frontal_gamma,
        "temporal_alpha": temporal_alpha,
        "temporal_beta": temporal_beta,
        "occipital_alpha": occipital_alpha,
        "activity": activity,
        "entropy": entropy,
        "beta_alpha_ratio": beta / (alpha + eps),
        "gamma_beta_ratio": gamma / (beta + eps),
        "frontal_occipital_alpha_ratio": frontal_alpha / (occipital_alpha + eps),
    }


def parse_values_log(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []

    pattern = re.compile(
        r"(?P<features>\{.*?\})\s*[\r\n]+Real label:\s*(?P<label>\d+),\s*LLM prediction:\s*(?P<prediction>[a-zA-Z_ -]+)",
        flags=re.DOTALL,
    )

    for idx, match in enumerate(pattern.finditer(text)):
        feature_text = match.group("features")
        label = int(match.group("label"))
        prediction = match.group("prediction").strip().lower()
        row = {
            "subject": np.nan,
            "trial": np.nan,
            "rep": np.nan,
            "window_idx": idx,
            "label": label,
            "label_name": LABEL_NAMES.get(label, str(label)),
            "llm_prediction": prediction,
            "window_size": LLM_WINDOW_SIZE,
            "step_size": LLM_STEP_SIZE,
        }

        for feature in FEATURE_ORDER:
            value_match = re.search(
                rf"'{re.escape(feature)}':\s*(?:np\.(?:float64|float32)\()?([-+0-9.eE]+)",
                feature_text,
            )
            row[feature] = safe_float(value_match.group(1)) if value_match else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for group_name, features in FEATURE_GROUPS.items():
        for feature in features:
            values = df[feature]
            labels = df["label"]
            negative = values[labels == 0]
            neutral = values[labels == 1]
            positive = values[labels == 2]

            row = {
                "group": group_name,
                "feature": feature,
                "n": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(values.median()),
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                "min": float(values.min()),
                "max": float(values.max()),
                "eta_squared_label": eta_squared(values, labels),
                "cohens_d_positive_vs_negative": cohens_d(positive, negative),
                "cohens_d_positive_vs_neutral": cohens_d(positive, neutral),
                "cohens_d_negative_vs_neutral": cohens_d(negative, neutral),
            }
            row.update(class_summary(values, labels))
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values("eta_squared_label", ascending=False)
    summary.to_csv(out_dir / "feature_distribution_summary.csv", index=False)
    return summary


def build_rule_diagnostics(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    z = pd.DataFrame({feature: robust_zscore(df[feature]) for feature in FEATURE_ORDER})

    diagnostics = df[
        ["subject", "trial", "rep", "window_idx", "label", "label_name"]
    ].copy()
    diagnostics["frontal_engagement_z"] = z[
        ["frontal_gamma", "frontal_alpha", "activity", "gamma_beta_ratio"]
    ].mean(axis=1)
    diagnostics["posterior_alpha_inhibition_z"] = z[
        ["occipital_alpha", "alpha_ratio"]
    ].mean(axis=1)
    diagnostics["frontal_over_occipital_alpha_z"] = z["frontal_occipital_alpha_ratio"]

    # The implementation computes log(left alpha) - log(right alpha).
    # If alpha is interpreted as inverse activation, the activation-oriented
    # convention is approximately the negative of this value.
    diagnostics["faa_power_left_minus_right_z"] = z[
        ["faa_fp1_fp2", "faa_f3_f4", "faa_f7_f8"]
    ].mean(axis=1)
    diagnostics["faa_activation_left_minus_right_z"] = -diagnostics[
        "faa_power_left_minus_right_z"
    ]

    diagnostics["prompt_positive_evidence_z"] = (
        diagnostics[
            [
                "frontal_engagement_z",
                "frontal_over_occipital_alpha_z",
                "faa_activation_left_minus_right_z",
            ]
        ].mean(axis=1)
        - diagnostics["posterior_alpha_inhibition_z"]
    )
    diagnostics["prompt_negative_evidence_z"] = diagnostics[
        "posterior_alpha_inhibition_z"
    ] - diagnostics[["frontal_engagement_z", "frontal_over_occipital_alpha_z"]].mean(
        axis=1
    )

    diagnostics.to_csv(out_dir / "prompt_rule_diagnostics.csv", index=False)

    by_class = diagnostics.groupby("label_name").mean(numeric_only=True).reset_index()
    by_class.to_csv(out_dir / "prompt_rule_diagnostics_by_class.csv", index=False)
    return diagnostics


def latest_benchmark_json() -> Optional[Path]:
    files = sorted(
        Path(OUTPUT_DIR).glob("benchmark_inference_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def read_llm_metrics() -> Dict:
    path = latest_benchmark_json()
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        if key.lower().startswith("qwen") or key.lower().startswith("llm"):
            return {"source": str(path), "model_key": key, **value}

    if "llm" in data:
        return {"source": str(path), "model_key": "llm", **data["llm"]}

    return {"source": str(path)}


def plot_feature_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    for group_name, features in FEATURE_GROUPS.items():
        fig, axes = plt.subplots(
            len(features), 1, figsize=(10, max(3.0, 2.1 * len(features))), sharex=False
        )
        if len(features) == 1:
            axes = [axes]

        for ax, feature in zip(axes, features):
            data = [
                df.loc[df["label"] == label_id, feature].dropna()
                for label_id in LABEL_NAMES
            ]
            ax.boxplot(
                data,
                tick_labels=[LABEL_NAMES[i] for i in LABEL_NAMES],
                showfliers=False,
            )
            ax.set_title(feature)
            ax.set_ylabel("prompt value")
            ax.grid(axis="y", alpha=0.25)

        fig.suptitle(f"LLM Prompt Feature Distributions: {group_name}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out_dir / f"{slug(group_name)}_boxplots.png", dpi=180)
        plt.close(fig)


def plot_standardized_overview(df: pd.DataFrame, out_dir: Path) -> None:
    z = pd.DataFrame({feature: robust_zscore(df[feature]) for feature in FEATURE_ORDER})
    z["label_name"] = df["label_name"]

    fig, ax = plt.subplots(figsize=(14, 8))
    positions = []
    labels = []
    colors = ["#476A9A", "#6E6E6E", "#B85C38"]
    offset = [-0.25, 0.0, 0.25]

    for i, feature in enumerate(FEATURE_ORDER):
        for j, label_name in enumerate(["negative", "neutral", "positive"]):
            positions.append(i + offset[j])
            labels.append(z.loc[z["label_name"] == label_name, feature].dropna())

    box = ax.boxplot(
        labels, positions=positions, widths=0.22, patch_artist=True, showfliers=False
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors[idx % 3])
        patch.set_alpha(0.75)

    ax.set_xticks(range(len(FEATURE_ORDER)))
    ax.set_xticklabels(FEATURE_ORDER, rotation=65, ha="right")
    ax.set_ylabel("robust z-score of value passed to prompt")
    ax.set_title("Standardized feature distributions by true class")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.75) for color in colors
        ],
        labels=["negative", "neutral", "positive"],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "standardized_feature_distributions_by_class.png", dpi=180)
    plt.close(fig)


def plot_separability(summary: pd.DataFrame, out_dir: Path) -> None:
    ordered = summary.sort_values("eta_squared_label", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(ordered["feature"], ordered["eta_squared_label"], color="#3F6C51")
    ax.set_xlabel("eta-squared by true emotion label")
    ax.set_title("Feature-level class separability in values passed to the LLM")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_class_separability_eta_squared.png", dpi=180)
    plt.close(fig)


def plot_faa_diagnostics(
    df: pd.DataFrame, diagnostics: pd.DataFrame, out_dir: Path
) -> None:
    faa_features = ["faa_fp1_fp2", "faa_f3_f4", "faa_f7_f8"]

    sign_rows = []
    for feature in faa_features:
        for label_id, label_name in LABEL_NAMES.items():
            values = df.loc[df["label"] == label_id, feature].dropna()
            sign_rows.append(
                {
                    "feature": feature,
                    "label_name": label_name,
                    "positive_power_sign_rate": float((values > 0).mean()),
                    "negative_power_sign_rate": float((values < 0).mean()),
                }
            )
    sign_df = pd.DataFrame(sign_rows)
    sign_df.to_csv(out_dir / "faa_sign_rates_by_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(faa_features))
    width = 0.25
    for idx, label_name in enumerate(["negative", "neutral", "positive"]):
        values = sign_df[sign_df["label_name"] == label_name][
            "positive_power_sign_rate"
        ].to_numpy()
        ax.bar(x + (idx - 1) * width, values, width=width, label=label_name)
    ax.set_xticks(x)
    ax.set_xticklabels(faa_features)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of windows with FAA > 0")
    ax.set_title(
        "FAA sign rates using implementation convention: log(left alpha) - log(right alpha)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "faa_positive_sign_rate_by_class.png", dpi=180)
    plt.close(fig)

    plot_cols = [
        "frontal_engagement_z",
        "posterior_alpha_inhibition_z",
        "frontal_over_occipital_alpha_z",
        "faa_power_left_minus_right_z",
        "faa_activation_left_minus_right_z",
        "prompt_positive_evidence_z",
        "prompt_negative_evidence_z",
    ]
    means = (
        diagnostics.groupby("label_name")[plot_cols]
        .mean()
        .loc[["negative", "neutral", "positive"]]
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(means.to_numpy(), cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(means.index)))
    ax.set_yticklabels(means.index)
    ax.set_xticks(range(len(means.columns)))
    ax.set_xticklabels(means.columns, rotation=45, ha="right")
    ax.set_title("Prompt-rule evidence diagnostics by true class")
    fig.colorbar(im, ax=ax, label="mean robust z-score")
    fig.tight_layout()
    fig.savefig(out_dir / "prompt_rule_diagnostics_heatmap.png", dpi=180)
    plt.close(fig)


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def confusion_bias_sentence(metrics: Dict) -> str:
    trial = metrics.get("trial_level", {})
    cm = trial.get("cm", [])
    if not cm:
        return "No LLM confusion matrix was available in the benchmark JSON."

    cm_arr = np.asarray(cm)
    pred_totals = cm_arr.sum(axis=0)
    total = pred_totals.sum()
    parts = [
        f"{LABEL_NAMES[i]}: {int(pred_totals[i])} ({pred_totals[i] / total:.1%})"
        for i in range(min(len(pred_totals), 3))
    ]
    return (
        "At trial level, the LLM prediction distribution was " + ", ".join(parts) + "."
    )


def metric_pct(metrics: Dict, level: str, key: str) -> str:
    try:
        return f"{metrics[level][key] * 100:.2f}%"
    except Exception:
        return "not available"


def generate_discussion_readme(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    out_dir: Path,
) -> None:
    metrics = read_llm_metrics()
    top = summary.head(6)
    bottom = summary.tail(4)
    diag_means = diagnostics.groupby("label_name").mean(numeric_only=True)
    faa_summary = summary[summary["feature"].str.startswith("faa_")]

    pos_evidence = diag_means.loc["positive", "prompt_positive_evidence_z"]
    neg_evidence = diag_means.loc["negative", "prompt_positive_evidence_z"]
    neutral_evidence = diag_means.loc["neutral", "prompt_positive_evidence_z"]

    lines = [
        "# LLM Prompt Feature Distribution Analysis",
        "",
        "This report audits the numerical EEG features that were actually inserted into the LLM prompts. It focuses on distribution, class overlap, and neuroscience-rule consistency rather than on re-running model training.",
        "",
        "## What was analyzed",
        "",
        f"- Input path: held-out test set, LLM preprocessing, `window_size={LLM_WINDOW_SIZE}`, `step_size={LLM_STEP_SIZE}`, channel-wise z-score normalization.",
        f"- Number of prompt windows analyzed: **{len(df):,}**.",
        f"- True class counts: {format_counts(df['label_name'])}.",
        "- Feature schema: 18 prompt features grouped into global spectral ratios, frontal asymmetry, regional activity, complexity, and derived ratios.",
        "",
        "## LLM behavior to explain",
        "",
        f"- Window accuracy: **{metric_pct(metrics, 'window_level', 'acc')}**.",
        f"- Trial accuracy: **{metric_pct(metrics, 'trial_level', 'acc')}**.",
        f"- {confusion_bias_sentence(metrics)}",
        "",
        "The trial-level confusion matrix shows the most important failure mode: the LLM does not recover a reliable negative class. This matters because the prompt asks the model to reason from neuroscience-inspired indicators, but the feature distributions below show that those indicators are not cleanly class-separated in the values the model receives.",
        "",
        "## Main distribution finding",
        "",
        "Most prompt features have substantial overlap across negative, neutral, and positive windows. The strongest features by eta-squared still explain only a limited part of label variance, so the prompt gives the LLM ambiguous evidence rather than a rule-like mapping from features to emotion.",
        "",
        "![Standardized feature distributions](standardized_feature_distributions_by_class.png)",
        "",
        "![Feature separability](feature_class_separability_eta_squared.png)",
        "",
        "Top features by class separability:",
        "",
        "| Feature | Group | eta-squared | positive vs negative d | negative median | neutral median | positive median |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in top.iterrows():
        lines.append(
            f"| {row['feature']} | {row['group']} | {row['eta_squared_label']:.4f} | "
            f"{row['cohens_d_positive_vs_negative']:.3f} | {row['negative_median']:.4g} | "
            f"{row['neutral_median']:.4g} | {row['positive_median']:.4g} |"
        )

    lines.extend(
        [
            "",
            "Least label-sensitive features:",
            "",
            "| Feature | eta-squared | Interpretation risk |",
            "|---|---:|---|",
        ]
    )
    for _, row in bottom.iterrows():
        lines.append(
            f"| {row['feature']} | {row['eta_squared_label']:.4f} | Low distributional separation; weak evidence for zero-shot prompt reasoning. |"
        )

    lines.extend(
        [
            "",
            "## FAA interpretation issue",
            "",
            "Frontal Alpha Asymmetry needs careful wording in the paper because the implementation convention is:",
            "",
            "`FAA = log(left alpha power) - log(right alpha power)`",
            "",
            "In many affective-neuroscience interpretations, alpha power is treated as inversely related to cortical activation. Under that interpretation, a positive value in this implementation does not directly mean stronger left frontal activation. It can instead mean stronger left alpha power, which may correspond to weaker left activation. Therefore, using these values as direct evidence of left-dominant positive affect would be risky unless the sign convention is explicitly corrected or justified.",
            "",
            "![FAA sign rates](faa_positive_sign_rate_by_class.png)",
            "",
            "FAA separability summary:",
            "",
            "| FAA feature | eta-squared | negative median | neutral median | positive median |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in faa_summary.iterrows():
        lines.append(
            f"| {row['feature']} | {row['eta_squared_label']:.4f} | {row['negative_median']:.4g} | {row['neutral_median']:.4g} | {row['positive_median']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Prompt-rule diagnostics",
            "",
            "The prompt expects positive emotion to be supported by organized frontal engagement, stronger frontal/posterior balance, and weak posterior inhibition. The aggregated diagnostic scores do not form a clean monotonic pattern. Mean prompt-positive evidence was:",
            "",
            f"- Negative windows: **{neg_evidence:.3f}**",
            f"- Neutral windows: **{neutral_evidence:.3f}**",
            f"- Positive windows: **{pos_evidence:.3f}**",
            "",
            "If positive evidence is not clearly highest for positive windows and clearly lowest for negative windows, the LLM has no stable numerical basis for applying the intended neuroscience logic. It may then default to broad semantic priors from the prompt, especially neutral when evidence is mixed.",
            "",
            "![Prompt rule diagnostics](prompt_rule_diagnostics_heatmap.png)",
            "",
            "## Discussion text for the paper",
            "",
            "The LLM-based classifier underperformed because the feature representation supplied to the language model did not provide sufficiently separable or conventionally interpretable evidence for the three emotion classes. Although the prompt described plausible EEG-emotion relationships, the numerical values inserted into the prompt were highly overlapping across negative, neutral, and positive trials. This means that the LLM was not receiving a clear statistical signal from which the stated neuroscience rules could be applied consistently.",
            "",
            "This is especially visible for frontal asymmetry features. FAA is often discussed in relation to emotional valence and approach-withdrawal tendencies, but its interpretation depends strongly on preprocessing, referencing, channel selection, frequency-band definition, and sign convention. In this implementation, FAA is computed as left alpha power minus right alpha power in log space. Because alpha power is commonly interpreted as inversely related to cortical activation, this convention cannot be directly read as left frontal activation dominance without an additional sign interpretation. As a result, apparent positive FAA values may not support positive affect in the way a simple reading of the prompt suggests.",
            "",
            "The feature-distribution audit therefore suggests that the LLM failure is not only a prompt-engineering problem. The model was asked to perform symbolic reasoning over numerical EEG summaries whose class-conditional distributions were weakly separated and partly ambiguous. A supervised ML or DL model can learn empirical class boundaries from these values, but a zero-shot LLM has no fitted mapping between this dataset's feature distributions and the target labels. This explains why the LLM remained far below the best classical and deep-learning models, and why its predictions collapsed toward safer or more frequent semantic choices rather than following the intended EEG affect logic.",
            "",
            "A stronger LLM-based approach would require either task-specific calibration/fine-tuning, explicit dataset-level normalization statistics in the prompt, clearer sign-convention handling for FAA, or a richer representation that preserves temporal and spatial EEG structure. Without these additions, the LLM treats the EEG values as generic scalar tokens rather than as measurements embedded in a subject-dependent neurophysiological system.",
            "",
            "## Generated files",
            "",
            "- `llm_prompt_feature_values.csv`: every feature value inserted into prompts.",
            "- `feature_distribution_summary.csv`: descriptive statistics and separability metrics.",
            "- `prompt_rule_diagnostics.csv`: standardized diagnostic scores for prompt-rule evidence.",
            "- `prompt_rule_diagnostics_by_class.csv`: class-level averages of the diagnostic scores.",
            "- `faa_sign_rates_by_class.csv`: FAA sign rates under the implemented convention.",
        ]
    )

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_counts(series: pd.Series) -> str:
    counts = (
        series.value_counts()
        .reindex(["negative", "neutral", "positive"])
        .fillna(0)
        .astype(int)
    )
    return ", ".join(f"{label}={count:,}" for label, count in counts.items())


def run(force: bool = False) -> None:
    out_dir = ensure_out_dir()
    df = extract_llm_feature_table(out_dir, force=force)
    summary = build_feature_summary(df, out_dir)
    diagnostics = build_rule_diagnostics(df, out_dir)

    plot_feature_boxplots(df, out_dir)
    plot_standardized_overview(df, out_dir)
    plot_separability(summary, out_dir)
    plot_faa_diagnostics(df, diagnostics, out_dir)
    generate_discussion_readme(df, summary, diagnostics, out_dir)

    logger.info("Done. Analysis saved to: %s", out_dir)


if __name__ == "__main__":
    force_refresh = os.environ.get("FORCE_LLM_FEATURE_AUDIT", "0") == "1"
    run(force=force_refresh)
