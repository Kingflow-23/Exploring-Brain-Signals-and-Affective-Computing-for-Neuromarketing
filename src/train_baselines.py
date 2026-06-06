"""
Classical ML Baseline Training Pipeline for EEG Emotion Classification.

Complete pipeline for training and evaluating traditional machine learning models on EEG data.

Supported Models:
    - Logistic Regression: Simple linear classifier, baseline model
    - SGDClassifier: Stochastic gradient descent, scalable linear model
    - Random Forest: Ensemble of decision trees, handles non-linearity
    - Extra Trees (Extremely Randomized Trees): Faster variant with more randomness
    - XGBoost: Gradient boosted trees, high predictive power

Training Pipeline:
    1. Load SEED dataset with all subjects
    2. Preprocess EEG into fixed-length windows
    3. Extract deterministic features (374 features per window)
    4. Subject-level train/test split (no subject data leakage)
    5. Standardize features (zero mean, unit variance)
    6. Train each model with cross-validation
    7. Evaluate on held-out test set
    8. Save model weights and metrics

Key Features:
    - No subject leakage: Subjects are split at dataset level, not window level
    - Feature-based approach: Uses extracted EEG features, not raw signals
    - Scikit-learn pipelines: Automatic feature scaling and model training
    - Comprehensive metrics: Accuracy, F1-score, classification report, confusion matrix
    - Model persistence: Trained models saved with joblib
    - Reproducible: Fixed random state for all stochastic operations

Features Used (374 total):
    - Bandpower: 248 features (62 channels × 4 bands)
    - Differential Entropy: 62 features (1 per channel)
    - Power Spectral Density: 4 features (1 per band)
    - Differential Asymmetry: 30 features (left-right differences)
    - Relative Asymmetry: 30 features (normalized asymmetry)

Input Data:
    - Raw EEG: (62, variable_length) per trial
    - Labels: {0: negative, 1: neutral, 2: positive}
    - Subjects: 15 individuals with 45 trials each (15 trials × 3 sessions)

Output:
    - Trained models: model/[model_name]/model.pkl
    - Metrics: model/[model_name]/metrics.json
    - Confusion matrix: model/[model_name]/confusion.npy
"""

import os
import time
import json
import joblib
import logging
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from feature_extraction import extract_features
from config import DATASET_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("BASELINE_TRAINER")


def extract_dataset_features(processed_dataset):
    """
    Convert preprocessed EEG windows into ML-ready feature matrix.

    Parameters
    ----------
    processed_dataset : list
        List of preprocessed samples containing windows and metadata.

    Returns
    -------
    tuple
        (X, y, groups) where X is feature matrix, y is labels, groups is subject IDs
    """

    logger.info("Extracting features using feature_extraction module...")

    X, y, groups = [], [], []

    for sample in tqdm(processed_dataset, desc="Feature extraction"):

        label = sample["label"]
        subject = sample["subject"]

        for window in sample["windows"]:
            feats = extract_features(window)

            X.append(feats)
            y.append(label)
            groups.append(subject)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


# =============================================================================
# MODEL FACTORY
# =============================================================================


def build_models():
    return {
        # -----------------------
        # LINEAR MODELS (SCALED)
        # -----------------------
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        solver="saga",
                    ),
                ),
            ]
        ),
        "sgd_clf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-4,
                        max_iter=3000,
                        tol=1e-3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        # -----------------------
        # TREE MODELS (NO SCALING)
        # -----------------------
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # -----------------------
        # BOOSTING MODELS
        # -----------------------
        "xgboost": XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        ),
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_all_models(X_train, y_train, X_test, y_test):

    logger.info("Training models...")

    results = {}
    models = build_models()

    for name, model in tqdm(models.items(), desc="Training models", unit="model"):

        logger.info(f"Training: {name}")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, preds)
        report_text = classification_report(
            y_test,
            preds,
            zero_division=0,
        )

        report_dict = classification_report(
            y_test,
            preds,
            output_dict=True,
            zero_division=0,
        )

        logger.info(f"{name} acc: {acc:.4f}")
        logger.info("\n" + report_text)

        results[name] = {
            "accuracy": float(acc),
            "classification_report": report_dict,
            "train_time": float(train_time),
            "inference_time": float(infer_time),
        }

        # save model
        model_dir = os.path.join(MODEL_DIR, name)
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

        np.save(
            os.path.join(model_dir, "confusion.npy"), confusion_matrix(y_test, preds)
        )

        with open(os.path.join(model_dir, "metrics.json"), "w") as f:
            json.dump(results[name], f, indent=4)

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():

    logger.info("Starting EEG training pipeline")

    raw = build_seed_dataset(DATASET_DIR)

    processed = preprocess_dataset(raw, normalize=False)

    X, y, groups = extract_dataset_features(processed)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = train_all_models(X_train, y_train, X_test, y_test)

    with open(os.path.join(MODEL_DIR, "benchmark_summary.json"), "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Done. ✅")


if __name__ == "__main__":
    main()
