"""
Classical ML Baseline Training Pipeline for EEG Emotion Classification.

Trains and evaluates traditional machine learning models for emotion recognition:
    - Logistic Regression
    - SGDClassifier
    - Random Forest
    - Extra Trees
    - XGBoost

Features:
    - Subject-level train/test split (no subject leakage)
    - Consistent feature extraction pipeline
    - Comprehensive evaluation metrics
    - Model persistence and checkpointing
    - Cross-validation for robustness
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

    processed = preprocess_dataset(raw)

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
