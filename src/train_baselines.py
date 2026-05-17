# =============================================================================
# BASELINE TRAINING PIPELINE (CLASSICAL ML ONLY)
# =============================================================================

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

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from feature_extraction import extract_features
from config import DATASET_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("BASELINE_TRAINER")


# =============================================================================
# FEATURE EXTRACTION (CLEAN WRAPPER ONLY)
# =============================================================================


def extract_dataset_features(processed_dataset):
    """
    Convert preprocessed EEG windows into ML-ready dataset.

    INPUT:
        processed_dataset:
            [
                {
                    "windows": (N, 62, W),
                    "label": int,
                    "subject": int
                }
            ]

    OUTPUT:
        X: (total_windows, F)
        y: (total_windows,)
        groups: (total_windows,)
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
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        solver="saga",
                    ),
                ),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=25,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_all_models(X_train, y_train, X_test, y_test):

    logger.info("Training models...")

    results = {}
    models = build_models()

    for name, model in tqdm(models.items(), desc="Training models"):

        logger.info(f"Training: {name}")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, preds)

        logger.info(f"{name} acc: {acc:.4f}")

        results[name] = {
            "accuracy": float(acc),
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

    logger.info("Done.")


if __name__ == "__main__":
    main()
