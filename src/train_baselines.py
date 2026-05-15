# =============================================================================
# BASELINE TRAINING PIPELINE (CLASSICAL ML ONLY)
# =============================================================================

import os
import time
import json
import joblib
import logging
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from config import DATASET_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("BASELINE_TRAINER")


# =============================================================================
# UTILS
# =============================================================================


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_model_dir(name: str):
    path = os.path.join(MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def save_artifacts(model, name, y_true, y_pred):
    model_dir = get_model_dir(name)

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    save_json(
        os.path.join(model_dir, "metrics.json"),
        {"accuracy": acc, "classification_report": report},
    )

    np.save(os.path.join(model_dir, "confusion.npy"), cm)


# =============================================================================
# FEATURE EXTRACTION (CLASSICAL ML ONLY)
# =============================================================================


def flatten_windows(processed_dataset):
    """
    Classical ML representation:
        (62, W) → (62*W,)
    """
    X, y, groups = [], [], []

    for sample in processed_dataset:
        label = sample["label"]
        subject = sample["subject"]

        for window in sample["windows"]:
            X.append(window.reshape(-1))
            y.append(label)
            groups.append(subject)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups, dtype=np.int64),
    )


# =============================================================================
# MODELS
# =============================================================================


def build_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", class_weight="balanced", gamma="scale")),
            ]
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


# =============================================================================
# TRAINING
# =============================================================================


def train_all_models(X_train, y_train, X_test, y_test):
    results = {}
    models = build_models()

    for name, model in models.items():
        logger.info(f"Training {name}")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - t0

        acc = float(accuracy_score(y_test, preds))

        save_artifacts(model, name, y_test, preds)

        results[name] = {
            "accuracy": acc,
            "train_time": float(train_time),
            "inference_time": float(infer_time),
        }

        logger.info(f"{name}: acc={acc:.4f}")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    ensure_model_dir()

    raw = build_seed_dataset(DATASET_DIR)
    processed = preprocess_dataset(raw)

    X, y, groups = flatten_windows(processed)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = train_all_models(X_train, y_train, X_test, y_test)

    save_json(os.path.join(MODEL_DIR, "benchmark_summary.json"), results)


if __name__ == "__main__":
    main()
