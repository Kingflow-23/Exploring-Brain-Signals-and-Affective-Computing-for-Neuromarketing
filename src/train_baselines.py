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
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("BASELINE_TRAINER")

# =============================================================================
# UTILS
# =============================================================================


def ensure_model_dir():
    """
    Create model directory if it does not exist.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Model directory ready: {MODEL_DIR}")


def get_model_dir(name: str):
    """
    Create subdirectory for a specific model.
    """
    path = os.path.join(MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)

    logger.info(f"Using model directory: {path}")

    return path


def save_json(path, obj):
    """
    Save dictionary as formatted JSON.
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

    logger.info(f"Saved JSON: {path}")


def save_artifacts(model, name, y_true, y_pred):
    """
    Save:
        - trained model
        - metrics
        - confusion matrix
    """

    logger.info(f"Saving artifacts for model: {name}")

    model_dir = get_model_dir(name)

    # save trained model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    logger.info(f"Saved model: {model_path}")

    # compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")

    save_json(
        metrics_path,
        {
            "accuracy": acc,
            "classification_report": report,
        },
    )

    # save confusion matrix
    cm_path = os.path.join(model_dir, "confusion.npy")
    np.save(cm_path, cm)

    logger.info(f"Saved confusion matrix: {cm_path}")


# =============================================================================
# FEATURE EXTRACTION (CLASSICAL ML ONLY)
# =============================================================================


def flatten_windows(processed_dataset):
    """
    Convert EEG windows into flat vectors for classical ML.

    Input:
        (62, W)

    Output:
        (62 * W,)
    """

    logger.info("Flattening EEG windows into feature vectors...")

    X, y, groups = [], [], []

    total_windows = 0

    for sample in processed_dataset:

        label = sample["label"]
        subject = sample["subject"]

        for window in sample["windows"]:

            X.append(window.reshape(-1))
            y.append(label)
            groups.append(subject)

            total_windows += 1

    logger.info(f"Total windows created: {total_windows}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"groups shape: {groups.shape}")

    return X, y, groups


# =============================================================================
# MODELS
# =============================================================================


def build_models():
    """
    Build baseline classical ML models.
    """

    logger.info("Building baseline models...")

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
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        gamma="scale",
                    ),
                ),
            ]
        ),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


# =============================================================================
# TRAINING
# =============================================================================


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate all baseline models.
    """

    logger.info("Starting model training pipeline...")

    results = {}
    models = build_models()

    logger.info(f"Total models to train: {len(models)}")

    for name, model in models.items():

        logger.info("=" * 80)
        logger.info(f"Training model: {name}")

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        logger.info("Fitting model...")

        t0 = time.time()

        model.fit(X_train, y_train)

        train_time = time.time() - t0

        logger.info(f"Training completed in {train_time:.2f} seconds")

        # ---------------------------------------------------------------------
        # INFERENCE
        # ---------------------------------------------------------------------

        logger.info("Running inference...")

        t0 = time.time()

        preds = model.predict(X_test)

        infer_time = time.time() - t0

        logger.info(f"Inference completed in {infer_time:.4f} seconds")

        # ---------------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------------

        acc = float(accuracy_score(y_test, preds))

        logger.info(f"Accuracy: {acc:.4f}")

        # ---------------------------------------------------------------------
        # SAVE
        # ---------------------------------------------------------------------

        save_artifacts(model, name, y_test, preds)

        results[name] = {
            "accuracy": acc,
            "train_time": float(train_time),
            "inference_time": float(infer_time),
        }

        logger.info(f"Finished model: {name}")

    logger.info("All models trained successfully.")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():

    logger.info("=" * 80)
    logger.info("STARTING BASELINE EEG TRAINING PIPELINE")
    logger.info("=" * 80)

    ensure_model_dir()

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------

    logger.info("Loading SEED dataset...")

    raw = build_seed_dataset(DATASET_DIR)

    logger.info(f"Raw dataset samples: {len(raw)}")

    # -------------------------------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------------------------------

    logger.info("Preprocessing dataset...")

    processed = preprocess_dataset(raw)

    logger.info(f"Processed dataset samples: {len(processed)}")

    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------------------

    X, y, groups = flatten_windows(processed)

    # -------------------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------------------------------

    logger.info("Creating train/test split...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    results = train_all_models(X_train, y_train, X_test, y_test)

    # -------------------------------------------------------------------------
    # SAVE BENCHMARK
    # -------------------------------------------------------------------------

    summary_path = os.path.join(MODEL_DIR, "benchmark_summary.json")

    save_json(summary_path, results)

    logger.info(f"Saved benchmark summary: {summary_path}")

    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
