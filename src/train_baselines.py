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

from scipy.signal import butter, filtfilt

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
# EEG BAND FILTERING
# =============================================================================


FS = 200
NYQ = 0.5 * FS
ORDER = 5


def create_filters():
    """
    Create bandpass Butterworth filters for EEG frequency bands.

    Returns
    -------
    dict
        Mapping of band name → (b, a) filter coefficients.

    Notes
    -----
    Bands used:
        - theta: 4–8 Hz
        - alpha: 8–13 Hz
        - beta : 13–30 Hz
        - gamma: 30–45 Hz

    Each filter is designed using a 5th-order Butterworth design
    normalized by Nyquist frequency (FS/2).
    """
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    filters = {}

    for name, (low, high) in bands.items():
        b, a = butter(ORDER, [low / NYQ, high / NYQ], btype="band")
        filters[name] = (b, a)

    return filters


FILTERS = create_filters()


def extract_features(window):
    """
    Extract bandpower features from a single EEG window.

    Parameters
    ----------
    window : np.ndarray
        EEG segment of shape (n_channels, n_samples),
        typically (62, T) for SEED dataset.

    Returns
    -------
    np.ndarray
        Feature vector of shape (62 * 4,) containing log bandpower
        for theta, alpha, beta, and gamma bands concatenated.

    Processing Steps
    ----------------
    1. Apply Butterworth bandpass filter per frequency band
    2. Compute signal power per channel: mean(x^2)
    3. Apply log transform for numerical stability
    4. Concatenate all band features into a single vector
    """
    feats = np.empty(62 * 4, dtype=np.float32)

    bands = ["theta", "alpha", "beta", "gamma"]

    i = 0
    for bname in bands:
        b, a = FILTERS[bname]
        sig = filtfilt(b, a, window, axis=1)
        feats[i : i + 62] = np.log(np.mean(sig**2, axis=1) + 1e-8)
        i += 62

    return feats


# =============================================================================
# UTILS
# =============================================================================


def ensure_model_dir():
    """
    Ensure that the global model output directory exists.

    Creates the directory defined in config.MODEL_DIR if missing.
    This is the root folder for all trained model artifacts.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Model directory ready: {MODEL_DIR}")


def get_model_dir(name: str):
    """
    Create and return a model-specific output directory.

    Parameters
    ----------
    name : str
        Name of the model (e.g., 'svm_rbf').

    Returns
    -------
    str
        Absolute path to the model's artifact directory.

    Notes
    -----
    Used to isolate artifacts per model:
        MODEL_DIR/
            ├── svm_rbf/
            ├── knn/
            └── random_forest/
    """
    path = os.path.join(MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)

    logger.info(f"Using model directory: {path}")

    return path


def save_json(path, obj):
    """
    Save a Python dictionary as a formatted JSON file.

    Parameters
    ----------
    path : str
        Destination file path.
    obj : dict
        JSON-serializable object.

    Notes
    -----
    Used for:
        - metrics.json
        - benchmark_summary.json
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

    logger.info(f"Saved JSON: {path}")


def save_artifacts(model, name, y_true, y_pred):
    """
    Persist trained model and evaluation outputs.

    Parameters
    ----------
    model : sklearn estimator
        Trained model pipeline.
    name : str
        Model identifier.
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Model predictions.

    Outputs
    -------
    Saved in MODEL_DIR/<name>/:
        - model.pkl            (serialized model)
        - metrics.json         (accuracy + classification report)
        - confusion.npy        (raw confusion matrix)

    Notes
    -----
    This function is the final checkpoint of each training run.
    It ensures full reproducibility of model performance.
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


def extract_sample_features(sample):
    """
    Extract features for all EEG windows in a single sample.

    Parameters
    ----------
    sample : dict
        Dictionary containing:
            - "windows": list/array of EEG windows (n_windows, 62, T)

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_windows, 62 * 4)

    Notes
    -----
    Each window is independently converted into a bandpower vector.
    """
    windows = np.asarray(sample["windows"])  # (n_windows, 62, T)

    feats = np.zeros((len(windows), 62 * 4), dtype=np.float32)

    for i, window in enumerate(windows):
        feats[i] = extract_features(window)

    return feats


def extract_dataset_features(processed_dataset):
    """
    Convert full EEG dataset into supervised learning arrays.

    Parameters
    ----------
    processed_dataset : list of dict
        Each entry contains:
            - "windows": EEG segments
            - "label": class label
            - "subject": subject ID

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (N_samples, 62 * 4)
    y : np.ndarray
        Labels of shape (N_samples,)
    groups : np.ndarray
        Subject IDs for group-aware splitting

    Notes
    -----
    - Each window becomes one training sample.
    - Group labels are used to prevent subject leakage.
    """
    logger.info("Extracting bandpower EEG features...")

    X, y, groups = [], [], []

    for sample in processed_dataset:

        label = sample["label"]
        subject = sample["subject"]

        feats = extract_sample_features(sample)

        X.append(feats)
        y.extend([label] * len(feats))
        groups.extend([subject] * len(feats))

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = np.asarray(y, dtype=np.int32)
    groups = np.asarray(groups, dtype=np.int32)

    logger.info(f"Feature shape: {X.shape}")
    return X, y, groups


# =============================================================================
# MODELS
# =============================================================================


def build_models():
    """
    Construct baseline classical machine learning models.

    Returns
    -------
    dict
        Mapping of model name → sklearn Pipeline

    Models included
    ---------------
    - Logistic Regression (saga solver, balanced weights)
    - SVM (RBF kernel)
    - KNN (distance-weighted)
    - Random Forest (200 trees)

    Notes
    -----
    All models (except RF) include StandardScaler preprocessing.
    Designed as baseline comparisons for EEG classification.
    """

    logger.info("Building EEG bandpower models...")

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
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", gamma="scale", class_weight="balanced")),
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
                )
            ]
        ),
    }


# =============================================================================
# TRAINING
# =============================================================================


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate all baseline models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels

    Returns
    -------
    dict
        Model-wise results containing:
            - accuracy
            - training time
            - inference time

    Process
    -------
    For each model:
        1. Fit on training data
        2. Predict test data
        3. Compute metrics
        4. Save artifacts to disk

    Purpose
    -------
    Provides a reproducible benchmarking suite for EEG classification.
    """

    logger.info("Starting model training pipeline...")

    results = {}
    models = build_models()

    logger.info(f"Total models to train: {len(models)}")

    for name, model in models.items():

        logger.info("=" * 80)
        logger.info(f"Training model: {name}")

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

    logger.info("All models trained successfully. ✅")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Execute full EEG baseline training pipeline.

    Pipeline Stages
    --------------
    1. Initialize logging and directories
    2. Load raw SEED dataset
    3. Apply preprocessing pipeline
    4. Extract bandpower features
    5. Perform group-aware train/test split
    6. Train multiple classical ML models
    7. Save benchmark summary

    Output
    ------
    - Trained models per algorithm
    - Per-model metrics and confusion matrices
    - Global benchmark summary JSON

    Purpose
    -------
    Provides a complete classical ML baseline for EEG emotion classification.
    """

    logger.info("=" * 80)
    logger.info("STARTING BASELINE EEG TRAINING PIPELINE")
    logger.info("=" * 80)

    ensure_model_dir()

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------

    raw = build_seed_dataset(DATASET_DIR)

    # -------------------------------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------------------------------

    processed = preprocess_dataset(raw)

    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------------------

    X, y, groups = extract_dataset_features(processed)

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
    logger.info("TRAINING PIPELINE FINISHED ✅")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
