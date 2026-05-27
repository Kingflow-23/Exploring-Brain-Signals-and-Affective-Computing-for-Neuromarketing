import os
import json
import torch
import joblib
import logging

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from feature_extraction import extract_dataset_features

from train_deep_models import EEGDataset
from torch.utils.data import DataLoader

from collections import Counter, defaultdict

from datetime import datetime
from model_registry import get_model

from config import (
    DATASET_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    LABELS_MAPPER,
    WINDOW_SIZE,
    ML_WINDOW_SIZE,
    LLM_WINDOW_SIZE,
    STEP_SIZE,
    ML_STEP_SIZE,
    LLM_STEP_SIZE,
)

from llm_inference import LMStudioClient, extract_eeg_features, build_eeg_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BENCHMARK_INFER")


# =========================================================
# LOAD TEST SET ONLY
# =========================================================


def load_test_data():
    """
    Load and preprocess held-out EEG test dataset.

    Pipeline:
        1. Load raw EEG recordings from dedicated test folder
        2. Reconstruct original data structure (subjects, trials, labels)

    IMPORTANT:
        This function assumes the dataset has already been split
        physically on disk to avoid any subject leakage.

    Returns
    -------
    list
        List of preprocessed EEG samples:
            [
                {
                    "subject": int,
                    "trial": int,
                    "label": int,
                    "windows": list of np.array
                },
    """
    test_path = os.path.join(DATASET_DIR, "test")
    logger.info(f"Loading test set from {test_path}")

    raw = build_seed_dataset(test_path)

    return raw


def prepare_ml_data(raw):
    return preprocess_dataset(raw, window_size=ML_WINDOW_SIZE, step_size=ML_STEP_SIZE)


def prepare_dl_data(raw):
    return preprocess_dataset(raw, window_size=WINDOW_SIZE, step_size=STEP_SIZE)


def prepare_llm_data(raw):
    return preprocess_dataset(raw, window_size=LLM_WINDOW_SIZE, step_size=LLM_STEP_SIZE)

def majority_vote(predictions):
    """
    Majority voting over window predictions.
    """

    if len(predictions) == 0:
        return None

    return Counter(predictions).most_common(1)[0][0]


def build_metrics(y_true, y_pred):
    """
    Standard metric bundle.
    """

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def extract_window_features(window, label, subject):
    """
    Extract features for a single EEG window.
    
    Encapsulates feature extraction logic to avoid repeated
    dataset pipeline initialization per window.
    
    Parameters
    ----------
    window : np.ndarray
        Single EEG window
    label : int
        Emotion label (0=negative, 1=neutral, 2=positive)
    subject : int
        Subject ID
    
    Returns
    -------
    np.ndarray
        Extracted features for the window
    """
    feats = extract_dataset_features(
        [{
            "windows": [window],
            "label": label,
            "subject": subject,
        }]
    )[0][0]
    return feats

# =========================================================
# ML INFERENCE
# =========================================================


def run_ml_inference(processed_test):
    """
    Run inference benchmark for classical ML EEG models.

    Workflow:
        1. Extract deterministic handcrafted EEG features
        2. Load all serialized sklearn models
        3. Run predictions on full test feature matrix
        4. Compute benchmark metrics

    Models evaluated:
        - Logistic Regression
        - SGDClassifier
        - Random Forest
        - Extra Trees
        - XGBoost
        - Any additional sklearn-compatible saved model

    Parameters
    ----------
    processed_test : list
        Preprocessed EEG dataset.

    Returns
    -------
    dict
        Per-model benchmark results:
            {
                model_name: {
                    "window_level": {"acc": float, "f1": float, "cm": list},
                    "trial_level": {"acc": float, "f1": float, "cm": list}
                }
            }
    """

    logger.info("ML inference running...")

    if not processed_test:
        logger.warning("No test data provided for ML inference")
        return {}

    results = {}

    model_names = os.listdir(MODEL_DIR)

    for name in tqdm(model_names, desc="ML models"):

        model_path = os.path.join(MODEL_DIR, name, "model.pkl")

        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        
        # =====================================================
        # BUILD WINDOW FEATURES + TRACK TRIAL IDS
        # =====================================================
        
        X = []
        y = []
        trial_ids = []

        for sample in processed_test:

            for window in sample["windows"]:

                feats = extract_window_features(
                    window,
                    sample["label"],
                    sample["subject"],
                )

                X.append(feats)
                y.append(sample["label"])
                trial_ids.append(sample["trial"])

        if not X:
            logger.warning(f"No features extracted for model {name}")
            continue

        preds = model.predict(X)

        # =====================================================
        # WINDOW LEVEL
        # =====================================================

        window_metrics = build_metrics(y, preds)

        # =====================================================
        # TRIAL LEVEL (Majority Vote)
        # =====================================================

        grouped_preds = defaultdict(list)
        grouped_targets = {}

        for pred, target, tid in zip(preds, y, trial_ids):
            grouped_preds[tid].append(pred)
            grouped_targets[tid] = target

        trial_preds = []
        trial_targets = []

        for tid in sorted(grouped_preds.keys()):
            trial_pred = majority_vote(grouped_preds[tid])
            if trial_pred is not None:
                trial_preds.append(trial_pred)
                trial_targets.append(grouped_targets[tid])

        if trial_preds:
            trial_metrics = build_metrics(trial_targets, trial_preds)
        else:
            logger.warning(f"No valid trial predictions for model {name}")
            trial_metrics = {"acc": 0.0, "f1": 0.0, "cm": []}

        results[name] = {
            "window_level": window_metrics,
            "trial_level": trial_metrics,
        }

    return results


# =========================================================
# DL INFERENCE
# =========================================================


def run_dl_inference(processed_test, model_dir, device):
    """
    Run inference benchmark for saved deep learning EEG models.

    Workflow:
        1. Flatten EEG windows into supervised samples
        2. Build PyTorch DataLoader
        3. Reconstruct original architectures via model registry
        4. Load saved model checkpoints
        5. Run forward inference
        6. Compute evaluation metrics

    Supports:
        - CNN EEG models
        - RNN EEG models
        - Temporal Conv models
        - Hybrid CNN-LSTM models
        - Transformer EEG models

    Important:
        No training occurs here.
        Models are evaluated strictly in inference mode.

    Parameters
    ----------
    processed_test : list
        Preprocessed EEG dataset.

    model_dir : str
        Root model directory containing saved checkpoints.

    device : torch.device
        Inference device (CPU or CUDA).

    Returns
    -------
    dict
        Per-model benchmark results:
            {
                model_name: {
                    "window_level": {"acc": float, "f1": float, "cm": list},
                    "trial_level": {"acc": float, "f1": float, "cm": list}
                }
            }
    """
    logger.info("DL inference running...")

    if not processed_test:
        logger.warning("No test data provided for DL inference")
        return {}

    samples = []

    for s in processed_test:

        for w in s["windows"]:

            samples.append(
                (
                    w,
                    s["label"],
                    s["subject"],
                    s["trial"],
                )
            )

    class TrialEEGDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x, y, subj, tid = self.samples[idx]
            return x, y, tid

    loader = DataLoader(
        TrialEEGDataset(samples),
        batch_size=64,
        shuffle=False
    )

    results = {}

    deep_dir = os.path.join(model_dir, "deep_experiment")

    if not os.path.exists(deep_dir):
        logger.warning(f"Deep model directory not found: {deep_dir}")
        return {}

    for file in os.listdir(deep_dir):

        if not file.endswith("_best.pt"):
            continue

        model_name = file.replace("_best.pt", "")
        model_path = os.path.join(deep_dir, file)

        entry = get_model(model_name)

        model = entry["model"].to(device)

        input_mode = entry["input_mode"]

        state = torch.load(model_path, map_location=device)

        model.load_state_dict(state)

        model.eval()

        preds = []
        targets = []
        trial_ids = []

        with torch.no_grad():

            for xb, yb, tids in loader:

                xb = xb.to(device)

                if input_mode == "eeg_4d":
                    out = model(xb)
                else:
                    out = model(xb.squeeze(1))

                pred = out.argmax(dim=1)

                preds.extend(pred.cpu().numpy())
                targets.extend(yb.numpy())
                trial_ids.extend(tids.numpy())

        if not preds:
            logger.warning(f"No predictions from model {model_name}")
            continue

        # =====================================================
        # WINDOW LEVEL
        # =====================================================

        window_metrics = build_metrics(targets, preds)

        # =====================================================
        # TRIAL LEVEL (Majority Vote)
        # =====================================================

        grouped_preds = defaultdict(list)
        grouped_targets = {}

        for pred, target, tid in zip(preds, targets, trial_ids):
            grouped_preds[tid].append(pred)
            grouped_targets[tid] = target

        trial_preds = []
        trial_targets = []

        for tid in sorted(grouped_preds.keys()):
            trial_pred = majority_vote(grouped_preds[tid])
            if trial_pred is not None:
                trial_preds.append(trial_pred)
                trial_targets.append(grouped_targets[tid])

        if trial_preds:
            trial_metrics = build_metrics(trial_targets, trial_preds)
        else:
            logger.warning(f"No valid trial predictions for model {model_name}")
            trial_metrics = {"acc": 0.0, "f1": 0.0, "cm": []}

        results[model_name] = {
            "window_level": window_metrics,
            "trial_level": trial_metrics,
        }

    return results


# =========================================================
# LLM INFERENCE
# =========================================================


def run_llm_inference(processed_test):
    """
    Run EEG-to-LLM emotion classification benchmark.

    Pipeline:
        1. Extract neuroscience-inspired EEG features
        2. Convert features into structured natural-language prompts
        3. Send prompts to LM Studio local LLM
        4. Parse emotion predictions
        5. Compute benchmark metrics

    Emotion classes:
        - positive
        - neutral
        - negative

    Important:
        This benchmark evaluates whether a language model can
        infer emotional state from symbolic EEG summaries.

        Raw EEG signals are NOT directly fed to the LLM.

    Parameters
    ----------
    processed_test : list
        Preprocessed EEG dataset.

    Returns
    -------
    dict
        Benchmark metrics:
            {
                "window_level": {"acc": float, "f1": float, "cm": list},
                "trial_level": {"acc": float, "f1": float, "cm": list}
            }
    """

    logger.info("LLM inference running...")

    if not processed_test:
        logger.warning("No test data provided for LLM inference")
        return {
            "window_level": {"acc": 0.0, "f1": 0.0, "cm": []},
            "trial_level": {"acc": 0.0, "f1": 0.0, "cm": []},
        }

    client = LMStudioClient()

    window_preds = []
    window_targets = []

    trial_preds = []
    trial_targets = []

    for sample in tqdm(processed_test, desc="LLM"):

        current_trial_preds = []
        trial_had_predictions = False

        for window in sample["windows"]:

            feats = extract_eeg_features(window)
            prompt = build_eeg_prompt(feats)
            pred = client.generate(prompt)

            # Handle invalid predictions by defaulting to neutral (label 1)
            if pred not in LABELS_MAPPER:
                pred_id = 1  # neutral as default
                logger.debug(f"Invalid LLM prediction '{pred}', defaulting to neutral")
            else:
                pred_id = LABELS_MAPPER[pred]

            window_preds.append(pred_id)
            window_targets.append(sample["label"])
            current_trial_preds.append(pred_id)
            trial_had_predictions = True

        # Trial-level prediction via majority voting
        if trial_had_predictions:
            trial_pred = majority_vote(current_trial_preds)
            if trial_pred is not None:
                trial_preds.append(trial_pred)
                trial_targets.append(sample["label"])

    # Build metrics with empty list handling
    window_metrics = (
        build_metrics(window_targets, window_preds)
        if window_preds
        else {"acc": 0.0, "f1": 0.0, "cm": []}
    )

    trial_metrics = (
        build_metrics(trial_targets, trial_preds)
        if trial_preds
        else {"acc": 0.0, "f1": 0.0, "cm": []}
    )

    return {
        "window_level": window_metrics,
        "trial_level": trial_metrics,
    }

# =========================================================
# MAIN BENCHMARK
# =========================================================


def run_benchmark():
    """
    Execute complete EEG inference benchmark suite.

    Benchmark categories:
        1. Classical ML models
        2. Deep learning EEG models
        3. LLM-based EEG reasoning pipeline

    Pipeline:
        1. Load held-out EEG test dataset
        2. Initialize inference device
        3. Run ML benchmark
        4. Run DL benchmark
        5. Run LLM benchmark
        6. Aggregate metrics
        7. Save timestamped benchmark report

    Output
    ------
    JSON file containing:
        - benchmark metadata
        - ML metrics
        - DL metrics
        - LLM metrics

    Notes
    -----
    This file performs STRICT evaluation only.

    No:
        - training
        - fine-tuning
        - gradient updates
        - data leakage

    occur during execution.
    """

    logger.info("========== INFERENCE BENCHMARK START ==========")

    raw = load_test_data()

    dl_data = prepare_dl_data(raw)
    ml_data = prepare_ml_data(raw)
    llm_data = prepare_llm_data(raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ml_results = run_ml_inference(ml_data)
    dl_results = run_dl_inference(dl_data, MODEL_DIR, device)
    llm_results = run_llm_inference(llm_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final = {
        "timestamp": timestamp,
        "device": str(device),
        "test_dataset": str(DATASET_DIR),
        "ml": ml_results,
        "dl": dl_results,
        "llm": llm_results,
    }

    out_path = os.path.join(
        OUTPUT_DIR,
        f"benchmark_inference_{timestamp}.json",
    )

    with open(out_path, "w") as f:
        json.dump(final, f, indent=4)

    logger.info(f"Saved benchmark → {out_path}")
    logger.info("========== DONE ✅ ==========")


if __name__ == "__main__":
    run_benchmark()
