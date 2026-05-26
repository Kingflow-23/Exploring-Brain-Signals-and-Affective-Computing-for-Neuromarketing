import os
import json
import torch
import numpy as np
import joblib
import logging

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from feature_extraction import extract_dataset_features

from train_deep_models import EEGDataset
from torch.utils.data import DataLoader

from model_registry import get_model

from config import DATASET_DIR, MODEL_DIR, OUTPUT_DIR

from llm_inference import LMStudioClient, extract_eeg_features, build_eeg_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BENCHMARK_INFER")


# =========================================================
# LOAD TEST SET ONLY
# =========================================================


def load_test_data():
    test_path = os.path.join(DATASET_DIR, "test")
    logger.info(f"Loading test set from {test_path}")

    raw = build_seed_dataset(test_path)
    processed = preprocess_dataset(raw)

    return processed


# =========================================================
# ML INFERENCE
# =========================================================


def run_ml_inference(processed_test):

    logger.info("ML inference running...")

    X, y, _ = extract_dataset_features(processed_test)

    results = {}

    model_names = os.listdir(MODEL_DIR)

    for name in tqdm(model_names, desc="ML models"):

        model_path = os.path.join(MODEL_DIR, name, "model.pkl")

        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)

        preds = model.predict(X)

        results[name] = {
            "acc": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, average="macro")),
            "cm": confusion_matrix(y, preds).tolist(),
        }

    return results


# =========================================================
# DL INFERENCE
# =========================================================


def run_dl_inference(processed_test, model_dir, device):

    samples = []
    for s in processed_test:
        for w in s["windows"]:
            samples.append((w, s["label"], s["subject"]))

    loader = DataLoader(EEGDataset(samples), batch_size=64)

    results = {}

    deep_dir = os.path.join(model_dir, "deep_experiment")

    for file in os.listdir(deep_dir):

        if not file.endswith("_best.pt"):
            continue

        model_name = file.replace("_best.pt", "")
        model_path = os.path.join(deep_dir, file)

        model = get_model(model_name).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(device)

                out = model(xb)
                pred = out.argmax(dim=1)

                preds.extend(pred.cpu().numpy())
                targets.extend(yb.numpy())

        results[model_name] = {
            "acc": float(accuracy_score(targets, preds)),
            "f1": float(f1_score(targets, preds, average="macro")),
            "cm": confusion_matrix(targets, preds).tolist(),
        }

    return results


# =========================================================
# LLM INFERENCE
# =========================================================


def run_llm_inference(processed_test):

    logger.info("LLM inference running...")

    client = LMStudioClient()

    results = []
    y_true = []

    for sample in tqdm(processed_test, desc="LLM"):

        for window in sample["windows"]:

            feats = extract_eeg_features(window)
            prompt = build_eeg_prompt(feats)

            pred = client.generate(prompt)

            results.append(pred)
            y_true.append(sample["label"])

    return {
        "acc": float(accuracy_score(y_true, results)),
        "f1": float(f1_score(y_true, results, average="macro")),
        "cm": confusion_matrix(y_true, results).tolist(),
    }


# =========================================================
# MAIN BENCHMARK
# =========================================================


def run_benchmark():

    logger.info("========== INFERENCE BENCHMARK START ==========")

    test_data = load_test_data()

    ml_results = run_ml_inference(test_data)
    dl_results = run_dl_inference(test_data)
    llm_results = run_llm_inference(test_data)

    final = {"ml": ml_results, "dl": dl_results, "llm": llm_results}

    out_path = os.path.join(OUTPUT_DIR, "benchmark_inference.json")

    with open(out_path, "w") as f:
        json.dump(final, f, indent=4)

    logger.info(f"Saved benchmark → {out_path}")
    logger.info("========== DONE ✅ ==========")


if __name__ == "__main__":
    run_benchmark()
