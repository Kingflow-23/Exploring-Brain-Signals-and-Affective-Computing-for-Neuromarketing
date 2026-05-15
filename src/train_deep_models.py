# =============================================================================
# DEEP LEARNING EEG TRAINING PIPELINE (SOTA MODELS)
# =============================================================================

import os
import time
import json
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from config import DATASET_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE

from braindecode.models import EEGNetv4, Deep4Net

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("DEEP_TRAINER")


# =============================================================================
# UTILITIES
# =============================================================================


def ensure_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# =============================================================================
# DATA PREPARATION (NO FLATTENING)
# =============================================================================


def build_deep_dataset(processed):
    """
    Keeps spatial + temporal structure.

    Output:
        X: (N, 1, 62, W)
        y: (N,)
        groups: subject IDs
    """

    X, y, g = [], [], []

    for sample in processed:
        for w in sample["windows"]:
            X.append(w[np.newaxis, :, :])  # (1, 62, W)
            y.append(sample["label"])
            g.append(sample["subject"])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(g, dtype=np.int64),
    )


# =============================================================================
# MODEL TRAINING CORE
# =============================================================================


def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info(f"Using device: {device}")

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # ---------------- TRAIN ----------------
    model.train()
    t0 = time.time()

    for ep in range(epochs):
        total_loss = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)

            loss.backward()
            opt.step()

            total_loss += loss.item()

        logger.info(f"epoch {ep+1}/{epochs} | loss={total_loss:.4f}")

    train_time = time.time() - t0

    # ---------------- EVAL ----------------
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds = model(xb).argmax(dim=1).cpu().numpy()

    acc = float((preds == y_test).mean())

    return acc, train_time


# =============================================================================
# MODELS (SOTA EEG)
# =============================================================================


def build_models(n_chans=62, n_classes=3, window_size=400):

    return {
        "eegnet": EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_samples=window_size,
            final_conv_length="auto",
        ),
        "deepconvnet": Deep4Net(
            n_chans=n_chans, n_outputs=n_classes, input_window_samples=window_size
        ),
    }


# =============================================================================
# TRAIN LOOP
# =============================================================================


def train_all(X_train, y_train, X_test, y_test):

    results = {}
    models = build_models(window_size=X_train.shape[-1])

    for name, model in models.items():
        logger.info(f"[DEEP] Training {name}")

        acc, train_time = train_model(model, X_train, y_train, X_test, y_test)

        results[name] = {"accuracy": acc, "train_time": train_time}

        logger.info(f"{name}: acc={acc:.4f}")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():

    ensure_dir()

    raw = build_seed_dataset(DATASET_DIR)
    processed = preprocess_dataset(raw)

    X, y, groups = build_deep_dataset(processed)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = train_all(X_train, y_train, X_test, y_test)

    save_json(os.path.join(MODEL_DIR, "deep_benchmark.json"), results)

    logger.info("DONE")


if __name__ == "__main__":
    main()
