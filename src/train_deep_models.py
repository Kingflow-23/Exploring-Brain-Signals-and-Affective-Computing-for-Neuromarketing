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
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("DEEP_TRAINER")


# =============================================================================
# UTILITIES
# =============================================================================


def ensure_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Model directory ready: {MODEL_DIR}")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

    logger.info(f"Saved JSON: {path}")


# =============================================================================
# DATA PREPARATION (NO FLATTENING)
# =============================================================================


def build_deep_dataset(processed):
    """
    Keeps EEG spatial + temporal structure.

    Output:
        X: (N, 1, 62, W)
        y: (N,)
        groups: (N,)
    """

    logger.info("Building deep learning dataset...")

    X, y, g = [], [], []

    total_windows = 0

    for sample_idx, sample in enumerate(processed):

        windows = sample["windows"]
        label = sample["label"]
        subject = sample["subject"]

        logger.info(
            f"Sample {sample_idx+1}/{len(processed)} | "
            f"subject={subject} | "
            f"label={label} | "
            f"windows={len(windows)}"
        )

        for w in windows:
            X.append(w[np.newaxis, :, :])  # (1, 62, W)
            y.append(label)
            g.append(subject)

        total_windows += len(windows)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)

    logger.info(f"Deep dataset created")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"groups shape: {g.shape}")
    logger.info(f"Total windows: {total_windows}")

    return X, y, g


# =============================================================================
# MODEL TRAINING CORE
# =============================================================================


def train_model(
    model, model_name, X_train, y_train, X_test, y_test, epochs=10, batch_size=64
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 80)
    logger.info(f"TRAINING MODEL: {model_name}")
    logger.info(f"Using device: {device}")

    model.to(device)

    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Number of batches per epoch: {len(loader)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # =========================================================================
    # TRAINING
    # =========================================================================

    logger.info("Starting training loop...")

    model.train()

    train_start = time.time()

    for ep in range(epochs):

        epoch_start = time.time()

        total_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(loader):

            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()

            out = model(xb)

            loss = loss_fn(out, yb)

            loss.backward()

            opt.step()

            total_loss += loss.item()

            # batch logging
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                logger.info(
                    f"[{model_name}] "
                    f"epoch={ep+1}/{epochs} | "
                    f"batch={batch_idx+1}/{len(loader)} | "
                    f"loss={loss.item():.6f}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)

        logger.info(
            f"[{model_name}] "
            f"epoch={ep+1}/{epochs} COMPLETE | "
            f"avg_loss={avg_loss:.6f} | "
            f"time={epoch_time:.2f}s"
        )

    train_time = time.time() - train_start

    logger.info(f"Training completed in {train_time:.2f}s")

    # =========================================================================
    # EVALUATION
    # =========================================================================

    logger.info(f"Evaluating model: {model_name}")

    model.eval()

    eval_start = time.time()

    with torch.no_grad():

        xb = torch.tensor(X_test, dtype=torch.float32).to(device)

        logits = model(xb)

        preds = logits.argmax(dim=1).cpu().numpy()

    eval_time = time.time() - eval_start

    acc = accuracy_score(y_test, preds)

    logger.info(f"{model_name} evaluation completed")
    logger.info(f"{model_name} accuracy: {acc:.4f}")
    logger.info(f"{model_name} inference time: {eval_time:.2f}s")

    return {
        "accuracy": float(acc),
        "train_time": float(train_time),
        "inference_time": float(eval_time),
    }


# =============================================================================
# MODELS (SOTA EEG)
# =============================================================================


def build_models(n_chans=62, n_classes=3, window_size=400):

    logger.info("Building deep learning models...")

    models = {
        "eegnet": EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_samples=window_size,
            final_conv_length="auto",
        ),
        "deepconvnet": Deep4Net(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_samples=window_size,
        ),
    }

    logger.info(f"Built {len(models)} deep models")

    for name in models.keys():
        logger.info(f"Registered model: {name}")

    return models


# =============================================================================
# TRAIN LOOP
# =============================================================================


def train_all(X_train, y_train, X_test, y_test):

    logger.info("=" * 80)
    logger.info("STARTING DEEP LEARNING BENCHMARK")
    logger.info("=" * 80)

    results = {}

    models = build_models(window_size=X_train.shape[-1])

    for name, model in models.items():

        logger.info(f"Launching training for: {name}")

        metrics = train_model(
            model=model,
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results[name] = metrics

        logger.info(f"{name} finished successfully")
        logger.info(f"{name} metrics: {metrics}")

    logger.info("All deep models completed")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():

    logger.info("=" * 80)
    logger.info("DEEP EEG TRAINING PIPELINE STARTED")
    logger.info("=" * 80)

    ensure_dir()

    raw = build_seed_dataset(DATASET_DIR)

    processed = preprocess_dataset(raw)

    logger.info("Building deep dataset...")
    X, y, groups = build_deep_dataset(processed)

    logger.info("Performing subject-wise split...")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    logger.info("Starting benchmark training...")

    results = train_all(X_train, y_train, X_test, y_test)

    output_path = os.path.join(MODEL_DIR, "deep_benchmark.json")

    save_json(output_path, results)

    logger.info("=" * 80)
    logger.info("DEEP TRAINING PIPELINE FINISHED ✅")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
