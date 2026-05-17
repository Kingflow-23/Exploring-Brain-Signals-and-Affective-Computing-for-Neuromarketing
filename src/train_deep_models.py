# =============================================================================
# EEG DEEP LEARNING FRAMEWORK (PRODUCTION-GRADE)
# =============================================================================

import os
import time
import json
import logging
import numpy as np

import torch
import torch.nn as nn

from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from braindecode.models import EEGNetv4, Deep4Net

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

logger = logging.getLogger("EEG_TRAINER")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================


def seed_everything(seed=42):
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATASET
# =============================================================================


class EEGDataset(Dataset):
    """
    EEG Window Dataset (PyTorch-ready)

    Each sample:
        X: (1, 62, T)
        y: scalar label
        group: subject id (for analysis only)
    """

    def __init__(self, X, y, groups=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# DATA PREPARATION
# =============================================================================


def build_deep_dataset(processed):
    """
    Convert processed SEED dataset into DL tensors.

    Output:
        X: (N, 1, 62, T)
        y: (N,)
        groups: subject ids
    """

    logger.info("Building deep dataset...")

    X, y, g = [], [], []

    for sample in processed:
        windows = sample["windows"]
        label = sample["label"]
        subject = sample["subject"]

        for w in windows:
            X.append(w[np.newaxis, :, :])  # (1, 62, T)
            y.append(label)
            g.append(subject)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)

    # =============================================================================
    # NORMALIZATION (CRITICAL FOR EEG DL)
    # =============================================================================
    X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    return X, y, g


# =============================================================================
# MODEL FACTORY
# =============================================================================


def build_models(n_chans=62, n_classes=3, window_size=400):
    """
    Factory for EEG deep models.
    """

    logger.info("Building EEG deep models...")

    return {
        "eegnet": EEGNetv4(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_samples=window_size,
            final_conv_length="auto",
        ),
        "deep4net": Deep4Net(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_samples=window_size,
        ),
    }


# =============================================================================
# TRAINER
# =============================================================================


class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="Training", leave=False)

        for xb, yb in pbar:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            out = self.model(xb)
            loss = self.criterion(out, yb)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"}
            )

        return total_loss / len(loader), correct / total

    def evaluate(self, loader):
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)

                out = self.model(xb)
                pred = out.argmax(dim=1).cpu().numpy()

                preds.extend(pred)
                targets.extend(yb.numpy())

        acc = accuracy_score(targets, preds)
        return acc


# =============================================================================
# TRAIN PIPELINE
# =============================================================================


def run_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=10,
    name="model",
    experiment_dir="deep_experiment",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(experiment_dir, f"{name}_best.pt")

    trainer = Trainer(model, device)

    logger.info(f"Training on device: {device}")

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    best_acc = 0

    for epoch in range(epochs):

        t0 = time.time()

        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_acc = trainer.evaluate(val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        epoch_time = time.time() - t0
        remaining = epoch_time * (epochs - epoch - 1)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"eta={remaining/60:.1f}m"
        )

    test_acc = trainer.evaluate(test_loader)

    return {"best_val_acc": best_acc, "test_acc": test_acc, "history": history}


# =============================================================================
# MAIN
# =============================================================================


def main():

    seed_everything(RANDOM_STATE)

    os.makedirs(MODEL_DIR, exist_ok=True)

    experiment_dir = os.path.join(MODEL_DIR, "deep_experiment")
    os.makedirs(experiment_dir, exist_ok=True)

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    raw = build_seed_dataset(DATASET_DIR)
    processed = preprocess_dataset(raw)

    X, y, groups = build_deep_dataset(processed)

    # --------------------------------------------------
    # SPLIT (SUBJECT-AWARE)
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_idx, temp_idx = next(splitter.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    g_temp = groups[temp_idx]

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)

    val_idx, test_idx = next(splitter2.split(X_temp, y_temp, g_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------

    train_loader = DataLoader(
        EEGDataset(X_train, y_train), batch_size=64, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        EEGDataset(X_val, y_val), batch_size=64, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        EEGDataset(X_test, y_test), batch_size=64, shuffle=False, pin_memory=True
    )

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    models = build_models(window_size=X.shape[-1])

    results = {}

    for name, model in models.items():

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING MODEL: {name}")
        logger.info(f"{'='*80}")

        metrics = run_training(
            model,
            train_loader,
            val_loader,
            test_loader,
            epochs=10,
            experiment_dir=experiment_dir,
            name=name,
        )

        results[name] = metrics

        logger.info(f"{name} done → {metrics}")

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    with open(os.path.join(MODEL_DIR, "deep_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Training complete ✅")


if __name__ == "__main__":
    main()
