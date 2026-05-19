# =============================================================================
# EEG DEEP LEARNING FRAMEWORK (PRODUCTION-GRADE)
# =============================================================================

import os
import time
import json
import logging
import random
import numpy as np

import torch
import torch.nn as nn

from tqdm.auto import tqdm

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from braindecode.models import EEGNet, Deep4Net, ShallowFBCSPNet, EEGConformer

from seed_loader import build_seed_dataset
from preprocessing import preprocess_dataset
from config import (
    DATASET_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    WINDOW_SIZE,
    BATCH_SIZE,
    N_EPOCHS,
)

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET
# =============================================================================


class EEGDataset(Dataset):
    """
    EEG Window Dataset for deep learning models.

    Each sample represents a fixed-length EEG segment.

    Expected tensor shape:
        X: (1, 62, T)
            - 1 = EEG input channel (required by Braindecode format)
            - 62 = EEG electrodes
            - T = time samples per window

        y: int
            - class label (0, 1, 2 for emotion classification)

    Notes:
        - Data is assumed to already be windowed during preprocessing
        - No normalization is performed here (must be handled upstream)
        - No subject leakage handling is done here (handled in split)
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, g = self.samples[idx]

        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 2:
            x = x[None, :, :]  # (1, 62, T)

        x = torch.from_numpy(x).float()

        y = torch.tensor(y, dtype=torch.long)
        g = torch.tensor(g, dtype=torch.long)

        return x, y, g


# =============================================================================
# DATA PREPARATION
# =============================================================================


def build_deep_dataset(processed):
    """
    Flattens preprocessed SEED dataset into supervised learning tensors.

    This function converts:
        - multiple subjects
        - multiple trials
        - multiple EEG windows per trial

    into a single training dataset.

    Returns:
        samples: list of tuples (window, label, subject)
    """

    logger.info("Building deep dataset...")

    samples = []

    for sample in processed:
        label = sample["label"]
        subject = sample["subject"]
        windows = sample["windows"]

        for w in windows:
            samples.append((w, label, subject))

    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Sample window shape: {samples[0][0].shape}")

    return samples


# =============================================================================
# MODEL FACTORY
# =============================================================================


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for EEG channel-wise recalibration.

    Purpose:
        Learns channel importance weights dynamically.

    Input:
        x: (B, C, T)

    Output:
        x reweighted by learned channel attention
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=-1)  # (B, C)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.sigmoid(w).unsqueeze(-1)
        return x * w


class EEG_CNN_Attention(nn.Module):
    """
    CNN-based EEG emotion classifier with channel + temporal attention.

    FAMILY:
        Custom CNN / Attention hybrid

    PURPOSE IN THIS PROJECT:
        Serves as a stronger-than-EEGNet baseline that explicitly learns:
            - spatial EEG channel interactions (via Conv2D)
            - channel importance (via SEBlock)
            - temporal importance (via learned attention pooling)

        Designed for subject-dependent EEG emotion recognition where
        discriminative patterns are localized in time and vary across channels.

    INPUT FORMAT:
        (B, 1, 62, T)
        - 62 EEG channels
        - T time samples per window

    ARCHITECTURE FLOW:
        1. Temporal Conv2D → extracts short-term EEG dynamics
        2. Spatial Conv2D → mixes EEG channels globally
        3. SEBlock → learns channel-wise importance weights
        4. Temporal attention → learns which time segments matter most
        5. Linear classifier → emotion class prediction

    WHY IT EXISTS (vs Braindecode models):
        - EEGNet / Deep4Net rely on fixed inductive bias filters
        - This model learns adaptive attention instead of fixed pooling
        - More flexible for noisy or subject-variable EEG signals

    TRADEOFFS:
        + Strong expressive power on small datasets
        - Higher overfitting risk than EEGNet/ShallowConvNet
    """

    input_mode = "eeg_4d"

    def __init__(self, n_chans=62, n_classes=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 15), padding=(0, 7)), nn.BatchNorm2d(16), nn.ELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (n_chans, 1)), nn.BatchNorm2d(32), nn.ELU()
        )

        self.se = SEBlock(32)

        self.temporal_attn = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            nn.ELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B, 1, 62, T)

        x = self.conv1(x)
        x = self.conv2(x)  # (B, 32, 1, T)
        x = x.squeeze(2)  # (B, 32, T)

        x = self.se(x)

        attn_logits = self.temporal_attn(x)  # (B, 1, T)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        x = (x * attn_weights).sum(dim=-1)  # (B, 32)

        return self.classifier(x)


class EEG_LSTM(nn.Module):
    """
    Recurrent EEG classifier using bidirectional LSTM over EEG channels.

    FAMILY:
        Sequence modeling (RNN family)

    PURPOSE IN THIS PROJECT:
        Captures temporal evolution of EEG signals by treating:
            EEG channels as feature vectors evolving over time.

        Useful for modeling:
            - temporal emotion transitions
            - long-range dependencies in EEG windows

    INPUT FORMAT:
        (B, 1, 62, T)
        internally reshaped to:
        (B, T, 62)

    ARCHITECTURE FLOW:
        1. Reshape EEG window into time sequence
        2. Bidirectional LSTM encodes temporal dynamics
        3. Last timestep embedding used as global representation
        4. Linear layer for classification

    WHY IT EXISTS:
        - CNN models focus on spatial structure
        - LSTM explicitly models temporal ordering
        - Complements CNN-based EEGNet/DeepConv architectures

    TRADEOFFS:
        + Good at temporal patterns
        - Weak spatial inductive bias compared to CNN EEG models
        - Slower training than convolutional models
    """

    def __init__(self, n_chans=62, n_classes=3, hidden=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_chans,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: (B, 1, 62, T)
        x = x.squeeze(1)  # (B, 62, T)
        x = x.permute(0, 2, 1)  # (B, T, 62)

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep

        return self.fc(out)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.residual(x)


class EEG_TCN(nn.Module):
    """
    Temporal Convolutional Network for EEG sequence modeling.

    FAMILY:
        Temporal CNN (TCN family)

    PURPOSE IN THIS PROJECT:
        Models long-range temporal dependencies in EEG signals
        using dilated causal convolutions instead of recurrence.

        Designed to:
            - replace RNNs with more stable convolutional sequence modeling
            - capture multi-scale temporal EEG dynamics

    INPUT FORMAT:
        (B, 1, 62, T)
        converted to:
        (B, 62, T)

    ARCHITECTURE FLOW:
        1. Stack of dilated residual Conv1D blocks:
            - dilation = 1 → local patterns
            - dilation = 2 → mid-range structure
            - dilation = 4 → long-range dependencies
        2. Residual connections stabilize training
        3. Global average pooling over time
        4. Linear classifier

    WHY IT EXISTS:
        - Replaces LSTM with parallelizable temporal modeling
        - More stable gradients than RNNs
        - Strong baseline for biosignals with temporal structure

    TRADEOFFS:
        + Faster than LSTM
        + Better long-range modeling than simple CNN
        - Less adaptive than attention-based models
    """

    def __init__(self, n_chans=62, n_classes=3):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(n_chans, 64, dilation=1),
            TCNBlock(64, 64, dilation=2),
            TCNBlock(64, 128, dilation=4),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (B,1,62,T)

        x = x.squeeze(1)  # (B,62,T)

        x = self.tcn(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)


class EEG_CNN_LSTM(nn.Module):
    """
    Hybrid CNN + LSTM EEG classifier.

    FAMILY:
        Hybrid spatial-temporal architecture

    PURPOSE IN THIS PROJECT:
        Combines:
            - CNN for spatial EEG feature extraction
            - LSTM for temporal sequence modeling

        Designed as a balanced model between:
            EEGNet (pure CNN) and EEG_LSTM (pure RNN)

    INPUT FORMAT:
        (B, 1, 62, T)

    ARCHITECTURE FLOW:
        1. CNN extracts spatial EEG features across channels
        2. Feature map becomes temporal sequence
        3. LSTM models temporal evolution of learned features
        4. Final hidden state used for classification

    WHY IT EXISTS:
        - CNN alone misses temporal dependencies
        - LSTM alone lacks spatial structure modeling
        - This hybrid bridges both inductive biases

    TRADEOFFS:
        + Strong general-purpose EEG baseline
        + More stable than pure attention models
        - Heavier than EEGNet / TCN
    """

    input_mode = "eeg_4d"

    def __init__(self, n_chans=62, n_classes=3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (1, 15), padding=(0, 7)),
            nn.ELU(),
            nn.Conv2d(16, 32, (n_chans, 1)),
            nn.ELU(),
        )

        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.cnn(x).squeeze(2)  # (B,32,T)
        x = x.permute(0, 2, 1)  # (B,T,32)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return self.fc(x)


def build_models(n_chans=62, n_classes=3, window_size=WINDOW_SIZE, sfreq=200):
    """
    Constructs a benchmark suite of EEG classification models.

    All models share the same input format:
        (batch_size, 1, channels, time)

    Models included:

    1. EEGNet
        - Compact CNN designed for EEG
        - Strong baseline for small datasets

    2. Deep4Net
        - Deeper CNN architecture
        - Captures hierarchical spatio-temporal EEG features

    3. ShallowFBCSPNet (ShallowConvNet)
        - Strong inductive bias for band-power features
        - Often competitive on emotion EEG datasets

    4. CNN + Attention (custom)
        - CNN feature extractor + channel SE + temporal attention
        - More flexible but higher overfitting risk

    Notes:
        - All models are trained under identical pipeline
        - No model-specific hyperparameter tuning is applied here
        - Window length is shared via WINDOW_SIZE config
    """

    logger.info("Building EEG deep models (benchmark suite)...")

    input_window_seconds = window_size / sfreq

    models = {
        # ==================================================
        # CNN FAMILY
        # ==================================================
        "cnn_attention": EEG_CNN_Attention(
            n_chans=n_chans,
            n_classes=n_classes,
        ),
        "eegnet": EEGNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            final_conv_length="auto",
        ),
        "deep4net": Deep4Net(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
            final_conv_length="auto",
        ),
        "shallowconv": ShallowFBCSPNet(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        ),
        # ==================================================
        # RNN FAMILY
        # ==================================================
        "lstm": EEG_LSTM(
            n_chans=n_chans,
            n_classes=n_classes,
        ),
        # ==================================================
        # TEMPORAL CONV FAMILY
        # ==================================================
        "tcn": EEG_TCN(
            n_chans=n_chans,
            n_classes=n_classes,
        ),
        # ==================================================
        # HYBRID FAMILY
        # ==================================================
        "cnn_lstm": EEG_CNN_LSTM(
            n_chans=n_chans,
            n_classes=n_classes,
        ),
        # ==================================================
        # TRANSFORMER FAMILY
        # ==================================================
        "eegconformer": EEGConformer(
            n_chans=n_chans,
            n_outputs=n_classes,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        ),
    }

    return models


# =============================================================================
# TRAINER
# =============================================================================


class Trainer:
    """
    Generic training engine for EEG classification models.

    Supports:
        - Any PyTorch model with (B, 1, 62, T) input
        - CrossEntropy classification
        - Learning rate scheduling
        - Early stopping via validation accuracy

    Training strategy:
        - Optimizer: Adam
        - Loss: CrossEntropyLoss
        - Metric: accuracy
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

    def forward_batch(self, xb):
        mode = getattr(self.model, "input_mode", "eeg_3d")
        if mode == "eeg_4d":
            return self.model(xb)
        return self.model(xb.squeeze(1))

    def train_epoch(self, loader):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="Training", leave=False)

        for xb, yb, _ in pbar:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            print(xb.shape)
            out = self.forward_batch(xb)

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

    def evaluate(self, loader, desc="Evaluating"):
        self.model.eval()

        preds, targets = [], []

        with torch.no_grad():
            for xb, yb, _ in tqdm(loader, desc=desc, leave=False):

                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                out = self.forward_batch(xb)

                pred = out.argmax(dim=1)

                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        cm = confusion_matrix(targets, preds, labels=[0, 1, 2])

        return {"acc": acc, "f1_macro": f1, "confusion_matrix": cm.tolist()}


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
    """
    Executes full training lifecycle for a single EEG model:

    Steps:
        1. Train for N epochs
        2. Evaluate on validation set each epoch
        3. Apply ReduceLROnPlateau scheduler
        4. Save best model checkpoint (by val accuracy)
        5. Early stopping if no improvement
        6. Final evaluation on held-out test set

    Returns:
        dict:
            best_val_acc: float
            test_acc: float
            history: dict (loss/accuracy curves)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(experiment_dir, f"{name}_best.pt")

    patience = 5
    epochs_no_improve = 0

    trainer = Trainer(model, device)

    logger.info(f"Training on device: {device}")

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    best_f1 = 0

    for epoch in range(epochs):

        t0 = time.time()

        train_loss, train_acc = trainer.train_epoch(train_loader)

        val_metrics = trainer.evaluate(val_loader, desc="Validation")
        val_acc = val_metrics["acc"]
        val_f1 = val_metrics["f1_macro"]

        trainer.scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered.")
            break

        epoch_time = time.time() - t0
        remaining = epoch_time * (epochs - epoch - 1)

        current_lr = trainer.optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"lr={current_lr:.2e} | "
            f"eta={remaining/60:.1f}m"
        )

    model.load_state_dict(torch.load(model_path))
    test_metrics = trainer.evaluate(test_loader, desc="Test")

    return {
        "best_val_f1": best_f1,
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1_macro"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": history,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Full EEG deep learning experiment pipeline.

    Pipeline stages:
        1. Load raw SEED EEG dataset
        2. Apply preprocessing (filtering + segmentation)
        3. Convert into supervised learning windows
        4. Perform subject-aware train/val/test split
        5. Train multiple deep learning models
        6. Evaluate on held-out test subjects
        7. Save results to disk

    Key design choice:
        Subject-level splitting is used to prevent data leakage.
    """

    seed_everything(RANDOM_STATE)

    os.makedirs(MODEL_DIR, exist_ok=True)

    experiment_dir = os.path.join(MODEL_DIR, "deep_experiment")
    os.makedirs(experiment_dir, exist_ok=True)

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    raw = build_seed_dataset(DATASET_DIR)
    processed = preprocess_dataset(raw)

    samples = build_deep_dataset(processed)

    labels = np.array([s[1] for s in samples])
    groups = np.array([s[2] for s in samples])

    # dummy X just for indexing consistency
    idx = np.arange(len(samples))

    # --------------------------------------------------
    # SPLIT (SUBJECT-AWARE)
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train_idx, temp_idx = next(splitter.split(idx, labels, groups))

    temp_groups = groups[temp_idx]
    temp_labels = labels[temp_idx]

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)

    val_rel_idx, test_rel_idx = next(
        splitter2.split(np.arange(len(temp_idx)), temp_labels, temp_groups)
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    # --------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------

    train_loader = DataLoader(
        EEGDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EEGDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        EEGDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------
    # MODELS
    # --------------------------------------------------

    models = build_models(window_size=train_samples[0][0].shape[-1])

    results = {}

    for name, model in tqdm(models.items(), desc="Training models"):

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING MODEL: {name}")
        logger.info(f"{'='*80}")

        metrics = run_training(
            model,
            train_loader,
            val_loader,
            test_loader,
            epochs=N_EPOCHS,
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
