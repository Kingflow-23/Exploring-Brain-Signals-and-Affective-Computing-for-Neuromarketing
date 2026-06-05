"""
Project Configuration Module.

Central configuration for all EEG processing, model training, and inference tasks.

Includes:
    - Dataset paths and loading parameters
    - Preprocessing window and step configurations
    - Label mappings for emotion classification
    - ML/DL model training hyperparameters
    - Random seed for reproducibility
"""

from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================

BASE_DIR = Path(".").resolve()

DATASET_DIR = BASE_DIR / "data" / "SEED_EEG" / "SEED_EEG" / "Preprocessed_EEG"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure reproducibility across systems
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SEED DATASET CONFIG
# =============================================================================

N_TRIALS = 15

LABEL_FILE = "label.mat"
EEG_KEY_PATTERN = "_eeg"

# =============================================================================
# PREPROCESSING CONFIG
# =============================================================================

WINDOW_SIZE = 450
ML_WINDOW_SIZE = 800
LLM_WINDOW_SIZE = 1000

STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap
ML_STEP_SIZE = ML_WINDOW_SIZE // 2
LLM_STEP_SIZE = LLM_WINDOW_SIZE // 2

# =============================================================================
# SAMPLING CONFIG
# =============================================================================

FS = 200
NYQ = FS / 2
ORDER = 5

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# =============================================================================
# CANONICAL SEED CHANNEL MONTAGE
# =============================================================================

CHANNELS = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]

CHANNEL_INDEX = {ch: i for i, ch in enumerate(CHANNELS)}

# =============================================================================
# LEFT-RIGHT PAIRS (ANATOMICAL SYMMETRY)
# =============================================================================

LEFT_RIGHT_PAIRS = [
    ("FP1", "FP2"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F5", "F6"),
    ("F3", "F4"),
    ("F1", "F2"),
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("FT7", "FT8"),
    ("T7", "T8"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("TP7", "TP8"),
    ("P7", "P8"),
    ("P5", "P6"),
    ("P3", "P4"),
    ("P1", "P2"),
    ("PO7", "PO8"),
    ("PO5", "PO6"),
    ("PO3", "PO4"),
    ("CB1", "CB2"),
    ("O1", "O2"),
]

FRONTAL_PAIRS = [
    ("F3", "F4"),
    ("F7", "F8"),
    ("FP1", "FP2"),
]

PAIR_INDICES = [(CHANNEL_INDEX[l], CHANNEL_INDEX[r]) for l, r in LEFT_RIGHT_PAIRS]

FRONTAL_PAIRS_IDX = [(CHANNEL_INDEX[l], CHANNEL_INDEX[r]) for l, r in FRONTAL_PAIRS]

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

DEFAULT_FS = 200
DEFAULT_MODEL = "local-model"

ALLOWED_LABELS = {"positive", "neutral", "negative"}
LABELS_MAPPER = {"positive": 2, "neutral": 1, "negative": 0}
LABELS_MAP = {-1: 0, 0: 1, 1: 2}  # Map SEED labels to indices for classification

# =============================================================================
# EXPERIMENT SETTINGS (IMPORTANT FOR BENCHMARKING)
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# =============================================================================
# MODEL TRAINING CONFIG
# =============================================================================
BATCH_SIZE = 64
N_EPOCHS = 50
