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
LLM_WINDOW_SIZE = 600

STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap
ML_STEP_SIZE = ML_WINDOW_SIZE // 2
LLM_STEP_SIZE = LLM_WINDOW_SIZE // 2

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
