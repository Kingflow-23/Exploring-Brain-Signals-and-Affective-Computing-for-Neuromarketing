# =============================================================================
# EEG MODEL ANALYSIS PIPELINE
# =============================================================================

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit

from config import MODEL_DIR, RANDOM_STATE


# =============================================================================
# MODEL DISCOVERY
# =============================================================================

def find_all_models(model_dir=MODEL_DIR):
    """
    Recursively scans MODEL_DIR and returns all saved models.

    Returns:
        List of:
        [
            {
                "name": str,
                "path": str
            }
        ]
    """
    models = []

    for root, _, files in os.walk(model_dir):
        if "model.pkl" in files:
            models.append({
                "name": os.path.basename(root),
                "path": os.path.join(root, "model.pkl")
            })

    return models


def load_model(path):
    """Load a serialized sklearn model."""
    return joblib.load(path)


# =============================================================================
# SAFE MODEL REPLICATION
# =============================================================================

def safe_clone_model(model):
    """
    Creates a clean independent copy of a loaded model.

    WHY:
    - Prevents contamination between learning curve iterations
    - Avoids sklearn clone limitations on fitted objects
    """
    return joblib.loads(joblib.dumps(model))


# =============================================================================
# LEARNING CURVE COMPUTATION
# =============================================================================

def compute_learning_curve(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    steps=5,
    stratified=True
):
    """
    Computes learning curve for a SINGLE model.

    -------------------------------------------------------
    WHAT IT MEASURES
    -------------------------------------------------------
    Effect of training set size on generalization accuracy.

    -------------------------------------------------------
    OUTPUT FORMAT
    -------------------------------------------------------
    [
        {
            "fraction": float,
            "size": int,
            "accuracy": float
        }
    ]
    """

    n = len(X_train)
    fractions = np.linspace(0.1, 1.0, steps)

    results = []

    for i, frac in enumerate(fractions):
        size = max(10, int(n * frac))

        # -------------------------------
        # SAMPLE SELECTION
        # -------------------------------
        if stratified:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=size,
                random_state=RANDOM_STATE + i
            )
            idx, _ = next(splitter.split(X_train, y_train))
        else:
            rng = np.random.default_rng(RANDOM_STATE + i)
            idx = rng.choice(n, size=size, replace=False)

        X_sub = X_train[idx]
        y_sub = y_train[idx]

        # -------------------------------
        # SAFE MODEL COPY
        # -------------------------------
        m = safe_clone_model(model)

        # -------------------------------
        # TRAIN + EVALUATE
        # -------------------------------
        m.fit(X_sub, y_sub)
        acc = m.score(X_test, y_test)

        results.append({
            "fraction": float(frac),
            "size": int(size),
            "accuracy": float(acc)
        })

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_curve(curve, title, save_path=None):
    """
    Plot learning curve.

    If save_path is provided → saves file instead of blocking UI.
    """

    x = [d["fraction"] for d in curve]
    y = [d["accuracy"] for d in curve]

    plt.figure()
    plt.plot(x, y, marker="o")

    plt.title(title)
    plt.xlabel("Training Fraction")
    plt.ylabel("Accuracy")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# =============================================================================
# FULL PIPELINE RUNNER
# =============================================================================

def run_all_learning_curves(
    X_train,
    y_train,
    X_test,
    y_test,
    save_plots=True
):
    """
    Runs learning curves for ALL models in MODEL_DIR recursively.

    -------------------------------------------------------
    OUTPUT
    -------------------------------------------------------
    {
        model_name: learning_curve_data
    }
    """

    results = {}

    models = find_all_models()

    for model_info in models:
        name = model_info["name"]
        path = model_info["path"]

        print(f"[INFO] Processing model: {name}")

        model = load_model(path)

        curve = compute_learning_curve(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        results[name] = curve

        # save plot per model
        if save_plots:
            plot_curve(
                curve,
                title=f"Learning Curve - {name}",
                save_path=os.path.join(MODEL_DIR, name, "learning_curve.png")
            )
        else:
            plot_curve(curve, name)

    return results


# =============================================================================
# OPTIONAL SAVE
# =============================================================================

def save_analysis(results, path):
    """Save full analysis results as JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=4)