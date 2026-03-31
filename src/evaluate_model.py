"""
evaluate_model.py – Comprehensive Evaluation for Binary Toxicity Models

This script explains and calculates every key metric needed for an imbalanced
toxicity dataset (ROC-AUC, PR-AUC, F1, Precision, Recall, Confusion Matrix)
and automatically generates visual plots.

How to interpret each metric (Simple Terms):
-------------------------------------------
1. Confusion Matrix: A 2x2 table of exactly where the model got it right/wrong.
   - True Negatives (TN): Safe drugs correctly labelled safe.
   - False Positives (FP): Safe drugs incorrectly labelled TOXIC (false alarm).
   - False Negatives (FN): Toxic drugs incorrectly labelled SAFE (dangerous!).
   - True Positives (TP): Toxic drugs correctly labelled TOXIC.

2. Recall (Sensitivity): "Out of all actual toxic drugs, how many did we catch?"
   - E.g. 0.80 means we caught 80% of toxic drugs, but 20% slipped through.
   - VERY IMPORTANT in toxicity prediction (you don't want to miss dangerous drugs).

3. Precision: "When the model says a drug is toxic, how often is it right?"
   - E.g. 0.50 means half the time it cries "toxic", it's a false alarm.

4. F1 Score: The harmonic mean of Precision and Recall.
   - Balances the trade-off. If you just call everything "toxic" to get 100% recall,
     your precision drops to near 0, ruining your F1. 

5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve):
   - "If I pick 1 random toxic drug and 1 random safe drug, what's the probability
     the model scores the toxic one higher?"
   - 1.0 = perfect. 0.5 = random guessing.
   - Good for general ranking ability.

6. PR-AUC (Precision-Recall Area Under Curve):
   - Like ROC-AUC, but MUCH BETTER for imbalanced datasets (e.g. 4% toxic data).
   - It summarizes the trade-off between Precision and Recall.

Usage:
    python src/evaluate_model.py
"""

import os
import tempfile
import joblib
import pandas as pd
MPL_DIR = os.path.join(tempfile.gettempdir(), "toxpredict-mpl")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "toxpredict-cache")
os.makedirs(MPL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

try:
    from .pipeline_utils import (
        DEFAULT_HAZARD_THRESHOLD,
        classify_probabilities,
        resolve_label_column,
        stratified_train_val_test_split,
        transform_feature_frame,
    )
except ImportError:
    from pipeline_utils import (  # type: ignore
        DEFAULT_HAZARD_THRESHOLD,
        classify_probabilities,
        resolve_label_column,
        stratified_train_val_test_split,
        transform_feature_frame,
    )

def load_test_data():
    """Load data and recreate the exact test set for evaluation."""
    print("Loading data...")
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "labels.csv"))

    label_col = resolve_label_column(labels)
    y = labels[label_col].values

    _, _, X_test, _, _, y_test, _, _, _ = stratified_train_val_test_split(
        features, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    return X_test, y_test

def load_model_pipeline(model_filename="baseline_best_model.pkl"):
    """Load the saved model and scaler."""
    filepath = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}. Run a training script first.")
    
    artifact = joblib.load(filepath)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"{filepath} is not a model artifact.")
    model_name = artifact.get("model_name", model_filename.split('.')[0].replace('_', ' ').title())
    hazard_threshold = float(artifact.get("hazard_threshold", DEFAULT_HAZARD_THRESHOLD))
    return artifact["model"], artifact["scaler"], model_name, artifact["feature_names"], hazard_threshold

def evaluate_and_plot(model, scaler, X_test, y_test, model_name, feature_names, artifact_id=None, decision_threshold=DEFAULT_HAZARD_THRESHOLD):
    """Generate all critical metrics and visual plots."""
    print(f"\nEvaluating: {model_name}")
    print("="*60)
    
    # 1. Prepare Data & Predict
    artifact = {"model": model, "scaler": scaler, "feature_names": feature_names}
    X_test_scaled = transform_feature_frame(X_test, artifact)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of class 1
    y_pred = classify_probabilities(y_proba, decision_threshold=decision_threshold)

    # 2. Calculate Strict Metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precision: {precision:.4f} (When it predicts Toxic, it is correct {precision*100:.1f}% of the time)")
    print(f"Recall:    {recall:.4f} (It caught {recall*100:.1f}% of all actual Toxic drugs)")
    print(f"F1 Score:  {f1:.4f} (Harmonic mean of both)")
    print(f"Decision Threshold: {decision_threshold:.2f} (matches CRITICAL HAZARD cutoff)")
    
    # 3. AUC Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    # PR-AUC calculation
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)
    
    print(f"\nROC-AUC:   {roc_auc:.4f} (General ranking ability)")
    print(f"PR-AUC:    {pr_auc:.4f} (Performance specific to evaluating the rare Toxic class)")

    # 4. Create Subplots layout (1 row, 3 columns)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Evaluation Suite: {model_name}', fontsize=16, fontweight='bold')

    # Plot A: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                cbar=False, annot_kws={"size": 14})
    ax1.set_title('Confusion Matrix', fontsize=14)
    ax1.set_xlabel('Predicted Label (0=Safe, 1=Toxic)', fontsize=12)
    ax1.set_ylabel('True Label (0=Safe, 1=Toxic)', fontsize=12)

    # Plot B: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (Fall-out)')
    ax2.set_ylabel('True Positive Rate (Recall)')
    ax2.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
    ax2.legend(loc="lower right")

    # Plot C: Precision-Recall Curve (Crucial for imbalanced data)
    # The baseline for PR curve is the ratio of positive class
    baseline = (y_test == 1).sum() / len(y_test)
    ax3.plot(recalls, precisions, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax3.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', label=f'Random ({baseline:.3f})')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('Precision (Positive Predictive Value)')
    ax3.set_title('Precision-Recall Curve (PR)', fontsize=14)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    
    # Save the plot
    os.makedirs(REPORTS_DIR, exist_ok=True)
    save_label = artifact_id or model_name.replace(" ", "_").lower()
    plot_name = f"{save_label.replace('.pkl', '').lower()}_evaluation.png"
    save_path = os.path.join(REPORTS_DIR, plot_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n💾 Saved visualization suite to: {save_path}")

if __name__ == "__main__":
    import glob
    
    X_test, y_test = load_test_data()
    
    # Try to find any trained models in the models directory
    preferred_order = [
        "baseline_best_model.pkl",
        "best_model.pkl",
        "tuned_xgboost_model_with_chembl.pkl",
        "tuned_xgboost_model_without_chembl.pkl",
        "tuned_xgboost_model.pkl",
    ]
    available = {os.path.basename(path): path for path in glob.glob(os.path.join(MODELS_DIR, "*.pkl"))}
    ordered_names = [name for name in preferred_order if name in available]
    ordered_names.extend(sorted(name for name in available if name not in ordered_names))

    if not ordered_names:
        print("No .pkl files found in models/. Please run baseline_models.py first.")
    else:
        seen_model_files = set()
        for filename in ordered_names:
            try:
                model, scaler, model_name, feature_names, hazard_threshold = load_model_pipeline(filename)
            except ValueError:
                print(f"Skipping non-model artifact: {filename}")
                continue
            if filename in {"tuned_xgboost_model.pkl"} and "best_model.pkl" in seen_model_files:
                print(f"Skipping duplicate compatibility alias: {filename}")
                continue
            seen_model_files.add(filename)
            evaluate_and_plot(
                model,
                scaler,
                X_test,
                y_test,
                model_name,
                feature_names,
                artifact_id=filename,
                decision_threshold=hazard_threshold,
            )
    
    print("\n✅ Evaluation complete.")
