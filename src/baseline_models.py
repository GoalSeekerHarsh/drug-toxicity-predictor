"""
baseline_models.py – Train and evaluate baseline toxicity prediction models

This script provides a beginner-friendly, modular pipeline to train baseline models
on the processed Tox21 + auxiliary ChEMBL dataset. It uses Logistic Regression
(the simplest linear baseline) and Random Forest (a strong tree-based baseline).

Steps:
  1. Load pre-processed features and labels
  2. Split data into train / validation / test sets (stratified)
  3. Preprocess data with the same continuous-only scaling contract used elsewhere
  4. Train models (Logistic Regression & Random Forest)
  5. Evaluate models providing ROC-AUC, PR-AUC, Precision, Recall, F1, and Confusion Matrix
  6. Save the best-performing model to disk
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from .pipeline_utils import (
        DEFAULT_HAZARD_THRESHOLD,
        DEFAULT_SAFE_THRESHOLD,
        build_sample_weights,
        compute_metrics_dict,
        get_feature_partitions,
        resolve_label_column,
        save_metrics_report,
        save_feature_pipeline_artifact,
        stratified_train_val_test_split,
        transform_feature_frame,
    )
except ImportError:
    from pipeline_utils import (  # type: ignore
        DEFAULT_HAZARD_THRESHOLD,
        DEFAULT_SAFE_THRESHOLD,
        build_sample_weights,
        compute_metrics_dict,
        get_feature_partitions,
        resolve_label_column,
        save_metrics_report,
        save_feature_pipeline_artifact,
        stratified_train_val_test_split,
        transform_feature_frame,
    )

# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
# ══════════════════════════════════════════════════════════════
#  STEP 1 & 2: Load Data and Split
# ══════════════════════════════════════════════════════════════

def load_and_split_data(random_state=42, chembl_weight=0.5):
    """
    Load the processed dataset and create the same 70/15/15 split used
    by the tuned XGBoost training path.
    """
    print("Loading data...")
    features_path = os.path.join(PROCESSED_DATA_DIR, "features.csv")
    labels_path = os.path.join(PROCESSED_DATA_DIR, "labels.csv")
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            "Processed data not found. Please run data_loader.py and feature_engineering.py first."
        )

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    label_col = resolve_label_column(labels)
    y = labels[label_col].values
    weights = build_sample_weights(labels, chembl_weight=chembl_weight)

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, (w_train,), (w_val,), (w_test,) = stratified_train_val_test_split(
        features,
        y,
        extra_arrays=[weights],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=random_state,
    )

    print(f"Dataset split: {len(y_train)} train | {len(y_val)} val | {len(y_test)} test")
    print(f"Toxic compounds in train: {(y_train == 1).sum()} / {len(y_train)}")

    return X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, w_train, w_val, w_test, features.columns.tolist()


# ══════════════════════════════════════════════════════════════
#  STEP 3: Preprocess Data (Scaling)
# ══════════════════════════════════════════════════════════════

def load_reference_scaler():
    """
    Load the unsupervised ZINC chemical-space scaler so every model family
    uses the same continuous-feature normalization contract.
    """
    scaler_path = os.path.join(MODELS_DIR, "zinc_chemical_space_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing {scaler_path}. Run src/zinc_baseline.py first.")

    artifact = joblib.load(scaler_path)
    return artifact["scaler"]


def scale_features(X_train_df, X_val_df, X_test_df, feature_names):
    """Transform train/val/test with the shared continuous-only scaling rule."""
    print("Scaling features with the ZINC chemical-space baseline...")
    scaler = load_reference_scaler()
    artifact = {"feature_names": feature_names, "scaler": scaler}
    X_train_scaled = transform_feature_frame(X_train_df, artifact)
    X_val_scaled = transform_feature_frame(X_val_df, artifact)
    X_test_scaled = transform_feature_frame(X_test_df, artifact)
    continuous_names, fingerprint_names = get_feature_partitions(feature_names, scaler=scaler)
    final_feature_names = continuous_names + fingerprint_names
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, final_feature_names


# ══════════════════════════════════════════════════════════════
#  STEP 4: Train Models
# ══════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, sample_weight=None, random_state=42):
    """Train a baseline Logistic Regression model."""
    print("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced", # Helps handle the class imbalance
        max_iter=5000,           # Gives the optimizer enough room after descriptor compression
        solver="saga",           # More stable than lbfgs on this wide descriptor + fingerprint matrix
        tol=0.005,               # A slightly looser tolerance keeps this baseline practical to rerun
        random_state=random_state,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model

def train_random_forest(X_train, y_train, sample_weight=None, random_state=42):
    """Train a baseline Random Forest model."""
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        class_weight="balanced", # Helps handle the class imbalance
        random_state=random_state,
        n_jobs=-1                # Use all available CPU cores for speed
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ══════════════════════════════════════════════════════════════
#  STEP 5: Evaluate Models
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_eval, y_eval, model_name, decision_threshold=DEFAULT_HAZARD_THRESHOLD):
    """
    Evaluate the model on the test set and print key metrics.
    
    Returns a dictionary of the metrics.
    """
    # Get predictions (0 or 1) and probabilities (0.0 to 1.0)
    y_proba = model.predict_proba(X_eval)[:, 1] # Probability of being 'toxic' (class 1)
    metrics = compute_metrics_dict(y_eval, y_proba, decision_threshold=decision_threshold)
    cm = np.array(metrics["confusion_matrix"])

    print(f"\n{'='*50}")
    print(f" 📊 Results: {model_name}")
    print(f"{'='*50}")
    print(f" ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f" PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f" Precision: {metrics['precision']:.4f}")
    print(f" Recall:    {metrics['recall']:.4f}")
    print(f" F1 Score:  {metrics['f1']:.4f}")
    print(f" Decision Threshold: {metrics['decision_threshold']:.2f}")
    
    print("\n Confusion Matrix:")
    print("                    Predicted")
    print("                  Non-toxic  Toxic")
    print(f" Actual Non-toxic   {cm[0][0]:<8}  {cm[0][1]}")
    print(f" Actual Toxic       {cm[1][0]:<8}  {cm[1][1]}")
    
    metrics["name"] = model_name
    metrics["model"] = model
    return metrics


# ══════════════════════════════════════════════════════════════
#  STEP 6: Save the Best Model
# ══════════════════════════════════════════════════════════════

def save_best_model(best_model_dict, scaler, feature_names):
    """Save the best model, the scaler, and the feature names to a file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    filepath = os.path.join(MODELS_DIR, "baseline_best_model.pkl")
    
    artifact = {
        "model": best_model_dict["model"],
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": best_model_dict["name"],
        "safe_threshold": float(DEFAULT_SAFE_THRESHOLD),
        "hazard_threshold": float(DEFAULT_HAZARD_THRESHOLD),
    }
    
    joblib.dump(artifact, filepath)
    print(f"\n💾 Saved best model ({best_model_dict['name']}) to {filepath}")
    pipeline_path = save_feature_pipeline_artifact(
        scaler,
        feature_names,
        filename="feature_pipeline.pkl",
        extra_metadata={"selected_model": best_model_dict["name"]},
    )
    print(f"💾 Saved shared feature pipeline to {pipeline_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Starting Baseline Toxicity Prediction Pipeline\n" + "-"*50)
    
    # 1. Load and Split
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, w_train, w_val, w_test, feature_names = load_and_split_data()
    
    # 2. Preprocess
    X_train_scaled, X_val_scaled, X_test_scaled, scaler, final_feature_names = scale_features(
        X_train_df, X_val_df, X_test_df, feature_names
    )
    print("-"*50)

    # 3. Train Models
    lr_model = train_logistic_regression(X_train_scaled, y_train, sample_weight=w_train)
    rf_model = train_random_forest(X_train_scaled, y_train, sample_weight=w_train)
    
    # 4. Evaluate Models
    lr_val_metrics = evaluate_model(lr_model, X_val_scaled, y_val, "Logistic Regression (Validation)")
    rf_val_metrics = evaluate_model(rf_model, X_val_scaled, y_val, "Random Forest (Validation)")
    
    # 5. Compare and Save Best Model
    def score(metrics):
        return (metrics["precision"], metrics["pr_auc"])

    best_val_metrics = lr_val_metrics if score(lr_val_metrics) >= score(rf_val_metrics) else rf_val_metrics
    best_name = "Logistic Regression" if best_val_metrics is lr_val_metrics else "Random Forest"
    best_model = lr_model if best_name == "Logistic Regression" else rf_model

    print(f"\n🏆 Best Model Summary & Saving")
    print("-" * 50)
    print(
        f" The best baseline is **{best_name}** based on validation precision "
        f"({best_val_metrics['precision']:.4f}) and PR-AUC ({best_val_metrics['pr_auc']:.4f})."
    )
    
    best_model_dict = {"model": best_model, "name": best_name}
    save_best_model(best_model_dict, scaler, final_feature_names)

    lr_test_metrics = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression (Test)")
    rf_test_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest (Test)")
    report_payload = {
        "selection_metric": "precision_then_pr_auc",
        "selection_threshold": DEFAULT_HAZARD_THRESHOLD,
        "selected_model": best_name,
        "validation": {
            "logistic_regression": {k: v for k, v in lr_val_metrics.items() if k not in {"model", "name"}},
            "random_forest": {k: v for k, v in rf_val_metrics.items() if k not in {"model", "name"}},
        },
        "test": {
            "logistic_regression": {k: v for k, v in lr_test_metrics.items() if k not in {"model", "name"}},
            "random_forest": {k: v for k, v in rf_test_metrics.items() if k not in {"model", "name"}},
        },
    }
    report_path = save_metrics_report("baseline_metrics.json", report_payload)
    print(f"Saved baseline comparison report to {report_path}")
    print("\nPipeline finished successfully. 🎉")
