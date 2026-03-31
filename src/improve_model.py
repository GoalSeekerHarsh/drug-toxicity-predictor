"""
improve_model.py – Train an optimized XGBoost model with Hyperparameter Tuning

This script improves upon the baseline models by:
  1. Using XGBoost, an advanced tree-based algorithm that handles tabular data exceptionally well.
  2. Introducing simple Hyperparameter Tuning using Grid Search to find the best settings.
  3. Keeping Tox21 dominant while down-weighting auxiliary ChEMBL rows.
  4. Saving a consistent artifact + metrics payload for downstream inference and explainability.

Key Hyperparameters Explained:
  - max_depth: How deep each tree can grow. Deeper trees learn more complex patterns but risk overfitting.
             (We try 3, 5, 7. 3 is conservative; 7 is complex).
  - learning_rate: How much each new tree corrects the mistakes of previous trees.
                 (Lower = slower learning but often better generalization. We try 0.01, 0.1).
  - sample_weight: ChEMBL rows receive a lighter weight than Tox21 rows so the auxiliary
                   withdrawn-drug signal helps the model without dominating it.

Usage:
    python src/improve_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)

try:
    from .pipeline_utils import (
        build_sample_weights,
        compute_metrics_dict,
        get_feature_partitions,
        resolve_label_column,
        save_metrics_report,
        stratified_train_val_test_split,
        transform_feature_frame,
    )
except ImportError:
    from pipeline_utils import (  # type: ignore
        build_sample_weights,
        compute_metrics_dict,
        get_feature_partitions,
        resolve_label_column,
        save_metrics_report,
        stratified_train_val_test_split,
        transform_feature_frame,
    )

# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ══════════════════════════════════════════════════════════════
#  STEP 1: Load and Prep Data
# ══════════════════════════════════════════════════════════════

def prepare_data(random_state=42, chembl_weight=0.5, include_chembl=True):
    """Load, stratified split (train/val/test), compute weights, and scale the data.

    Args:
        random_state: Reproducibility seed.
        chembl_weight: Weight applied to rows where labels['source'] == 'chembl'.
        include_chembl: If False, drops all ChEMBL rows (source!='tox21').
    """
    print("Loading data...")
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "labels.csv"))

    label_col = resolve_label_column(labels)
    feature_names = features.columns.tolist()

    # Optionally remove auxiliary ChEMBL rows for an ablation experiment
    if (not include_chembl) and ("source" in labels.columns):
        keep_mask = labels["source"].astype(str).values == "tox21"
        features = features.loc[keep_mask].reset_index(drop=True)
        labels = labels.loc[keep_mask].reset_index(drop=True)
        print(f"Ablation: include_chembl=False → using {len(labels)} Tox21-only samples")

    y = labels[label_col].values
    w = build_sample_weights(labels, chembl_weight=chembl_weight)

    # Stratified train/val/test split
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test, (w_train,), (w_val,), (w_test,) = stratified_train_val_test_split(
        features, y, extra_arrays=[w], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=random_state
    )

    print(f"Dataset split: {len(y_train)} train | {len(y_val)} val | {len(y_test)} test")
    
    # Scale Features via Unsupervised ZINC Chemical Space Baseline
    print("Normalizing features via ZINC-250k Global Chemical Space baseline...")
    zinc_scaler_path = os.path.join(MODELS_DIR, "zinc_chemical_space_scaler.pkl")
    if not os.path.exists(zinc_scaler_path):
        raise FileNotFoundError(f"Missing {zinc_scaler_path} - please run src/zinc_baseline.py first.")
        
    zinc_artifact = joblib.load(zinc_scaler_path)
    scaler = zinc_artifact["scaler"]
    transform_artifact = {"feature_names": feature_names, "scaler": scaler}
    X_train_scaled = transform_feature_frame(X_train_df, transform_artifact)
    X_val_scaled = transform_feature_frame(X_val_df, transform_artifact)
    X_test_scaled = transform_feature_frame(X_test_df, transform_artifact)
    continuous_cols, fp_cols = get_feature_partitions(feature_names, scaler=scaler)
    final_feature_names = continuous_cols + fp_cols
    
    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        w_train,
        w_val,
        w_test,
        scaler,
        final_feature_names,
    )


# ══════════════════════════════════════════════════════════════
#  STEP 2: Hyperparameter Tuning (Grid Search)
# ══════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, sample_weight=None, random_state=42):
    """
    Find the best XGBoost parameters using GridSearchCV.
    
    GridSearchCV tries every combination of parameters in the `param_grid`,
    evaluates them using cross-validation (splitting the *training* set into folds),
    and returns the model with the best score.
    """
    print("\nStarting Hyperparameter Tuning for XGBoost...")
    print("This may take a few minutes as we test multiple parameter combinations.")
    
    # For Precision-First: we intentionally keep scale_pos_weight neutral.
    # The objective is to reduce false positives while relying on source-aware
    # sample weights for the noisier ChEMBL supplement.
    sp_weight = 1.0
    
    print(f"Set scale_pos_weight to {sp_weight:.2f} (Strict Precision over Recall).")

    # 1. Define the base model
    xgb_base = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=sp_weight, # Fix the imbalance weight for all tests
        n_estimators=100,           # Fixed number of trees to speed up tuning
        n_jobs=1                    # Safe in restricted environments
    )

    # 2. Define the exact parameters we want to test
    # (Reduced tree depth prevents memorizing complex structures -> cuts false positives)
    param_grid = {
        'max_depth': [2, 3, 5],         # Tree complexity
        'learning_rate': [0.01, 0.1],   # Step size at each iteration
        'subsample': [0.8, 1.0]         # % of data used per tree (prevents overfitting)
    }

    # 3. Setup Stratified Cross-Validation (5 folds)
    # This ensures each test fold during tuning has the same ratio of toxic/non-toxic
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # 4. Run the Grid Search
    # We score models based on PRECISION (Correctness > Complete catching)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='precision',
        cv=cv,
        verbose=1,
        n_jobs=1 # Avoid loky semaphore issues in restricted environments
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    grid_search.fit(X_train, y_train, **fit_kwargs)
    
    print("\n" + "="*50)
    print(" Tuning Complete!")
    print("="*50)
    print(f" Best Parameters found: {grid_search.best_params_}")
    print(f" Best Cross-Validation Precision: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_ # Return the tuned model


# ══════════════════════════════════════════════════════════════
#  STEP 3: Evaluate and Save
# ══════════════════════════════════════════════════════════════

def evaluate_and_save(
    model,
    X_test,
    y_test,
    scaler,
    feature_names,
    artifact_name="tuned_xgboost_model.pkl",
    report_name="tuned_xgboost_metrics.json",
    extra_metadata=None,
    update_best_model=True,
):
    """Evaluate model on test set, save model artifact + metrics report."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics_dict(y_test, y_pred, y_proba)
    roc_auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]
    f1 = metrics["f1"]
    recall = metrics["recall"]
    precision = metrics["precision"]
    cm = np.array(metrics["confusion_matrix"])

    print(f"\n{'='*50}")
    print(f" 📊 Final Test Set Evaluation (Precision Tuned XGBoost)")
    print(f"{'='*50}")
    print(f" ROC-AUC:   {roc_auc:.4f}")
    print(f" PR-AUC:    {pr_auc:.4f}")
    print(f" Precision: {precision:.4f}  <-- THE MOST IMPORTANT METRIC NOW")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    print("\n Confusion Matrix:")
    print("                    Predicted")
    print("                  Non-toxic  Toxic")
    print(f" Actual Non-toxic   {cm[0][0]:<8}  {cm[0][1]}")
    print(f" Actual Toxic       {cm[1][0]:<8}  {cm[1][1]}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, artifact_name)
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": "Tuned XGBoost"
    }
    joblib.dump(artifact, filepath)
    print(f"\n💾 Saved tuned model to {filepath}")

    # Also write a canonical "best_model.pkl" pointer artifact for downstream scripts
    if update_best_model:
        best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(artifact, best_path)
        print(f"💾 Updated best model at {best_path}")

    report_path = save_metrics_report(report_name, metrics, extra_metadata=extra_metadata)
    print(f"🧾 Saved metrics report to {report_path}")

    return metrics

# ══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Starting Tuned XGBoost Pipeline\n" + "-"*50)
    
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, w_train, w_val, w_test, scaler, feature_names = prepare_data()
    
    # Tune and get the best model
    best_xgb_model = tune_xgboost(X_train_scaled, y_train, sample_weight=w_train)
    
    # Quick sanity check on validation before final test
    try:
        y_val_pred = best_xgb_model.predict(X_val_scaled)
        y_val_proba = best_xgb_model.predict_proba(X_val_scaled)[:, 1]
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_roc = roc_auc_score(y_val, y_val_proba)
        print(f"\nValidation: ROC-AUC={val_roc:.4f} | Precision={val_precision:.4f} | Recall={val_recall:.4f}")
    except Exception:
        pass

    # Evaluate final model on the test set
    evaluate_and_save(
        best_xgb_model,
        X_test_scaled,
        y_test,
        scaler,
        feature_names,
        artifact_name="tuned_xgboost_model.pkl",
        report_name="tuned_xgboost_metrics.json",
        extra_metadata={"include_chembl": True},
    )
    
    print("\nPipeline finished successfully. 🎉")
