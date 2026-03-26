"""
improve_model.py – Train an optimized XGBoost model with Hyperparameter Tuning

This script improves upon the baseline models by:
  1. Using XGBoost, an advanced tree-based algorithm that handles tabular data exceptionally well.
  2. Introducing simple Hyperparameter Tuning using Grid Search to find the best settings.
  3. Addressing class imbalance heavily using 'scale_pos_weight'.

Key Hyperparameters Explained:
  - max_depth: How deep each tree can grow. Deeper trees learn more complex patterns but risk overfitting.
             (We try 3, 5, 7. 3 is conservative; 7 is complex).
  - learning_rate: How much each new tree corrects the mistakes of previous trees.
                 (Lower = slower learning but often better generalization. We try 0.01, 0.1).
  - scale_pos_weight: Crucial for imbalanced data! It tells the model to penalize mistakes
                      on the rare 'toxic' class more heavily than the common 'non-toxic' class.

Usage:
    python src/improve_model.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix
)

# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ══════════════════════════════════════════════════════════════
#  STEP 1: Load and Prep Data
# ══════════════════════════════════════════════════════════════

def prepare_data(test_size=0.2, random_state=42):
    """Load, split (stratified), and scale the data."""
    print("Loading data...")
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "labels.csv"))

    label_col = [c for c in labels.columns if c != "smiles"][0]
    X = features.values
    y = labels[label_col].values
    feature_names = features.columns.tolist()

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset split: {len(y_train)} train | {len(y_test)} test")
    
    # Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ══════════════════════════════════════════════════════════════
#  STEP 2: Hyperparameter Tuning (Grid Search)
# ══════════════════════════════════════════════════════════════

def tune_xgboost(X_train, y_train, random_state=42):
    """
    Find the best XGBoost parameters using GridSearchCV.
    
    GridSearchCV tries every combination of parameters in the `param_grid`,
    evaluates them using cross-validation (splitting the *training* set into folds),
    and returns the model with the best score.
    """
    print("\nStarting Hyperparameter Tuning for XGBoost...")
    print("This may take a few minutes as we test multiple parameter combinations.")
    
    # Calculate class imbalance weight
    # Example: If 95% non-toxic, 5% toxic -> ratio is 19:1. scale_pos_weight = 19.
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    sp_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"Set scale_pos_weight to {sp_weight:.2f} due to class imbalance.")

    # 1. Define the base model
    xgb_base = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=sp_weight, # Fix the imbalance weight for all tests
        n_estimators=100            # Fixed number of trees to speed up tuning
    )

    # 2. Define the exact parameters we want to test
    param_grid = {
        'max_depth': [3, 5, 7],         # Tree complexity
        'learning_rate': [0.01, 0.1],   # Step size at each iteration
        'subsample': [0.8, 1.0]         # % of data used per tree (prevents overfitting)
    }

    # 3. Setup Stratified Cross-Validation (5 folds)
    # This ensures each test fold during tuning has the same ratio of toxic/non-toxic
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # 4. Run the Grid Search
    # We score models based on ROC-AUC, as it's the most robust metric for imbalanced data.
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1 # Use all cores
    )

    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*50)
    print(" Tuning Complete!")
    print("="*50)
    print(f" Best Parameters found: {grid_search.best_params_}")
    print(f" Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_ # Return the tuned model


# ══════════════════════════════════════════════════════════════
#  STEP 3: Evaluate and Save
# ══════════════════════════════════════════════════════════════

def evaluate_and_save(model, X_test, y_test, scaler, feature_names):
    """Evaluate the tuned model on the unseen test set and save it."""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f" 📊 Final Test Set Evaluation (Tuned XGBoost)")
    print(f"{'='*50}")
    print(f" ROC-AUC: {roc_auc:.4f}")
    print(f" F1 Score: {f1:.4f}")
    print(f" Recall:   {recall:.4f}")
    print("\n Confusion Matrix:")
    print("                    Predicted")
    print("                  Non-toxic  Toxic")
    print(f" Actual Non-toxic   {cm[0][0]:<8}  {cm[0][1]}")
    print(f" Actual Toxic       {cm[1][0]:<8}  {cm[1][1]}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, "tuned_xgboost_model.pkl")
    
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": "Tuned XGBoost"
    }
    joblib.dump(artifact, filepath)
    print(f"\n💾 Saved tuned model to {filepath}")


# ══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Starting Tuned XGBoost Pipeline\n" + "-"*50)
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = prepare_data()
    
    # Tune and get the best model
    best_xgb_model = tune_xgboost(X_train_scaled, y_train)
    
    # Evaluate final model on the test set
    evaluate_and_save(best_xgb_model, X_test_scaled, y_test, scaler, feature_names)
    
    print("\nPipeline finished successfully. 🎉")
