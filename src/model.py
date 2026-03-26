"""
model.py – Train and evaluate toxicity prediction models

This module handles:
  1. Loading the feature matrix and labels
  2. Stratified train / validation / test split (handles class imbalance)
  3. Training RandomForest and XGBoost classifiers
  4. Evaluating with ROC-AUC, F1, precision-recall, confusion matrix
  5. Saving the best model

Usage:
    python src/model.py

What is stratified splitting?
    In our dataset, only ~4% of compounds are toxic and ~96% are non-toxic.
    If we split randomly, we might end up with a test set that has 0 toxic
    compounds by bad luck. Stratified splitting ensures each split
    (train/val/test) has the SAME RATIO of toxic vs non-toxic.

Why train / validation / test?
    - Train set (70%): The model learns from this data
    - Validation set (15%): We tune hyperparameters using this data
    - Test set (15%): We evaluate the FINAL model here (touched only once)
    This prevents overfitting and gives an honest performance estimate.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    confusion_matrix, precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Only RandomForest will be available.")


# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


# ══════════════════════════════════════════════════════════════
#  STEP 1: Load features and labels
# ══════════════════════════════════════════════════════════════

def load_data():
    """Load the feature matrix and labels from data/processed/.

    Returns:
        X: numpy array of features (each row = one molecule)
        y: numpy array of labels (0 = non-toxic, 1 = toxic)
        feature_names: list of feature column names
    """
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "labels.csv"))

    # The label column is whichever column isn't "smiles"
    label_col = [c for c in labels.columns if c != "smiles"][0]
    y = labels[label_col].values
    X = features.values
    feature_names = features.columns.tolist()

    print(f"📊 Loaded data: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"   Label column: '{label_col}'")
    print(f"   Class 0 (non-toxic): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"   Class 1 (toxic):     {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

    return X, y, feature_names


# ══════════════════════════════════════════════════════════════
#  STEP 2: Stratified Train / Validation / Test Split
# ══════════════════════════════════════════════════════════════
#
#  We split the data into 3 parts:
#    70% → Train (model learns from this)
#    15% → Validation (tune hyperparameters)
#    15% → Test (final evaluation)
#
#  "Stratified" means each split preserves the class ratio.
#  So if 4% of the full dataset is toxic, then ~4% of the
#  train set, ~4% of the validation set, and ~4% of the
#  test set will also be toxic.

def stratified_split(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                     random_state=42):
    """Split data into train/val/test with stratification.

    What is stratification?
        It ensures each split has the same proportion of toxic vs non-toxic.
        Without it, rare classes (like toxic compounds at 4%) might end up
        barely represented in the test set.

    How it works:
        1. First split: separate TEST set (15%) from the rest (85%)
        2. Second split: separate VALIDATION set from the remaining 85%
           → 15% of total = 15/85 ≈ 17.6% of remaining

    Args:
        X: Feature matrix (numpy array)
        y: Label array (numpy array of 0s and 1s)
        train_ratio: Fraction for training (default 0.70)
        val_ratio:   Fraction for validation (default 0.15)
        test_ratio:  Fraction for testing (default 0.15)
        random_state: Seed for reproducibility (same number = same split every time)

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Sanity check: ratios should add up to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # ── First split: separate out the TEST set ──
    # We take test_ratio (15%) off the top
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,          # 15% goes to test
        random_state=random_state,     # For reproducibility
        stratify=y                     # ← KEY: preserves class ratio
    )

    # ── Second split: separate VALIDATION from TRAINING ──
    # We need val_ratio of the ORIGINAL data from X_temp
    # val_ratio / (train_ratio + val_ratio) = what fraction of X_temp is validation
    val_fraction = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,        # ~17.6% of remaining → 15% of total
        random_state=random_state,
        stratify=y_temp                # ← Stratified again
    )

    # Print summary with class distributions
    print(f"\n📊 Stratified Split Summary:")
    print(f"   {'Set':<12} {'Samples':<10} {'Toxic':<8} {'Non-toxic':<12} {'Toxic %'}")
    print(f"   {'─'*55}")

    for name, yy in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        toxic = (yy == 1).sum()
        non_t = (yy == 0).sum()
        pct = toxic / len(yy) * 100
        print(f"   {name:<12} {len(yy):<10} {toxic:<8} {non_t:<12} {pct:.1f}%")

    print(f"   {'─'*55}")
    print(f"   {'Total':<12} {len(y):<10} {(y==1).sum():<8} {(y==0).sum():<12} {(y==1).mean()*100:.1f}%")
    print(f"\n   ✅ Class ratios are consistent across all splits (stratified)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════
#  STEP 3: Scale features & handle imbalance
# ══════════════════════════════════════════════════════════════

def preprocess(X_train, X_val, X_test, y_train, use_smote=True, random_state=42):
    """Scale features and optionally apply SMOTE to training data.

    What is scaling?
        Molecular descriptors have very different ranges (e.g. MolWt = 100-500,
        LogP = -3 to +5). Scaling puts everything on the same scale so no
        single feature dominates.

    What is SMOTE?
        SMOTE = Synthetic Minority Over-sampling Technique.
        Since toxic compounds are rare (~4%), the model might just predict
        "non-toxic" for everything and get 96% accuracy. SMOTE creates
        synthetic toxic examples so the model sees more balanced training data.

    IMPORTANT: We fit the scaler on TRAINING data only, then transform val/test.
               We apply SMOTE to TRAINING data only.
               This prevents "data leakage" — information from val/test leaking
               into training.
    """
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit + transform
    X_val = scaler.transform(X_val)           # transform only (no fit!)
    X_test = scaler.transform(X_test)         # transform only (no fit!)

    print(f"\n🔧 Preprocessing:")
    print(f"   ✅ StandardScaler fitted on training data, applied to all splits")

    # Apply SMOTE to training data only
    if use_smote:
        before = len(y_train)
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   ✅ SMOTE applied to training data:")
        print(f"      Before: {before} → After: {len(y_train)}")
        print(f"      Non-toxic: {(y_train == 0).sum()} | Toxic: {(y_train == 1).sum()}")

    return X_train, X_val, X_test, y_train, scaler


# ══════════════════════════════════════════════════════════════
#  STEP 4: Train models
# ══════════════════════════════════════════════════════════════

def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """Train a RandomForest classifier.

    RandomForest = many decision trees that vote together.
    It's a great first model: fast, handles mixed data, and gives
    feature importances for free.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,  # Number of trees (more = better but slower)
        max_depth=15,               # Max depth per tree (prevents overfitting)
        min_samples_split=5,        # Min samples needed to split a node
        class_weight="balanced",    # Give more weight to rare (toxic) class
        random_state=random_state,  # Reproducibility
        n_jobs=-1                   # Use all CPU cores
    )
    model.fit(X_train, y_train)
    print("   ✅ RandomForest trained")
    return model


def train_xgboost(X_train, y_train, random_state=42):
    """Train an XGBoost classifier.

    XGBoost = boosted trees that learn from each other's mistakes.
    Often more accurate than RandomForest, especially for imbalanced data.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    print("   ✅ XGBoost trained")
    return model


# ══════════════════════════════════════════════════════════════
#  STEP 5: Evaluate models
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X, y, set_name="Test", model_name="Model"):
    """Evaluate a model and print detailed metrics.

    Metrics explained:
        - ROC-AUC: Area under the ROC curve. 1.0 = perfect, 0.5 = random guess.
                   Best single metric for imbalanced data.
        - F1 Score: Harmonic mean of precision and recall. Balances FP and FN.
        - Precision: Of all predicted toxic, how many actually were? (avoids false alarms)
        - Recall: Of all actually toxic, how many did we catch? (avoids missing dangers)
        - Confusion Matrix: Shows TP, TN, FP, FN counts
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, y_proba)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{'═' * 55}")
    print(f"  {model_name} – {set_name} Set Results")
    print(f"{'═' * 55}")
    print(f"  ROC-AUC:  {roc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Non-toxic  Toxic")
    print(f"  Actual Non-toxic   {cm[0][0]:<8}  {cm[0][1]}")
    print(f"  Actual Toxic       {cm[1][0]:<8}  {cm[1][1]}")
    print(f"\n{classification_report(y, y_pred, target_names=['Non-toxic', 'Toxic'])}")

    return {"roc_auc": roc, "f1": f1, "confusion_matrix": cm.tolist()}


# ══════════════════════════════════════════════════════════════
#  STEP 6: Save the best model
# ══════════════════════════════════════════════════════════════

def save_model(model, scaler, feature_names, filename="best_model.pkl"):
    """Save model + scaler + feature names together as one .pkl file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names
    }
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(artifact, filepath)
    print(f"\n💾 Model saved to {filepath}")


# ══════════════════════════════════════════════════════════════
#  MAIN: Run the full training pipeline
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🧪 Model Training Pipeline")
    print("=" * 60)

    # Step 1: Load data
    X, y, feature_names = load_data()

    # Step 2: Stratified split (70% train / 15% val / 15% test)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # Step 3: Scale and apply SMOTE
    X_train, X_val, X_test, y_train, scaler = preprocess(
        X_train, X_val, X_test, y_train, use_smote=True
    )

    # Step 4: Train models
    print(f"\n🏋️ Training models...")
    rf_model = train_random_forest(X_train, y_train)
    rf_val_metrics = evaluate_model(rf_model, X_val, y_val, "Validation", "RandomForest")

    best_model = rf_model
    best_name = "RandomForest"
    best_auc = rf_val_metrics["roc_auc"]

    if HAS_XGBOOST:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_val_metrics = evaluate_model(xgb_model, X_val, y_val, "Validation", "XGBoost")

        if xgb_val_metrics["roc_auc"] > best_auc:
            best_model = xgb_model
            best_name = "XGBoost"
            best_auc = xgb_val_metrics["roc_auc"]

    # Step 5: Evaluate best model on TEST set (final, unbiased score)
    print(f"\n🏆 Best model: {best_name} (Val ROC-AUC: {best_auc:.4f})")
    print(f"   Evaluating on held-out TEST set (never seen during training/tuning)...")
    test_metrics = evaluate_model(best_model, X_test, y_test, "TEST", best_name)

    # Step 6: Save
    save_model(best_model, scaler, feature_names)

    # Save metrics for reporting
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics = {
        "best_model": best_name,
        "val_roc_auc": best_auc,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_f1": test_metrics["f1"],
    }
    with open(os.path.join(REPORTS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Training complete!")
    print(f"   Best model: {best_name}")
    print(f"   Val  ROC-AUC: {best_auc:.4f}")
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Test F1:      {test_metrics['f1']:.4f}")
    print("=" * 60)
