"""
baseline_models.py – Train and evaluate baseline toxicity prediction models

This script provides a beginner-friendly, modular pipeline to train baseline models
on the Tox21 dataset. It uses Logistic Regression (the simplest linear baseline)
and Random Forest (a strong tree-based baseline).

Steps:
  1. Load pre-processed features and labels
  2. Split data into train and test sets (stratified)
  3. Preprocess data (scaling)
  4. Train models (Logistic Regression & Random Forest)
  5. Evaluate models providing ROC-AUC, F1, Recall, and Confusion Matrix
  6. Save the best-performing model to disk
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ══════════════════════════════════════════════════════════════
#  STEP 1 & 2: Load Data and Split
# ══════════════════════════════════════════════════════════════

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Load feature matrix and labels, then split into training and test sets.
    Uses 'stratify' to ensure the train and test sets have the same ratio
    of toxic to non-toxic compounds.
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

    # Get the target label column (the toxicity assay name, e.g., 'NR-AR')
    label_col = [c for c in labels.columns if c != "smiles"][0]
    
    X = features.values
    y = labels[label_col].values
    feature_names = features.columns.tolist()

    # Split: 80% train, 20% test
    # stratify=y is CRITICAL for imbalanced datasets like Tox21
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y 
    )
    
    print(f"Dataset split: {len(y_train)} train | {len(y_test)} test")
    print(f"Toxic compounds in train: {(y_train == 1).sum()} / {len(y_train)}")
    
    return X_train, X_test, y_train, y_test, feature_names


# ══════════════════════════════════════════════════════════════
#  STEP 3: Preprocess Data (Scaling)
# ══════════════════════════════════════════════════════════════

def scale_features(X_train, X_test):
    """
    Scale features so they have a mean of 0 and a standard deviation of 1.
    This is especially important for Logistic Regression.
    """
    print("Scaling features...")
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the training data, then transform both
    # This prevents information from the test set leaking into the training process
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


# ══════════════════════════════════════════════════════════════
#  STEP 4: Train Models
# ══════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train, random_state=42):
    """Train a baseline Logistic Regression model."""
    print("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced", # Helps handle the class imbalance
        max_iter=1000,           # Allows more time for the algorithm to find a solution
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """Train a baseline Random Forest model."""
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        class_weight="balanced", # Helps handle the class imbalance
        random_state=random_state,
        n_jobs=-1                # Use all available CPU cores for speed
    )
    model.fit(X_train, y_train)
    return model


# ══════════════════════════════════════════════════════════════
#  STEP 5: Evaluate Models
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model on the test set and print key metrics.
    
    Returns a dictionary of the metrics.
    """
    # Get predictions (0 or 1) and probabilities (0.0 to 1.0)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of being 'toxic' (class 1)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred) # How many of the actual toxic compounds did we find?
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f" 📊 Results: {model_name}")
    print(f"{'='*50}")
    print(f" ROC-AUC: {roc_auc:.4f} (1.0 is perfect, 0.5 is random guessing)")
    print(f" F1 Score: {f1:.4f} (Balances false positives and false negatives)")
    print(f" Recall:   {recall:.4f} (Sensitivity - fraction of toxic compounds caught)")
    
    print("\n Confusion Matrix:")
    print("                    Predicted")
    print("                  Non-toxic  Toxic")
    print(f" Actual Non-toxic   {cm[0][0]:<8}  {cm[0][1]}")
    print(f" Actual Toxic       {cm[1][0]:<8}  {cm[1][1]}")
    
    return {
        "name": model_name,
        "model": model,
        "roc_auc": roc_auc,
        "f1": f1,
        "recall": recall
    }


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
        "model_name": best_model_dict["name"]
    }
    
    joblib.dump(artifact, filepath)
    print(f"\n💾 Saved best model ({best_model_dict['name']}) to {filepath}")


# ══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Starting Baseline Toxicity Prediction Pipeline\n" + "-"*50)
    
    # 1. Load and Split
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    
    # 2. Preprocess
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("-"*50)

    # 3. Train Models
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)
    
    # 4. Evaluate Models
    lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    # 5. Compare and Save Best Model
    # We'll use ROC-AUC as our primary metric to decide the "best" model, 
    # but you could also choose F1 or Recall depending on your goals.
    best_model_metrics = lr_metrics if lr_metrics["roc_auc"] > rf_metrics["roc_auc"] else rf_metrics
    
    print(f"\n🏆 Best Model Summary & Saving")
    print("-" * 50)
    print(f" The best model is **{best_model_metrics['name']}** based on ROC-AUC ({best_model_metrics['roc_auc']:.4f}).")
    
    save_best_model(best_model_metrics, scaler, feature_names)
    print("\nPipeline finished successfully. 🎉")
