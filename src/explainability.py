"""
explainability.py – SHAP-based feature importance analysis

Functions:
    - compute_shap_values(): Generate SHAP values for the trained model
    - plot_feature_importance(): Bar plot of top N important features
    - plot_shap_summary(): SHAP beeswarm summary plot
    - get_top_features(): Return a DataFrame of top contributing molecular descriptors
"""

import os
import tempfile
import numpy as np
import pandas as pd

MPL_DIR = os.path.join(tempfile.gettempdir(), "toxpredict-mpl")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "toxpredict-cache")
os.makedirs(MPL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

try:
    from .pipeline_utils import load_model_artifact as load_runtime_artifact, transform_feature_frame
except ImportError:
    from pipeline_utils import load_model_artifact as load_runtime_artifact, transform_feature_frame  # type: ignore


# ── Configuration ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_model_artifact(filename="best_model.pkl"):
    """Load saved model, scaler, and feature names."""
    artifact = load_model_artifact_from_disk(filename)
    return artifact["model"], artifact["scaler"], artifact["feature_names"], artifact


def load_model_artifact_from_disk(filename="best_model.pkl"):
    """Load a specific artifact by filename, falling back to the preferred runtime loader."""
    filepath = os.path.join(MODELS_DIR, filename)
    if os.path.exists(filepath):
        artifact = joblib.load(filepath)
        artifact = dict(artifact)
        artifact["artifact_path"] = filepath
        return artifact

    artifact = load_runtime_artifact(prefer_best=True)
    if artifact is None:
        raise FileNotFoundError(f"Model artifact not found at {filepath}")
    return artifact


def compute_shap_values(model, X_sample, feature_names):
    """Compute SHAP values for the given data sample.
    
    Uses TreeExplainer for tree-based models (RF, XGBoost).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use SHAP values for the toxic (positive) class
    
    return explainer, shap_values


def plot_feature_importance(shap_values, feature_names, top_n=15, save_path=None):
    """Bar plot showing mean absolute SHAP values for top features."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values("Mean |SHAP|", ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Mean |SHAP|", y="Feature", hue="Feature", palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Molecular Descriptors by SHAP Importance", fontsize=14)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance plot to {save_path}")
    
    # plt.show()
    return importance_df


def plot_shap_summary(shap_values, X_sample, feature_names, save_path=None):
    """SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title("SHAP Summary – Feature Impact on Toxicity Prediction", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP summary plot to {save_path}")
    
    # plt.show()


def get_top_features(shap_values, feature_names, top_n=10):
    """Return a DataFrame of top N features by SHAP importance."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_df = pd.DataFrame({
        "Rank": range(1, top_n + 1),
        "Molecular Descriptor": [feature_names[i] for i in np.argsort(mean_abs_shap)[::-1][:top_n]],
        "Mean |SHAP value|": sorted(mean_abs_shap, reverse=True)[:top_n]
    })
    return top_df


if __name__ == "__main__":
    # Load model and data
    model, scaler, feature_names, artifact = load_model_artifact()
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    
    # Scale and sample for SHAP (use 500 samples for speed)
    X = transform_feature_frame(features, artifact)
    sample_size = min(500, len(X))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]
    
    # Compute SHAP
    explainer, shap_values = compute_shap_values(model, X_sample, feature_names)
    
    # Plots
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    importance_df = plot_feature_importance(
        shap_values, feature_names, top_n=15,
        save_path=os.path.join(REPORTS_DIR, "feature_importance.png")
    )
    
    plot_shap_summary(
        shap_values, X_sample, feature_names,
        save_path=os.path.join(REPORTS_DIR, "shap_summary.png")
    )
    
    top_features = get_top_features(shap_values, feature_names)
    print("\n🔬 Top 10 Molecular Descriptors Linked to Toxicity:")
    print(top_features.to_string(index=False))
    
    # Save to CSV
    top_features.to_csv(os.path.join(REPORTS_DIR, "top_features.csv"), index=False)
