"""
shap_explain.py – Model Explainability using SHAP

This script opens up the "black box" of the tuned XGBoost model.
It uses SHAP (SHapley Additive exPlanations), an advanced game-theory technique,
to explain EXACTLY why the model makes its decisions.

We will generate 3 things:
  1. Global Summary (Beeswarm Plot): Overview of the top features over the whole dataset.
  2. Top 10 Feature List: A simple list of the most important descriptors.
  3. Local Explanation (Waterfall Plot): Why the model predicted a SINGLE specific molecule
     to be toxic or safe.
     
Note on parameters:
  - We use `shap.TreeExplainer` because it is highly optimized for forest/xgboost models.
  - We plot without GUI `show()` and save directly to `.png` to allow background execution.
"""

import os
import tempfile
import pandas as pd
import numpy as np

MPL_DIR = os.path.join(tempfile.gettempdir(), "toxpredict-mpl")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "toxpredict-cache")
os.makedirs(MPL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Set backend to avoid GUI issues when running scripts

# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

try:
    from .pipeline_utils import load_model_artifact, transform_feature_frame
except ImportError:
    from pipeline_utils import load_model_artifact, transform_feature_frame  # type: ignore


def load_model_and_data():
    """Load the best tuned model and the feature data."""
    artifact = load_model_artifact(prefer_best=True)
    if artifact is None:
        raise FileNotFoundError("No trained model artifact found. Run improve_model.py first.")

    print(f"Loading model from: {artifact['artifact_path']}")
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    
    # Load dataset
    print("Loading feature data for explanation...")
    features = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"))
    
    # We only need a random sample of the data to get the global picture (saves time)
    # 500 molecules is plenty for a global summary plot
    n_sample = min(500, len(features))
    sample_df = features.sample(n=n_sample, random_state=42)
    X_sample_scaled = transform_feature_frame(sample_df, artifact)
    
    return model, X_sample_scaled, feature_names


def explain_model():
    """Run SHAP explanations: global, top 10, and a single local inference."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 1. Load data
    model, X_sample, feature_names = load_model_and_data()
    
    # 2. Create the SHAP Explainer
    # TreeExplainer is lightning fast for XGBoost/RandomForest
    print("\nCalculations starting... Computing SHAP values.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Some models return SHAP values for each class (safe vs toxic). 
    # If it's a list, we take the target class [1] (which is "toxic").
    # For newer shap versions, shap_values is an Explanation object
    if len(shap_values.shape) == 3: # (samples, features, classes)
        shap_values = shap_values[:, :, 1]
    
    # Re-assign feature names so plots look nice
    shap_values.feature_names = feature_names

    print("\n" + "="*50)
    print(" 1. GLOBAL FEATURE IMPORTANCE & TOP 10")
    print("="*50)
    
    # Calculate the average absolute impact each feature has
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create a DataFrame to sort and view the Top 10 easily
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP| Impact": mean_abs_shap
    }).sort_values(by="Mean |SHAP| Impact", ascending=False)
    
    print("\n🌟 Top 10 Most Important Descriptors for Toxicity:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save the Global Summary (Beeswarm Plot)
    # A beeswarm shows the top features, their distribution, and whether a HIGH value
    # for that feature pushes the prediction towards Toxic (red) or Safe (blue).
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Global Summary (What drives Toxicity overall?)", fontsize=14)
    plt.tight_layout()
    summary_path = os.path.join(REPORTS_DIR, "shap_global_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Global Summary Plot to: {summary_path}")

    print("\n" + "="*50)
    print(" 2. LOCAL EXPLANATION (Single Molecule Analysis)")
    print("="*50)
    
    # Pick a single molecule (e.g., the very first one in our 500-sample set)
    molecule_idx = 0 
    
    # Generate a Waterfall Plot for this single molecule.
    # A waterfall plot starts at the "base expected value" (the average output of the model)
    # and shows how each individual feature pushes the prediction up or down until it reaches
    # the final prediction for THIS SPECIFIC molecule.
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[molecule_idx], show=False)
    plt.title(f"SHAP Waterfall Plot for Molecule #{molecule_idx}", fontsize=14)
    plt.tight_layout()
    
    waterfall_path = os.path.join(REPORTS_DIR, "shap_local_waterfall.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Local Explanation (Waterfall) Plot for Molecule #{molecule_idx} to: {waterfall_path}")

    print("\n🎉 Explainability complete. Check the 'reports/' folder for the images!")


if __name__ == "__main__":
    explain_model()
