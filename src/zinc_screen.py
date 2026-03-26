"""
zinc_screen.py – Virtual Drug Toxicity Screening on ZINC-250k Library

This script uses the trained XGBoost model to predict which molecules
in the ZINC-250k library are predicted to be toxic (NR-AR assay).

This is exactly the kind of workflow drug discovery pipelines use:
  STEP 1: Train model on known data (Tox21)
  STEP 2: Screen a large library of unknowns (ZINC-250k)
  STEP 3: Flag predicted positives for follow-up

Output:
  - reports/zinc_screen_results.csv  (predicted label + probability for all molecules)
  - reports/zinc_screen_summary.txt  (statistics: how many flagged, top hits)

Usage:
    python src/zinc_screen.py
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# ── Connect src modules ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from zinc_loader import load_zinc, validate_zinc
from feature_engineering import smiles_to_mol, compute_descriptors, compute_morgan_fingerprint

# ── Configuration ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# Screen 1,000 molecules (fast demo). Increase for deeper screening.
SCREEN_SIZE = 1000


def load_model():
    """Load the best trained model."""
    paths = [
        os.path.join(MODELS_DIR, "tuned_xgboost_model.pkl"),
        os.path.join(MODELS_DIR, "baseline_best_model.pkl"),
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"  Loading model: {os.path.basename(p)}")
            return joblib.load(p)
    raise FileNotFoundError("No trained model found. Run improve_model.py first.")


def build_feature_vector(smiles, artifact):
    """
    Convert a SMILES string to the exact feature vector the model expects.
    Returns None if the SMILES is invalid.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        desc = compute_descriptors(mol)
        fp = compute_morgan_fingerprint(mol, radius=2, n_bits=1024)
        if desc is None or fp is None:
            return None
    except Exception:
        return None

    feature_names = artifact["feature_names"]
    feature_vector = []
    for name in feature_names:
        if name.startswith("FP_"):
            try:
                bit_idx = int(name.split("_")[1])
                feature_vector.append(fp[bit_idx] if bit_idx < len(fp) else 0.0)
            except (ValueError, IndexError):
                feature_vector.append(0.0)
        else:
            feature_vector.append(desc.get(name, 0.0))

    v = np.array(feature_vector, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v.reshape(1, -1)


def screen_zinc(artifact, screen_size=SCREEN_SIZE):
    """
    Run predictions on screen_size ZINC-250k molecules.

    Returns:
        DataFrame with columns: smiles, toxicity_prob, verdict, logP, qed, SAS
    """
    print(f"\n  Loading {screen_size:,} ZINC molecules for virtual screening...")
    df = load_zinc(n_sample=screen_size)
    df = validate_zinc(df)

    model  = artifact["model"]
    scaler = artifact["scaler"]

    results = []
    skipped = 0

    print(f"  Screening {len(df):,} molecules...")
    for i, row in df.iterrows():
        smiles = str(row["smiles"])
        fv = build_feature_vector(smiles, artifact)

        if fv is None:
            skipped += 1
            continue

        try:
            fv_scaled = scaler.transform(fv)
            prob = model.predict_proba(fv_scaled)[0][1]
            verdict = "TOXIC" if prob >= 0.5 else "SAFE"
        except Exception:
            skipped += 1
            continue

        results.append({
            "smiles":       smiles,
            "toxicity_prob": round(prob, 4),
            "verdict":      verdict,
            "logP":         row.get("logP", None),
            "qed":          row.get("qed", None),
            "SAS":          row.get("SAS", None),
        })

    results_df = pd.DataFrame(results)
    print(f"\n  Screened: {len(results_df):,} | Skipped (invalid): {skipped}")
    return results_df


def print_and_save_summary(results_df):
    """Print key statistics and save results to CSV."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    total    = len(results_df)
    n_toxic  = (results_df["verdict"] == "TOXIC").sum()
    n_safe   = (results_df["verdict"] == "SAFE").sum()
    pct_tox  = n_toxic / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("  🔍 Virtual Screening Summary")
    print("=" * 60)
    print(f"  Total molecules screened: {total:,}")
    print(f"  Predicted TOXIC:          {n_toxic:,} ({pct_tox:.1f}%)")
    print(f"  Predicted SAFE:           {n_safe:,} ({100 - pct_tox:.1f}%)")

    print("\n  🏆 Top 10 Highest-Risk Molecules (sorted by toxicity probability):")
    top_hits = results_df.sort_values("toxicity_prob", ascending=False).head(10)
    print(top_hits[["smiles", "toxicity_prob", "qed", "SAS"]].to_string(index=False))

    # Save CSV
    csv_path = os.path.join(REPORTS_DIR, "zinc_screen_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾  Full results saved to: {csv_path}")

    # Save summary
    txt_path = os.path.join(REPORTS_DIR, "zinc_screen_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"ZINC-250k Virtual Screening Summary\n")
        f.write(f"Molecules screened:   {total:,}\n")
        f.write(f"Predicted TOXIC:      {n_toxic:,} ({pct_tox:.1f}%)\n")
        f.write(f"Predicted SAFE:       {n_safe:,} ({100 - pct_tox:.1f}%)\n\n")
        f.write("Top 10 Highest-Risk Molecules:\n")
        f.write(top_hits[["smiles", "toxicity_prob"]].to_string(index=False))
    print(f"📝  Summary saved to: {txt_path}")


if __name__ == "__main__":
    print("\n🔬 ZINC-250k Virtual Toxicity Screening")
    print("─" * 60)

    # 1. Load model
    artifact = load_model()

    # 2. Screen ZINC library
    results = screen_zinc(artifact, screen_size=SCREEN_SIZE)

    # 3. Report and save
    print_and_save_summary(results)

    print("\n✅ Virtual screening complete!")
