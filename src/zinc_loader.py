"""
zinc_loader.py – Load, Validate, and Describe the ZINC-250k Dataset

ZINC-250k is a curated set of 250,000 drug-like molecules from the ZINC database.
Unlike Tox21, it does NOT have toxicity labels, but it has 3 pre-computed properties:

  - logP : Octanol-water partition coefficient (lipophilicity).
            High logP → fat-soluble (crosses membranes easily).
            Low logP  → water-soluble.
  - qed  : Quantitative Estimate of Drug-likeness (0 to 1).
            Higher = more drug-like (obeys Lipinski's Rule of Five).
  - SAS  : Synthetic Accessibility Score (1 to 10).
            1 = trivially easy to synthesize. 10 = extremely hard.

Use Cases:
  1. Virtual Screening library – run our trained toxicity model on 1000 ZINC molecules.
  2. Benchmark – compare model outputs against known drug-like molecules.
  3. Demo – show the Batch Upload feature in the Streamlit app at scale.

Usage:
    python src/zinc_loader.py
"""

import os
import pandas as pd
import numpy as np

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ── Configuration ──────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

ZINC_FILE = "250k_rndm_zinc_drugs_clean_3.csv"


def load_zinc(n_sample=None, random_state=42):
    """
    Load the ZINC-250k dataset.

    Args:
        n_sample: If set, randomly sample this many rows (useful for demos).
                  None = load all 250k rows.
        random_state: Seed for reproducibility when sampling.

    Returns:
        A pandas DataFrame with columns: smiles, logP, qed, SAS
    """
    filepath = os.path.join(RAW_DATA_DIR, ZINC_FILE)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"ZINC-250k file not found at: {filepath}\n"
            f"Expected filename: {ZINC_FILE}"
        )

    print(f"Loading ZINC-250k from: {filepath}")
    df = pd.read_csv(filepath)

    # Normalize column names (strip whitespace, lowercase)
    df.columns = [c.strip() for c in df.columns]

    # Normalize smiles column (trim whitespace and newlines)
    if "smiles" in df.columns:
        df["smiles"] = df["smiles"].str.strip()

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols: {list(df.columns)}")

    if n_sample is not None and n_sample < len(df):
        df = df.sample(n=n_sample, random_state=random_state).reset_index(drop=True)
        print(f"  Sampled {n_sample:,} rows (random_state={random_state})")

    return df


def validate_zinc(df, smiles_col="smiles"):
    """
    Remove rows with missing, empty, or RDKit-unparseable SMILES.

    Returns:
        A cleaned DataFrame.
    """
    before = len(df)

    # Drop null / whitespace-only
    df = df.dropna(subset=[smiles_col])
    df = df[df[smiles_col].str.strip() != ""].copy()

    if HAS_RDKIT:
        valid = df[smiles_col].apply(
            lambda s: Chem.MolFromSmiles(str(s)) is not None
        )
        df = df[valid].copy()

    df = df.reset_index(drop=True)
    print(f"  Validation: {before:,} → {len(df):,} valid SMILES ({before - len(df):,} removed)")
    return df


def describe_zinc(df):
    """Print a simple statistical summary of the ZINC dataset properties."""
    print("\n" + "=" * 60)
    print("  ZINC-250k Property Summary")
    print("=" * 60)
    for col in ["logP", "qed", "SAS"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col}: mean={vals.mean():.3f} | min={vals.min():.3f} | max={vals.max():.3f}")

    print(f"\n  Total valid molecules: {len(df):,}")
    print(f"  Drug-likeness (qed > 0.5): {(df['qed'] > 0.5).sum():,} ({(df['qed'] > 0.5).mean()*100:.1f}%)")
    print(f"  Easy to synthesize (SAS < 4): {(df['SAS'] < 4).sum():,} ({(df['SAS'] < 4).mean()*100:.1f}%)")


if __name__ == "__main__":
    print("\n🔬 Loading ZINC-250k Dataset")
    print("─" * 60)

    # 1. Load full dataset (or sample for speed)
    df = load_zinc()

    # 2. Validate SMILES
    df = validate_zinc(df)

    # 3. Print property stats
    describe_zinc(df)

    # 4. Save a clean sample for use in Streamlit demo
    demo_sample = df.sample(n=min(1000, len(df)), random_state=42)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "zinc_demo_sample.csv")
    demo_sample[["smiles"]].to_csv(out_path, index=False)
    print(f"\n✅ Saved 1,000-molecule demo sample to: {out_path}")
    print("   (Contains only 'smiles' column — ready for Streamlit Batch Upload!)")
