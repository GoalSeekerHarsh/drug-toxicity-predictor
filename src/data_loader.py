"""
data_loader.py – Load and clean the Tox21 toxicity dataset

This script performs 6 cleaning steps:
  1. Load the raw CSV file
  2. Remove rows with missing SMILES strings
  3. Validate each SMILES using RDKit (can it be parsed into a molecule?)
  4. Drop rows with invalid/unparseable SMILES
  5. Remove duplicate compounds (same SMILES string)
  6. Save the cleaned dataset to data/processed/

Each step is explained in simple language with print statements so you
can follow along when the script runs.

Usage:
    python src/data_loader.py
"""

import os
import pandas as pd
import numpy as np


# ── Configuration ──────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ══════════════════════════════════════════════════════════════
#  STEP 1: Load the raw data
# ══════════════════════════════════════════════════════════════
#  We read the CSV file that contains:
#    - A "smiles" column (text representation of each molecule)
#    - 12 toxicity assay columns (0 = non-toxic, 1 = toxic, NaN = not tested)
#    - A "mol_id" identifier column

def load_tox21(filename="tox21.csv"):
    """Load the Tox21 dataset from data/raw/.

    What this does:
        Reads the CSV file into a pandas DataFrame (a table in Python).
        If the file isn't there, it tells you where to download it.
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            f"Download from https://www.kaggle.com/datasets/epicskills/tox21-dataset\n"
            f"and place the CSV in data/raw/"
        )

    df = pd.read_csv(filepath)

    print("=" * 60)
    print("  STEP 1: Load raw data")
    print("=" * 60)
    print(f"  ✅ Loaded {len(df)} compounds with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    return df


# ══════════════════════════════════════════════════════════════
#  STEP 2: Remove rows with missing SMILES
# ══════════════════════════════════════════════════════════════
#  SMILES (Simplified Molecular Input Line Entry System) is how
#  we represent molecules as text. Example: "CCO" = ethanol.
#  If a row has no SMILES, we can't compute any features → drop it.

def remove_missing_smiles(df, smiles_col="smiles"):
    """Drop rows where the SMILES string is missing or empty.

    What this does:
        - Removes rows where SMILES is NaN (null/missing)
        - Removes rows where SMILES is just whitespace ("")
    """
    initial_count = len(df)

    # Drop rows where the SMILES column is null
    df = df.dropna(subset=[smiles_col])

    # Drop rows where SMILES is an empty string after stripping whitespace
    df = df[df[smiles_col].str.strip() != ""]

    removed = initial_count - len(df)

    print(f"\n{'=' * 60}")
    print("  STEP 2: Remove missing SMILES")
    print("=" * 60)
    print(f"  Before: {initial_count} compounds")
    print(f"  After:  {len(df)} compounds")
    print(f"  ❌ Removed {removed} rows with missing/empty SMILES")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  STEP 3: Validate molecules using RDKit
# ══════════════════════════════════════════════════════════════
#  Just because a SMILES string exists doesn't mean it's valid.
#  RDKit tries to parse each SMILES into a molecular structure.
#  If it can't → the SMILES is malformed or represents an
#  impossible molecule → we flag it as invalid.

def validate_smiles(df, smiles_col="smiles"):
    """Use RDKit to check whether each SMILES is a valid molecule.

    What this does:
        - Tries to parse each SMILES string with RDKit
        - Adds a column 'is_valid' (True/False) so you can see which failed
        - Returns the DataFrame with the new column

    Why this matters:
        Invalid SMILES would crash the feature engineering step later.
        It's better to catch them now.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # Suppress noisy RDKit warnings

    valid_flags = []
    invalid_examples = []

    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(smi))
        is_valid = mol is not None
        valid_flags.append(is_valid)

        if not is_valid and len(invalid_examples) < 5:
            invalid_examples.append(smi)

    df = df.copy()
    df["is_valid"] = valid_flags

    valid_count = sum(valid_flags)
    invalid_count = len(valid_flags) - valid_count

    print(f"\n{'=' * 60}")
    print("  STEP 3: Validate SMILES with RDKit")
    print("=" * 60)
    print(f"  ✅ Valid:   {valid_count}")
    print(f"  ❌ Invalid: {invalid_count}")
    if invalid_examples:
        print(f"  Examples of invalid SMILES: {invalid_examples}")

    return df


# ══════════════════════════════════════════════════════════════
#  STEP 4: Drop invalid rows
# ══════════════════════════════════════════════════════════════
#  Now we remove the rows that RDKit couldn't parse.
#  We also drop the helper 'is_valid' column since we no longer need it.

def drop_invalid_rows(df):
    """Remove rows flagged as invalid by RDKit.

    What this does:
        Keeps only the rows where is_valid == True, then removes
        the helper column. Clean data only from here on.
    """
    before = len(df)
    df = df[df["is_valid"] == True].drop(columns=["is_valid"])
    after = len(df)

    print(f"\n{'=' * 60}")
    print("  STEP 4: Drop invalid rows")
    print("=" * 60)
    print(f"  Before: {before} → After: {after}")
    print(f"  ❌ Dropped {before - after} compounds with unparseable SMILES")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  STEP 5: Remove duplicate compounds
# ══════════════════════════════════════════════════════════════
#  Some molecules might appear more than once (same SMILES string).
#  Duplicates can bias our model — it might see the same molecule
#  in both training and test sets, giving falsely high accuracy.
#  We keep only the first occurrence.

def remove_duplicates(df, smiles_col="smiles"):
    """Remove duplicate SMILES, keeping the first occurrence.

    What this does:
        - Identifies rows that have the exact same SMILES string
        - Keeps the first one, drops the rest
        - This prevents data leakage in train/test splits
    """
    before = len(df)
    df = df.drop_duplicates(subset=[smiles_col], keep="first")
    after = len(df)
    dupes = before - after

    print(f"\n{'=' * 60}")
    print("  STEP 5: Remove duplicate compounds")
    print("=" * 60)
    print(f"  Before: {before} → After: {after}")
    if dupes > 0:
        print(f"  ❌ Removed {dupes} duplicate SMILES")
    else:
        print(f"  ✅ No duplicates found")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  STEP 6: Save cleaned data
# ══════════════════════════════════════════════════════════════
#  Finally, we extract one toxicity endpoint (assay) for our
#  binary classification task, and save the cleaned CSV.

def get_single_endpoint(df, endpoint="NR-AR", smiles_col="smiles"):
    """Extract a single toxicity assay for binary classification.

    What this does:
        - Picks one toxicity column (default: NR-AR, androgen receptor)
        - Drops rows where that assay was NOT TESTED (NaN)
        - Returns a simple 2-column DataFrame: smiles + label (0 or 1)

    Why just one endpoint?
        Starting simple. One binary target is easier to debug.
        You can expand to multi-target later.
    """
    if endpoint not in df.columns:
        available = [c for c in df.columns if c not in [smiles_col, "mol_id"]]
        raise ValueError(f"Endpoint '{endpoint}' not found. Available: {available}")

    subset = df[[smiles_col, endpoint]].dropna()
    subset = subset.copy()
    subset[endpoint] = subset[endpoint].astype(int)

    toxic = subset[endpoint].sum()
    non_toxic = (subset[endpoint] == 0).sum()

    print(f"\n  📊 Endpoint: '{endpoint}'")
    print(f"     Total tested: {len(subset)}")
    print(f"     Toxic (1):    {toxic} ({toxic/len(subset)*100:.1f}%)")
    print(f"     Non-toxic (0): {non_toxic} ({non_toxic/len(subset)*100:.1f}%)")

    return subset


def save_cleaned(df, filename="tox21_cleaned.csv"):
    """Save the cleaned DataFrame to data/processed/."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print("  STEP 6: Save cleaned data")
    print("=" * 60)
    print(f"  ✅ Saved to {output_path}")
    print(f"  Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return output_path


# ══════════════════════════════════════════════════════════════
#  MAIN: Run all 6 steps in sequence
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🧪 Tox21 Data Cleaning Pipeline")
    print("━" * 60)

    # Step 1: Load
    df = load_tox21()

    # Step 2: Remove missing SMILES
    df = remove_missing_smiles(df)

    # Step 3: Validate with RDKit
    df = validate_smiles(df)

    # Step 4: Drop invalid rows
    df = drop_invalid_rows(df)

    # Step 5: Remove duplicates
    df = remove_duplicates(df)

    # Step 6: Extract endpoint and save
    endpoint_df = get_single_endpoint(df)
    save_cleaned(endpoint_df)

    # Also save the full cleaned dataset (all 12 endpoints) for later use
    save_cleaned(df, filename="tox21_all_endpoints_cleaned.csv")

    print("\n━" * 60)
    print("🎉 Cleaning pipeline complete!")
    print("━" * 60)
