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

def canonicalize_smiles(smiles: str):
    """Parse + canonicalize a SMILES string via RDKit.

    Returns:
        canonical_smiles (str) if valid, else None
    """
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    try:
        from rdkit import Chem
    except ImportError as e:
        raise ImportError("RDKit is required for SMILES canonicalization. Install: pip install rdkit") from e

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_smiles_column(df, smiles_col="smiles", output_col="smiles"):
    """Canonicalize a SMILES column in-place (by default, overwrites `smiles`)."""
    before = len(df)
    canon = df[smiles_col].apply(canonicalize_smiles)
    valid_mask = canon.notna()
    df2 = df.loc[valid_mask].copy()
    df2[output_col] = canon.loc[valid_mask].values

    print(f"\n{'=' * 60}")
    print("  STEP X: Canonicalize SMILES (RDKit)")
    print("=" * 60)
    print(f"  Before: {before} → After: {len(df2)}")
    print(f"  ❌ Dropped {before - len(df2)} rows that failed canonicalization")
    return df2.reset_index(drop=True)


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


def get_multi_endpoint(df, smiles_col="smiles"):
    """Create an aggregate toxicity label across ALL 12 Tox21 assays.

    What this does:
        - Looks at all 12 toxicity columns for each molecule.
        - If the molecule is TOXIC (1) in ANY of the 12 assays → label = 1.
        - If the molecule is SAFE (0) in ALL tested assays   → label = 0.
        - Drops rows where NO assay was tested (all NaN).

    Why this is better than single endpoint:
        NR-AR alone only catches androgen receptor disruptors (4.3% of molecules).
        Multi-endpoint catches stress response, estrogen disruption, aromatase
        inhibition, p53 activation, and more → covers 36.7% of molecules.
    """
    assay_cols = [c for c in df.columns if c not in [smiles_col, "mol_id"]]
    print(f"\n  📊 Multi-Endpoint Aggregation")
    print(f"     Using {len(assay_cols)} assays: {assay_cols}")

    # Keep only rows that have at least one assay tested
    has_any_label = df[assay_cols].notna().any(axis=1)
    subset = df[has_any_label].copy()

    # Aggregate: toxic if ANY endpoint fires (max across row, ignoring NaN)
    subset["toxicity"] = subset[assay_cols].max(axis=1).fillna(0).astype(int)

    result = subset[[smiles_col, "toxicity"]].copy()

    toxic = (result["toxicity"] == 1).sum()
    non_toxic = (result["toxicity"] == 0).sum()
    total = len(result)

    print(f"     Total tested: {total}")
    print(f"     Toxic (any assay):  {toxic} ({toxic/total*100:.1f}%)")
    print(f"     Non-toxic (all):    {non_toxic} ({non_toxic/total*100:.1f}%)")

    return result


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
#  STEP 7: Load ChEMBL Withdrawn Drugs (Auxiliary Toxic Samples)
# ══════════════════════════════════════════════════════════════

def load_chembl_withdrawn(filename="chembl_withdrawn.csv"):
    """Load ChEMBL withdrawn drugs as supplementary toxic samples.

    What this does:
        - Reads the CSV produced by scripts/fetch_chembl_withdrawn.py
        - Validates each SMILES via RDKit
        - Assigns toxicity = 1 to every entry (they were withdrawn for safety)
        - Adds a 'source' column = 'chembl' so we can apply lower sample weight
          during model training

    Returns:
        DataFrame with columns [smiles, toxicity, source]
        or None if the file doesn't exist
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  ⚠️  ChEMBL file not found: {filepath}")
        print(f"     Run `python scripts/fetch_chembl_withdrawn.py` first.")
        return None

    print(f"\n{'=' * 60}")
    print("  STEP 7: Load ChEMBL Withdrawn Drugs")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} entries from {filepath}")

    # Use canonical_smiles column
    if "canonical_smiles" not in df.columns:
        print("  ❌ Missing 'canonical_smiles' column")
        return None

    # Conservative label quality: keep only truly withdrawn compounds when flag exists
    if "withdrawn_flag" in df.columns:
        before_flag = len(df)
        df = df[df["withdrawn_flag"].astype(str) == "1"].copy()
        print(f"  Conservative filter: withdrawn_flag==1 → {len(df)} (dropped {before_flag - len(df)})")

    # Canonicalize again (defensive) and drop invalid
    df = df.dropna(subset=["canonical_smiles"]).copy()
    df["smiles"] = df["canonical_smiles"].apply(canonicalize_smiles)
    df = df.dropna(subset=["smiles"]).copy()

    # Deduplicate on canonical SMILES
    before_dupes = len(df)
    df = df.drop_duplicates(subset=["smiles"], keep="first").copy()
    if before_dupes != len(df):
        print(f"  Removed {before_dupes - len(df)} duplicate ChEMBL structures (canonical SMILES)")

    result = pd.DataFrame(
        {
            "smiles": df["smiles"].values,
            "toxicity": 1,  # Auxiliary label: withdrawn => toxic, but treated as noisy
            "source": "chembl",
        }
    )

    print(f"  ✅ ChEMBL supplement: {len(result)} toxic compounds")
    return result


# ══════════════════════════════════════════════════════════════
#  MAIN: Run all steps in sequence
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🧪 Tox21 + ChEMBL Data Cleaning Pipeline")
    print("━" * 60)

    # ── Tox21 Pipeline (Steps 1-6) ────────────────────────────
    df = load_tox21()
    df = remove_missing_smiles(df)
    df = validate_smiles(df)
    df = drop_invalid_rows(df)
    # Canonicalize BEFORE deduplication so we dedupe chemically (not string-wise)
    df = canonicalize_smiles_column(df, smiles_col="smiles", output_col="smiles")
    df = remove_duplicates(df, smiles_col="smiles")

    # Extract multi-endpoint aggregate label
    endpoint_df = get_multi_endpoint(df)
    endpoint_df["source"] = "tox21"  # Tag source for sample weighting

    # ── ChEMBL Supplement (Step 7) ────────────────────────────
    chembl_df = load_chembl_withdrawn()

    if chembl_df is not None and len(chembl_df) > 0:
        # Deduplicate: Tox21 labels take precedence
        tox21_smiles = set(endpoint_df["smiles"].values)
        chembl_new = chembl_df[~chembl_df["smiles"].isin(tox21_smiles)].copy()

        overlap = len(chembl_df) - len(chembl_new)
        print(f"\n  📊 Merge Statistics:")
        print(f"     Tox21 compounds:         {len(endpoint_df)}")
        print(f"     ChEMBL compounds:         {len(chembl_df)}")
        print(f"     Overlap (Tox21 wins):     {overlap}")
        print(f"     New from ChEMBL:          {len(chembl_new)}")

        # Concatenate
        merged = pd.concat([endpoint_df, chembl_new], ignore_index=True)
        print(f"     Final merged dataset:     {len(merged)} compounds")
    else:
        merged = endpoint_df
        print("\n  ⚠️  No ChEMBL data available, using Tox21 only.")

    # ── Save ──────────────────────────────────────────────────
    save_cleaned(merged, filename="labels.csv")
    save_cleaned(df, filename="tox21_all_endpoints_cleaned.csv")

    print("\n" + "━" * 60)
    print("🎉 Cleaning pipeline complete!")
    print("━" * 60)
