"""
inspect_data.py – Beginner-friendly script to inspect the Tox21 dataset

This script walks you through understanding your dataset step by step:
  1. Load the CSV file
  2. Show basic shape and column names
  3. Identify the SMILES column (molecular structures)
  4. Identify target label columns (toxicity assays)
  5. Check for missing values
  6. Check for invalid SMILES strings
  7. Show class distribution (toxic vs non-toxic per assay)

Run this script first to understand your data before doing anything else!
"""

import os
import pandas as pd
import numpy as np

# ── Step 1: Load the CSV ──────────────────────────────────────

print("=" * 60)
print("  STEP 1: Loading the dataset")
print("=" * 60)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "tox21.csv")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded: {DATA_PATH}")
print(f"   Rows (compounds): {df.shape[0]}")
print(f"   Columns: {df.shape[1]}")


# ── Step 2: Show column names & types ─────────────────────────

print("\n" + "=" * 60)
print("  STEP 2: Column names & data types")
print("=" * 60)

print(f"\n{'Column Name':<30} {'Data Type':<15} {'Non-Null Count'}")
print("-" * 65)
for col in df.columns:
    non_null = df[col].notna().sum()
    print(f"{col:<30} {str(df[col].dtype):<15} {non_null}/{len(df)}")


# ── Step 3: Identify the SMILES column ────────────────────────

print("\n" + "=" * 60)
print("  STEP 3: Finding the SMILES column")
print("=" * 60)

# SMILES columns typically contain strings like 'CCO', 'c1ccccc1', etc.
# We look for columns named 'smiles', 'SMILES', or 'canonical_smiles'
smiles_candidates = [c for c in df.columns if 'smiles' in c.lower()]

if smiles_candidates:
    smiles_col = smiles_candidates[0]
    print(f"✅ SMILES column found: '{smiles_col}'")
    print(f"\n   First 5 SMILES examples:")
    for i, smi in enumerate(df[smiles_col].head(5)):
        print(f"   {i+1}. {smi}")
else:
    # Fallback: look for object-type columns with chemical-looking strings
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"⚠️  No column named 'smiles' found. Object columns: {obj_cols}")
    smiles_col = obj_cols[0] if obj_cols else None
    if smiles_col:
        print(f"   Best guess: '{smiles_col}'")
        print(f"   First 3 values: {df[smiles_col].head(3).tolist()}")


# ── Step 4: Identify target label columns ─────────────────────

print("\n" + "=" * 60)
print("  STEP 4: Identifying target labels (toxicity assays)")
print("=" * 60)

# Target columns are typically numeric (0/1) and NOT the SMILES or ID column
non_target_cols = set()
if smiles_col:
    non_target_cols.add(smiles_col)
# Also exclude any 'id' or 'mol_id' columns
for c in df.columns:
    if 'id' in c.lower() or 'name' in c.lower():
        non_target_cols.add(c)

target_cols = [c for c in df.columns if c not in non_target_cols]

print(f"\n   Found {len(target_cols)} potential target columns:")
print(f"   {target_cols}")

# Show unique values for each target column
print(f"\n   Unique values per target column:")
for col in target_cols:
    unique = sorted(df[col].dropna().unique())
    print(f"   • {col}: {unique}")


# ── Step 5: Check for missing values ──────────────────────────

print("\n" + "=" * 60)
print("  STEP 5: Missing values analysis")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(1)

print(f"\n{'Column':<30} {'Missing':<10} {'Percentage'}")
print("-" * 55)
for col in df.columns:
    if missing[col] > 0:
        print(f"{col:<30} {missing[col]:<10} {missing_pct[col]}%")
    else:
        print(f"{col:<30} {'0':<10} 0.0%")

total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\n   Total missing cells: {total_missing}/{total_cells} ({total_missing/total_cells*100:.1f}%)")


# ── Step 6: Check for invalid SMILES ──────────────────────────

print("\n" + "=" * 60)
print("  STEP 6: Checking for invalid SMILES strings")
print("=" * 60)

if smiles_col:
    # Check for empty/null SMILES
    null_smiles = df[smiles_col].isna().sum()
    empty_smiles = (df[smiles_col].astype(str).str.strip() == '').sum()
    print(f"   Null SMILES:  {null_smiles}")
    print(f"   Empty SMILES: {empty_smiles}")

    # Try to validate with RDKit if available
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")

        valid = 0
        invalid = 0
        invalid_examples = []

        for smi in df[smiles_col].dropna():
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid += 1
            else:
                invalid += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(smi)

        print(f"\n   ✅ Valid SMILES:   {valid}")
        print(f"   ❌ Invalid SMILES: {invalid}")
        if invalid_examples:
            print(f"   Invalid examples: {invalid_examples}")
    except ImportError:
        print("   (RDKit not installed – skipping SMILES validation)")


# ── Step 7: Class distribution ────────────────────────────────

print("\n" + "=" * 60)
print("  STEP 7: Class distribution (toxic vs non-toxic)")
print("=" * 60)

for col in target_cols:
    tested = df[col].notna().sum()
    if tested == 0:
        continue
    toxic = (df[col] == 1).sum()
    non_toxic = (df[col] == 0).sum()
    imbalance = toxic / tested * 100 if tested > 0 else 0

    print(f"\n   📊 {col}:")
    print(f"      Tested:     {tested} compounds")
    print(f"      Toxic (1):  {toxic} ({imbalance:.1f}%)")
    print(f"      Non-toxic:  {non_toxic} ({100-imbalance:.1f}%)")
    print(f"      Untested:   {len(df) - tested}")


# ── Summary ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  📋 SUMMARY")
print("=" * 60)
print(f"   Dataset:      Tox21")
print(f"   Compounds:    {len(df)}")
print(f"   SMILES col:   {smiles_col}")
print(f"   Target cols:  {len(target_cols)} assays")
print(f"   Missing data: {total_missing/total_cells*100:.1f}% of all cells")
print(f"\n   💡 Recommendation: Start with one assay (e.g., '{target_cols[0]}'),")
print(f"      filter out missing labels, and build a binary classifier.")
print("=" * 60)
