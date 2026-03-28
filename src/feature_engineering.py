"""
feature_engineering.py – Convert SMILES strings into ML-ready features

This module provides two types of molecular features:
  1. Molecular Descriptors  – numeric properties like weight, LogP, etc.
  2. Morgan Fingerprints    – binary "bit patterns" encoding molecular substructures

Both are returned as pandas DataFrames ready for model training.

Usage:
    python src/feature_engineering.py

What is a SMILES string?
    SMILES = "Simplified Molecular Input Line Entry System"
    It's a way to write a molecule as text.
    Example: "CCO" = ethanol (two carbons + one oxygen)
             "c1ccccc1" = benzene (aromatic ring)

What is RDKit?
    RDKit is a free chemistry toolkit for Python.
    It can read SMILES, compute properties, and draw molecules.
"""

import os
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm  # Progress bar library

# ──────────────────────────────────────────────────────────────
#  Import RDKit (the chemistry toolkit)
# ──────────────────────────────────────────────────────────────
# RDKit lets us:
#   - Parse SMILES strings into molecule objects
#   - Compute ~200 molecular descriptors (weight, LogP, etc.)
#   - Generate fingerprints (structural "barcodes" of molecules)

try:
    from rdkit import Chem                       # Core chemistry module
    from rdkit.Chem import Descriptors           # ~200 molecular property calculators
    from rdkit.Chem import rdMolDescriptors      # Additional descriptors
    from rdkit.Chem import AllChem               # Morgan fingerprints live here
    from rdkit import DataStructs                # Convert fingerprints to arrays
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # Suppress noisy RDKit warnings
except ImportError:
    raise ImportError(
        "RDKit is required but not installed.\n"
        "Install with: pip install rdkit"
    )


# ── Configuration ──────────────────────────────────────────────
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ══════════════════════════════════════════════════════════════
#  HELPER: Parse a SMILES string into a molecule object
# ══════════════════════════════════════════════════════════════

def smiles_to_mol(smiles):
    """Convert a SMILES string to an RDKit Mol object.

    What this does:
        RDKit reads the text (e.g. "CCO") and builds a molecular
        structure in memory. If the SMILES is invalid or empty,
        it returns None instead of crashing.

    Args:
        smiles: A SMILES string like "CCO" or "c1ccccc1"

    Returns:
        An RDKit Mol object, or None if the SMILES is invalid
    """
    # Guard against non-string or empty inputs
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None

    # Try to parse the SMILES → molecule
    mol = Chem.MolFromSmiles(smiles)
    # mol will be None if RDKit can't understand the SMILES
    return mol


# ══════════════════════════════════════════════════════════════
#  PART 1: Molecular Descriptors
# ══════════════════════════════════════════════════════════════
#
#  Descriptors are NUMERIC PROPERTIES of a molecule, for example:
#    - MolWt:       Molecular weight (how heavy the molecule is)
#    - MolLogP:     Lipophilicity (how well it dissolves in fat vs water)
#    - NumHDonors:  Number of hydrogen bond donors
#    - TPSA:        Topological polar surface area
#    - NumRotatableBonds: How "floppy" the molecule is
#
#  RDKit can compute ~200 of these automatically.
#  They're useful because they give the ML model interpretable,
#  physically meaningful features.

def compute_descriptors(mol):
    """Compute ALL available RDKit 2D molecular descriptors for one molecule.

    What this does:
        Loops through RDKit's built-in list of ~200 descriptor calculators
        and computes each one for the given molecule.

    Args:
        mol: An RDKit Mol object (from smiles_to_mol)

    Returns:
        A dictionary like {"MolWt": 46.07, "MolLogP": -0.31, ...}
        Returns None if mol is invalid.
    """
    if mol is None:
        return None

    desc_dict = {}

    # Descriptors.descList is a list of (name, calculator_function) pairs
    # Example: ("MolWt", <function>), ("MolLogP", <function>), ...
    for name, calculator_func in Descriptors.descList:
        try:
            # Call the calculator function on our molecule
            value = calculator_func(mol)
            desc_dict[name] = value
        except Exception:
            # If any single descriptor fails, store NaN instead of crashing
            desc_dict[name] = np.nan

    return desc_dict


def build_descriptor_dataframe(smiles_list, show_progress=True):
    """Convert a list of SMILES into a DataFrame of molecular descriptors.

    What this does:
        1. Loops through each SMILES string
        2. Parses it into a molecule
        3. Computes ~200 descriptors for that molecule
        4. Stacks everything into a pandas DataFrame

    Args:
        smiles_list:   List of SMILES strings
        show_progress: Whether to show a progress bar

    Returns:
        features_df:     DataFrame (rows = molecules, cols = descriptors)
        invalid_indices: List of row indices where SMILES couldn't be parsed
    """
    all_descriptors = []    # Will hold one dict per molecule
    invalid_indices = []    # Track which SMILES failed

    # Set up a progress bar (helpful when processing thousands of molecules)
    iterator = tqdm(
        enumerate(smiles_list),
        total=len(smiles_list),
        desc="Computing descriptors"
    ) if show_progress else enumerate(smiles_list)

    for i, smi in iterator:
        # Step A: Parse SMILES → molecule object
        mol = smiles_to_mol(smi)

        if mol is None:
            # This SMILES couldn't be parsed → skip it
            invalid_indices.append(i)
            all_descriptors.append(None)
            continue

        # Step B: Compute all ~200 descriptors for this molecule
        desc = compute_descriptors(mol)
        all_descriptors.append(desc)

    # Step C: Filter out the None entries (invalid molecules)
    valid_descriptors = [d for d in all_descriptors if d is not None]

    if not valid_descriptors:
        raise ValueError("No valid molecules found! Check your SMILES strings.")

    # Step D: Convert list of dicts → pandas DataFrame
    features_df = pd.DataFrame(valid_descriptors)

    # Step E: Clean up the numbers
    #   - Replace infinity values with NaN
    #   - Fill remaining NaN cells with the column median
    #     (median is more robust than mean to outliers)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(features_df.median(), inplace=True)
    features_df.fillna(0, inplace=True)  # Fallback: if a descriptor failed for ALL molecules, median is NaN.

    print(f"\n📊 Descriptor matrix: {features_df.shape[0]} molecules × {features_df.shape[1]} descriptors")
    print(f"   Invalid SMILES skipped: {len(invalid_indices)}")

    return features_df, invalid_indices


# ══════════════════════════════════════════════════════════════
#  PART 2: Morgan Fingerprints (Circular Fingerprints)
# ══════════════════════════════════════════════════════════════
#
#  A fingerprint is a FIXED-LENGTH BINARY VECTOR (0s and 1s)
#  that encodes which chemical substructures are present.
#
#  How Morgan fingerprints work (simplified):
#    1. Look at each atom in the molecule
#    2. Look at its neighborhood (atoms within radius=2 bonds)
#    3. Hash that neighborhood into a bit position
#    4. Set that bit to 1
#
#  The result is a vector like [0, 1, 0, 0, 1, 1, 0, ...]
#  where each position represents a particular substructure pattern.
#
#  Why use fingerprints?
#    - They capture STRUCTURAL PATTERNS that descriptors might miss
#    - They're fast to compute
#    - They're widely used in drug discovery
#
#  Typical settings:
#    - radius=2  → equivalent to ECFP4 (industry standard)
#    - nBits=1024 → fingerprint length (more bits = less collision)

def compute_morgan_fingerprint(mol, radius=2, n_bits=1024):
    """Compute a Morgan fingerprint for one molecule.

    What this does:
        Creates a binary "barcode" of the molecule's substructures.

    Args:
        mol:    An RDKit Mol object
        radius: How far around each atom to look (2 = look 2 bonds out)
        n_bits: Length of the fingerprint vector (1024 is a good default)

    Returns:
        A numpy array of 0s and 1s with length n_bits,
        or None if mol is invalid.
    """
    if mol is None:
        return None

    # Generate the Morgan fingerprint as a bit vector
    # radius=2 is equivalent to ECFP4, the industry standard
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits
    )

    # Convert the RDKit fingerprint object → numpy array
    arr = np.zeros(n_bits, dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def build_fingerprint_dataframe(smiles_list, radius=2, n_bits=1024, show_progress=True):
    """Convert a list of SMILES into a DataFrame of Morgan fingerprints.

    What this does:
        1. Parses each SMILES into a molecule
        2. Computes a Morgan fingerprint (binary vector) for each
        3. Stacks them into a DataFrame with columns "FP_0", "FP_1", ..., "FP_1023"

    Args:
        smiles_list:   List of SMILES strings
        radius:        Fingerprint radius (2 = ECFP4, 3 = ECFP6)
        n_bits:        Length of fingerprint (1024 is standard)
        show_progress: Whether to show a progress bar

    Returns:
        fp_df:           DataFrame (rows = molecules, cols = fingerprint bits)
        invalid_indices: List of row indices where SMILES couldn't be parsed
    """
    all_fingerprints = []
    invalid_indices = []

    iterator = tqdm(
        enumerate(smiles_list),
        total=len(smiles_list),
        desc="Computing fingerprints"
    ) if show_progress else enumerate(smiles_list)

    for i, smi in iterator:
        # Parse SMILES
        mol = smiles_to_mol(smi)

        if mol is None:
            invalid_indices.append(i)
            continue

        # Compute Morgan fingerprint
        fp_array = compute_morgan_fingerprint(mol, radius=radius, n_bits=n_bits)
        all_fingerprints.append(fp_array)

    if not all_fingerprints:
        raise ValueError("No valid molecules found! Check your SMILES strings.")

    # Convert to DataFrame with named columns: FP_0, FP_1, ..., FP_1023
    column_names = [f"FP_{i}" for i in range(n_bits)]
    fp_df = pd.DataFrame(all_fingerprints, columns=column_names)

    # Count how many bits are "on" on average (shows coverage)
    avg_bits_on = fp_df.sum(axis=1).mean()

    print(f"\n🔬 Fingerprint matrix: {fp_df.shape[0]} molecules × {fp_df.shape[1]} bits")
    print(f"   Radius: {radius} | Bits: {n_bits}")
    print(f"   Avg bits ON per molecule: {avg_bits_on:.1f} / {n_bits}")
    print(f"   Invalid SMILES skipped: {len(invalid_indices)}")

    return fp_df, invalid_indices


# ══════════════════════════════════════════════════════════════
#  COMBINED: Descriptors + Fingerprints together
# ══════════════════════════════════════════════════════════════

def build_combined_features(smiles_list, include_fingerprints=True,
                            fp_radius=2, fp_bits=1024, show_progress=True):
    """Build a combined feature matrix: descriptors + fingerprints.

    What this does:
        1. Computes ~200 molecular descriptors
        2. Optionally computes 1024-bit Morgan fingerprints
        3. Joins them side-by-side into one big DataFrame

    This gives the model BOTH:
        - Interpretable properties (weight, LogP, etc.)
        - Structural patterns (fingerprint bits)

    Args:
        smiles_list:          List of SMILES strings
        include_fingerprints: Whether to add Morgan fingerprints (True/False)
        fp_radius:            Fingerprint radius (default 2 = ECFP4)
        fp_bits:              Fingerprint length (default 1024)
        show_progress:        Show progress bars

    Returns:
        combined_df:     DataFrame with all features
        invalid_indices: Indices of invalid SMILES (consistent across both)
    """
    # Step 1: Compute molecular descriptors
    print("━" * 50)
    print("PART 1: Molecular Descriptors")
    print("━" * 50)
    desc_df, invalid_desc = build_descriptor_dataframe(smiles_list, show_progress)

    if not include_fingerprints:
        return desc_df, invalid_desc

    # Step 2: Compute Morgan fingerprints
    # We need to use the SAME valid SMILES (skip the same invalid ones)
    print(f"\n{'━' * 50}")
    print("PART 2: Morgan Fingerprints")
    print("━" * 50)

    # Get only the valid SMILES (exclude the ones that failed in Step 1)
    valid_smiles = [s for i, s in enumerate(smiles_list) if i not in invalid_desc]
    fp_df, invalid_fp = build_fingerprint_dataframe(
        valid_smiles, radius=fp_radius, n_bits=fp_bits, show_progress=show_progress
    )

    # Step 3: Combine descriptors + fingerprints side by side
    # Reset indices to make sure they align
    desc_df = desc_df.reset_index(drop=True)
    fp_df = fp_df.reset_index(drop=True)
    combined_df = pd.concat([desc_df, fp_df], axis=1)

    print(f"\n{'━' * 50}")
    print("COMBINED FEATURES")
    print("━" * 50)
    print(f"  📊 Descriptors:   {desc_df.shape[1]} columns")
    print(f"  🔬 Fingerprints:  {fp_df.shape[1]} columns")
    print(f"  📋 Total:         {combined_df.shape[1]} features × {combined_df.shape[0]} molecules")

    return combined_df, invalid_desc


# ══════════════════════════════════════════════════════════════
#  MAIN: Run when script is executed directly
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🧪 Feature Engineering Pipeline")
    print("=" * 60)

    # Load the cleaned data from data_loader.py
    input_path = os.path.join(PROCESSED_DATA_DIR, "labels.csv")

    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        print(f"   Run `python src/data_loader.py` first to create the cleaned data.")
        exit(1)

    data = pd.read_csv(input_path)
    smiles_col = "smiles"
    print(f"Loaded {len(data)} compounds from {input_path}")

    # Build features (descriptors + fingerprints)
    features, invalid = build_combined_features(
        data[smiles_col].tolist(),
        include_fingerprints=True,  # Set to False to skip fingerprints
        fp_radius=2,                # ECFP4 standard
        fp_bits=1024                # 1024-bit fingerprint
    )

    # Drop invalid rows from labels too (keep everything aligned)
    labels = data.drop(index=invalid).reset_index(drop=True)

    # Save to CSV
    features.to_csv(os.path.join(PROCESSED_DATA_DIR, "features.csv"), index=False)
    labels.to_csv(os.path.join(PROCESSED_DATA_DIR, "labels.csv"), index=False)

    # Also save descriptor-only version (useful for SHAP interpretation)
    desc_only, _ = build_descriptor_dataframe(
        data[smiles_col].tolist(), show_progress=False
    )
    desc_only.to_csv(os.path.join(PROCESSED_DATA_DIR, "features_descriptors_only.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("✅ Feature engineering complete!")
    print(f"   Saved to {PROCESSED_DATA_DIR}:")
    print(f"   • features.csv             ({features.shape[1]} cols: descriptors + fingerprints)")
    print(f"   • features_descriptors_only.csv ({desc_only.shape[1]} cols: descriptors only)")
    print(f"   • labels.csv               (SMILES + toxicity label)")
    print("=" * 60)
