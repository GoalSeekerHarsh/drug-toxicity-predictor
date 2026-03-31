"""
zinc_baseline.py - Unsupervised Chemical Space Normalization

This script samples 25,000 molecules from the commercially available
ZINC-250k drug library to establish an unbiased, healthy mathematical 
baseline for physical molecular continuous features (LogP, Weight, etc.).
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

# Connect to src modules
sys.path.insert(0, os.path.dirname(__file__))
try:
    from feature_engineering import smiles_to_mol, compute_descriptors, stabilize_descriptor_dict
except ImportError:
    print("Error: Must run from project root.")
    sys.exit(1)

def build_zinc_baseline():
    print("🧪 Modeling Global Chemical Space via ZINC-250k...")
    zinc_path = os.path.join("data", "raw", "250k_rndm_zinc_drugs_clean_3.csv")
    if not os.path.exists(zinc_path):
        print(f"Error: Could not find {zinc_path}")
        return

    # Load and randomly sample 25,000 molecules for our baseline distribution
    print("=> Loading ZINC dataset...")
    df_zinc = pd.read_csv(zinc_path)
    sample_size = min(25000, len(df_zinc))
    df_sample = df_zinc.sample(n=sample_size, random_state=42).copy()
    
    print(f"=> Computing 208 continuous RDKit descriptors for {sample_size} commercial molecules...")
    descriptors_list = []
    
    for smiles in tqdm(df_sample['smiles'].tolist(), desc="Processing ZINC"):
        mol = smiles_to_mol(smiles)
        if mol:
            desc = stabilize_descriptor_dict(compute_descriptors(mol))
            if desc:
                descriptors_list.append(desc)
                
    df_desc = pd.DataFrame(descriptors_list)
    df_desc = df_desc.fillna(0) # Standard imputation for failures
    
    # We purposefully DO NOT compute Morgan Fingerprints (binary bits) here. 
    # Scaling binary bits ruins matrix sparsity. We only want to normalize 
    # continuous physical attributes (Weight, Polarity, Solubility) against the commercial norm.
    
    print("=> Fitting RobustScaler on ZINC continuous properties...")
    scaler = RobustScaler()
    scaler.fit(df_desc)
    
    # Save the fitted scaler AND the exact column order so the training script can apply it
    artifact = {
        "scaler": scaler,
        "feature_names": df_desc.columns.tolist()
    }
    
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "zinc_chemical_space_scaler.pkl")
    joblib.dump(artifact, out_path)
    
    print(f"✅ Saved Unbiased ZINC Normalizer to {out_path}")
    print("This matrix will now be used to normalize Tox21 outliers during XGBoost training.")

if __name__ == "__main__":
    build_zinc_baseline()
