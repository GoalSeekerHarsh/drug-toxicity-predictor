"""
fetch_chembl_withdrawn.py – Pull withdrawn drugs from the ChEMBL REST API

This script queries the European Bioinformatics Institute (EBI) ChEMBL database
for drugs that have been withdrawn from the market due to safety/toxicity concerns.

Each entry is validated via RDKit, canonicalized, and saved to CSV for downstream
integration into the ToxPredict training pipeline and priority toxin dictionary.

Usage:
    python scripts/fetch_chembl_withdrawn.py
"""

import os
import sys
import json
import requests
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("ERROR: RDKit is required. Install via: pip install rdkit")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "chembl_withdrawn.csv")

CHEMBL_API_URL = (
    "https://www.ebi.ac.uk/chembl/api/data/drug.json"
    "?max_phase=4&first_approval__isnull=false&withdrawn_flag=true&limit=500"
)


def canonicalize(smiles: str):
    """Parse and canonicalize a SMILES string via RDKit. Returns None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fetch_chembl_withdrawn():
    """
    Query the ChEMBL REST API for withdrawn drugs.
    
    Returns a list of dicts with keys:
        chembl_id, name, smiles, canonical_smiles, withdrawn_flag,
        withdrawn_reason, hazard_class, source
    """
    print("📡 Fetching withdrawn drugs from ChEMBL REST API...")
    print(f"   URL: {CHEMBL_API_URL[:80]}...")

    try:
        response = requests.get(CHEMBL_API_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ API request failed: {e}")
        # Fall back to local file if user has one
        local_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "drug.json")
        alt_path = os.path.expanduser("~/Downloads/drug.json")
        for p in [local_path, alt_path]:
            if os.path.exists(p):
                print(f"   ⚠️  Using local fallback: {p}")
                with open(p) as f:
                    data = json.load(f)
                break
        else:
            print("   No local fallback found. Exiting.")
            sys.exit(1)
    else:
        data = response.json()

    drugs = data.get("drugs", [])
    print(f"   Received {len(drugs)} total drug entries from ChEMBL")

    results = []
    skipped_no_smiles = 0
    skipped_invalid = 0

    for drug in drugs:
        # Extract SMILES
        structures = drug.get("molecule_structures")
        if not structures or not structures.get("canonical_smiles"):
            skipped_no_smiles += 1
            continue

        raw_smiles = structures["canonical_smiles"]
        canonical = canonicalize(raw_smiles)
        if canonical is None:
            skipped_invalid += 1
            continue

        # Extract drug name (prefer pref_name, fall back to synonyms)
        name = "Unknown"
        if drug.get("molecule_synonyms"):
            for syn in drug["molecule_synonyms"]:
                if syn.get("syn_type") in ("INN", "BAN", "USAN", "FDA"):
                    name = syn["molecule_synonym"]
                    break
            if name == "Unknown" and drug["molecule_synonyms"]:
                name = drug["molecule_synonyms"][0].get("molecule_synonym", "Unknown")

        # Extract withdrawn reason if available
        withdrawn_flag = drug.get("withdrawn_flag", "0")

        # Determine hazard class from ATC classification
        hazard_class = "Withdrawn Drug"
        atc = drug.get("atc_classification", [])
        if atc:
            hazard_class = f"Withdrawn Drug ({atc[0].get('description', '')[:60]})"

        results.append({
            "chembl_id": drug.get("molecule_chembl_id", ""),
            "name": name,
            "smiles": raw_smiles,
            "canonical_smiles": canonical,
            "withdrawn_flag": withdrawn_flag,
            "hazard_class": hazard_class,
            "source": "ChEMBL",
        })

    print(f"\n   ✅ Valid entries:      {len(results)}")
    print(f"   ⚠️  Skipped (no SMILES): {skipped_no_smiles}")
    print(f"   ⚠️  Skipped (invalid):   {skipped_invalid}")

    return results


def save_to_csv(entries, output_path):
    """Save the cleaned entries to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(entries)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved {len(df)} withdrawn drugs to: {output_path}")
    
    # Print a sample
    print("\n   Sample entries:")
    for _, row in df.head(5).iterrows():
        print(f"     {row['name'][:30]:<32} {row['canonical_smiles'][:40]}")
    
    return df


if __name__ == "__main__":
    print("\n🧪 ChEMBL Withdrawn Drugs Fetcher")
    print("=" * 60)

    entries = fetch_chembl_withdrawn()
    df = save_to_csv(entries, OUTPUT_FILE)

    # Summary statistics
    truly_withdrawn = df[df["withdrawn_flag"] == "1"]
    print(f"\n📊 Summary:")
    print(f"   Total with valid SMILES:  {len(df)}")
    print(f"   Truly withdrawn (flag=1): {len(truly_withdrawn)}")
    print(f"   Other (flag=0):           {len(df) - len(truly_withdrawn)}")

    print("\n✅ Done!")
