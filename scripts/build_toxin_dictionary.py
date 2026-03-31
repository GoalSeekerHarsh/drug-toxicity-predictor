"""
build_toxin_dictionary.py – Build a comprehensive priority toxin dictionary

Merges three sources:
  1. The existing 6 hardcoded OSHA/EPA priority toxins from the app
  2. All withdrawn drugs fetched from ChEMBL
  3. (Future) Any additional regulatory lists (ECHA SVHC, EPA CompTox)

Produces a single `data/priority_toxins.json` file keyed by canonical SMILES.
The Streamlit app loads this at startup and bypasses ML inference for any match.

Usage:
    python scripts/build_toxin_dictionary.py
"""

import os
import sys
import json
import pandas as pd

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("ERROR: RDKit is required.")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUTPUT_FILE = os.path.join(DATA_DIR, "priority_toxins.json")

# The original 6 hardcoded toxins from streamlit_app.py
OSHA_EPA_TOXINS = {
    "C=O": {
        "name": "Formaldehyde",
        "hazard_class": "IARC Group 1 Carcinogen, OSHA PEL",
        "source": "OSHA/EPA Priority Hazard",
    },
    "CN=C=O": {
        "name": "Methyl Isocyanate (MIC)",
        "hazard_class": "Acute Lethal (Bhopal Disaster Agent)",
        "source": "OSHA/EPA Priority Hazard",
    },
    "C#N": {
        "name": "Hydrogen Cyanide",
        "hazard_class": "Acute Lethal (Cytochrome Oxidase Inhibitor)",
        "source": "OSHA/EPA Priority Hazard",
    },
    "c1ccccc1": {
        "name": "Benzene",
        "hazard_class": "IARC Group 1 Carcinogen, OSHA PEL",
        "source": "OSHA/EPA Priority Hazard",
    },
    "O=C(Cl)Cl": {
        "name": "Phosgene",
        "hazard_class": "Chemical Warfare Agent (Pulmonary)",
        "source": "OSHA/EPA Priority Hazard",
    },
    "[AsH3]": {
        "name": "Arsine",
        "hazard_class": "Acute Lethal (Hemolytic Agent)",
        "source": "OSHA/EPA Priority Hazard",
    },
}


def canonicalize(smiles: str):
    """Canonicalize via RDKit. Returns None if invalid."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_osha_epa():
    """Load the hardcoded OSHA/EPA toxins and canonicalize their keys."""
    print("📋 Source 1: OSHA/EPA Priority Hazards")
    entries = {}
    for smiles, meta in OSHA_EPA_TOXINS.items():
        canonical = canonicalize(smiles)
        if canonical:
            entries[canonical] = meta
    print(f"   Loaded {len(entries)} entries")
    return entries


def load_chembl_withdrawn():
    """Load ChEMBL withdrawn drugs CSV and format for dictionary."""
    csv_path = os.path.join(RAW_DIR, "chembl_withdrawn.csv")
    print(f"\n📋 Source 2: ChEMBL Withdrawn Drugs")

    if not os.path.exists(csv_path):
        print(f"   ⚠️  File not found: {csv_path}")
        print(f"   Run `python scripts/fetch_chembl_withdrawn.py` first.")
        return {}

    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} entries from CSV")

    entries = {}
    for _, row in df.iterrows():
        canonical = str(row.get("canonical_smiles", "")).strip()
        if not canonical or canonical == "nan":
            continue

        # Re-validate via RDKit to be absolutely sure
        verified = canonicalize(canonical)
        if verified is None:
            continue

        entries[verified] = {
            "name": str(row.get("name", "Unknown")),
            "hazard_class": str(row.get("hazard_class", "Withdrawn Drug")),
            "source": "ChEMBL Withdrawn Drug",
            "chembl_id": str(row.get("chembl_id", "")),
        }

    print(f"   Valid entries: {len(entries)}")
    return entries


def build_dictionary():
    """
    Merge all sources into a single dictionary.
    
    Priority order (for duplicates):
      1. OSHA/EPA (highest confidence)
      2. ChEMBL (high confidence)
    """
    print("\n🔧 Building merged dictionary...\n")

    # Start with lowest priority, overwrite with higher
    merged = {}

    # Layer 1: ChEMBL (base layer)
    chembl = load_chembl_withdrawn()
    merged.update(chembl)

    # Layer 2: OSHA/EPA (overrides ChEMBL if duplicate)
    osha = load_osha_epa()
    merged.update(osha)

    print(f"\n📊 Final Dictionary:")
    print(f"   Total unique canonical SMILES: {len(merged)}")
    print(f"   From OSHA/EPA:    {sum(1 for v in merged.values() if 'OSHA' in v.get('source', ''))}")
    print(f"   From ChEMBL:      {sum(1 for v in merged.values() if 'ChEMBL' in v.get('source', ''))}")

    return merged


def save_dictionary(dictionary, output_path):
    """Save the dictionary to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved dictionary to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    print("\n🧪 Priority Toxin Dictionary Builder")
    print("=" * 60)

    dictionary = build_dictionary()
    save_dictionary(dictionary, OUTPUT_FILE)

    # Show some famous entries
    print("\n🔍 Spot check (sample entries):")
    famous = ["Formaldehyde", "Troglitazone", "Methyl Isocyanate"]
    for name in famous:
        found = [k for k, v in dictionary.items() if name.lower() in v.get("name", "").lower()]
        if found:
            entry = dictionary[found[0]]
            print(f"   ✅ {entry['name'][:35]:<37} [{entry['source']}]")
        else:
            print(f"   ❌ {name} — NOT FOUND")

    print("\n✅ Dictionary build complete!")
