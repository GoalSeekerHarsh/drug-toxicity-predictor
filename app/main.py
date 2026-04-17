import os
import sys
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to python path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline_utils import (
    load_model_artifact,
    load_priority_toxin_dict,
    lookup_priority_toxin_by_name,
    predict_with_model,
)
from feature_engineering import smiles_to_mol
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ToxPredict Premium API")

# Enable CORS for potential separate frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources AOT (Ahead of Time)
try:
    logger.info("Loading model artifact...")
    ARTIFACT = load_model_artifact(prefer_best=True)
    if ARTIFACT is None:
        logger.warning("No model artifact found in models/. Inference will fail.")
        
    logger.info("Loading offline priority dictionary...")
    TOXIN_DICT = load_priority_toxin_dict()
    logger.info(f"Loaded {len(TOXIN_DICT)} priority toxins.")
except Exception as e:
    logger.error(f"Failed to load resources: {e}")
    ARTIFACT = None
    TOXIN_DICT = {}


def resolve_to_smiles(query: str) -> tuple[Optional[str], Optional[str]]:
    """Query PubChem API to resolve a common name to canonical SMILES."""
    query = query.strip()
    if not query:
        return None, None
        
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/SMILES/JSON"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            smiles = data['PropertyTable']['Properties'][0]['SMILES']
            return smiles, query
    except Exception:
        pass
    return None, None


class PredictRequest(BaseModel):
    query: str

@app.post("/api/predict")
async def predict(request: PredictRequest) -> Dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    resolved_name = None
    resolved_via = "smiles"

    # 1. Direct offline lookup first (by name)
    toxin_entry = lookup_priority_toxin_by_name(query, toxin_dict=TOXIN_DICT)
    if toxin_entry is not None:
        logger.info(f"AOT match by name: {query}")
        return {
            "status": "success",
            "is_priority_toxin": True,
            "verdict": "CRITICAL HAZARD",
            "probability": 1.0,
            "confidence": 1.0,
            "source": toxin_entry.get("source"),
            "hazard_class": toxin_entry.get("hazard_class"),
            "name": toxin_entry.get("name"),
            "smiles": toxin_entry.get("canonical_smiles"),
            "resolved_via": "offline_dictionary"
        }

    # 2. Check if valid SMILES
    mol = smiles_to_mol(query)
    input_smiles = query
    
    # 3. If invalid SMILES, try to resolve via PubChem
    if mol is None:
        input_smiles, resolved_name = resolve_to_smiles(query)
        if input_smiles:
            mol = smiles_to_mol(input_smiles)
            resolved_via = "pubchem"
        else:
            raise HTTPException(status_code=400, detail=f"Could not resolve '{query}' to a valid SMILES via PubChem.")

    # 4. We now have a SMILES. Do a dictionary lookup by exact SMILES (in case PubChem resolved it)
    if resolved_via == "pubchem" and input_smiles:
        # Check dictionary by resolved SMILES
        from rdkit import Chem
        can_smiles = Chem.MolToSmiles(mol, canonical=True)
        if can_smiles in TOXIN_DICT:
            tox_meta = TOXIN_DICT[can_smiles]
            logger.info(f"AOT match by PubChem-resolved SMILES: {can_smiles}")
            return {
                "status": "success",
                "is_priority_toxin": True,
                "verdict": "CRITICAL HAZARD",
                "probability": 1.0,
                "confidence": 1.0,
                "source": tox_meta.get("source"),
                "hazard_class": tox_meta.get("hazard_class"),
                "name": tox_meta.get("name") or resolved_name,
                "smiles": can_smiles,
                "resolved_via": "pubchem_then_dict"
            }

    # 5. ML Inference
    if ARTIFACT is None:
        raise HTTPException(status_code=500, detail="XGBoost Model is not loaded. Please train first.")

    try:
        prediction = predict_with_model(mol, ARTIFACT)
        probability = float(prediction["probability"][1])
        
        # Determine confidence
        envelope_dist = prediction.get("applicability_distance", 0.0)
        radius = ARTIFACT.get("validation_envelope", {}).get("radius_threshold", 10.0)
        in_envelope = prediction.get("in_validated_envelope", True)
        
        confidence = 1.0
        if not in_envelope:
            # Drop confidence drastically if out of distribution
            ratio = min(envelope_dist / radius, 2.0)
            confidence = max(0.1, 1.0 - (ratio - 1.0))
        else:
            # Map distance to 0.7 - 0.99 range inside envelope
            confidence = 0.99 - (0.29 * (envelope_dist / radius))

        return {
            "status": "success",
            "is_priority_toxin": False,
            "verdict": prediction["verdict"],
            "probability": probability,
            "confidence": confidence,
            "smiles": input_smiles,
            "resolved_name": resolved_name,
            "resolved_via": resolved_via,
            "in_envelope": in_envelope
        }
    except Exception as e:
        logger.error(f"ML Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Return dictionary size and model info."""
    return {
        "dictionary_size": len(TOXIN_DICT),
        "model_name": ARTIFACT.get("model_name", "Unknown") if ARTIFACT else "Not Loaded"
    }

# Mount static files at the root
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
