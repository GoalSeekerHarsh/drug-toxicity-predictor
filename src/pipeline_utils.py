"""
pipeline_utils.py – Shared helpers for training, inference, and reporting.

This module centralizes the contract that every consumer should follow:
  - load the preferred model artifact
  - canonicalize and check the priority toxin dictionary
  - build scaled feature vectors using the same descriptor/fingerprint rules
  - split datasets consistently
  - compute and persist a standard metrics payload
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from .feature_engineering import (
        compute_descriptors,
        compute_morgan_fingerprint,
        smiles_to_mol,
        stabilize_descriptor_dict,
    )
except ImportError:
    from feature_engineering import (  # type: ignore
        compute_descriptors,
        compute_morgan_fingerprint,
        smiles_to_mol,
        stabilize_descriptor_dict,
    )

try:
    from rdkit import Chem
except ImportError as exc:
    raise ImportError("RDKit is required for pipeline_utils.") from exc


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_SAFE_THRESHOLD = 0.30
DEFAULT_HAZARD_THRESHOLD = 0.62


def resolve_label_column(labels: pd.DataFrame) -> str:
    """Return the toxicity target column while ignoring metadata columns."""
    if "toxicity" in labels.columns:
        return "toxicity"

    ignored = {"smiles", "source"}
    for column in labels.columns:
        if column not in ignored:
            return column

    raise ValueError("Could not infer the label column from labels.csv")


def get_feature_partitions(feature_names: Iterable[str], scaler=None) -> tuple[list[str], list[str]]:
    """Split features into continuous descriptors and fingerprint bits."""
    feature_names = list(feature_names)
    fingerprint_names = [name for name in feature_names if name.startswith("FP_")]

    scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
    if scaler_feature_names:
        continuous_names = [name for name in scaler_feature_names if name in feature_names and not name.startswith("FP_")]
    else:
        continuous_names = [name for name in feature_names if not name.startswith("FP_")]

    # Keep any unexpected non-FP columns in artifact order rather than silently dropping them.
    seen = set(continuous_names)
    continuous_names.extend(
        [name for name in feature_names if not name.startswith("FP_") and name not in seen]
    )

    return continuous_names, fingerprint_names


def stratified_train_val_test_split(
    X,
    y,
    extra_arrays=None,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
):
    """Stratified split into train/validation/test, with aligned extra arrays."""
    if extra_arrays is None:
        extra_arrays = []

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    split_args = [X, y, *extra_arrays]
    temp = train_test_split(
        *split_args,
        test_size=test_ratio,
        random_state=random_state,
        stratify=y,
    )
    X_temp, X_test = temp[0], temp[1]
    y_temp, y_test = temp[2], temp[3]
    extras_temp = []
    extras_test = []
    if extra_arrays:
        extras_pairs = temp[4:]
        extras_temp = extras_pairs[0::2]
        extras_test = extras_pairs[1::2]

    val_fraction = val_ratio / (train_ratio + val_ratio)
    split_args2 = [X_temp, y_temp, *extras_temp]
    temp2 = train_test_split(
        *split_args2,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_temp,
    )
    X_train, X_val = temp2[0], temp2[1]
    y_train, y_val = temp2[2], temp2[3]
    extras_train = []
    extras_val = []
    if extras_temp:
        extras_pairs2 = temp2[4:]
        extras_train = extras_pairs2[0::2]
        extras_val = extras_pairs2[1::2]

    return X_train, X_val, X_test, y_train, y_val, y_test, extras_train, extras_val, extras_test


def build_sample_weights(labels: pd.DataFrame, chembl_weight=0.5) -> np.ndarray:
    """Build source-aware sample weights while keeping Tox21 dominant."""
    if "source" not in labels.columns:
        return np.ones(len(labels), dtype=float)

    source = labels["source"].astype(str).values
    return np.where(source == "chembl", float(chembl_weight), 1.0).astype(float)


def classify_probabilities(y_proba, decision_threshold=DEFAULT_HAZARD_THRESHOLD) -> np.ndarray:
    """Convert toxic-class probabilities into hard labels with an explicit threshold."""
    return (np.asarray(y_proba, dtype=float) >= float(decision_threshold)).astype(int)


def compute_metrics_dict(y_true, y_proba, decision_threshold=DEFAULT_HAZARD_THRESHOLD) -> dict:
    """Compute the standard binary classification metrics payload."""
    y_pred = classify_probabilities(y_proba, decision_threshold=decision_threshold)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "decision_threshold": float(decision_threshold),
    }


def save_metrics_report(filename: str, metrics: dict, extra_metadata: dict | None = None) -> Path:
    """Persist a metrics payload into reports/ as JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    if extra_metadata:
        payload.update(extra_metadata)

    report_path = REPORTS_DIR / filename
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return report_path


def save_feature_pipeline_artifact(
    scaler,
    feature_names,
    filename: str = "feature_pipeline.pkl",
    extra_metadata: dict | None = None,
) -> Path:
    """Persist the reusable inference preprocessing contract to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scaler": scaler,
        "feature_names": list(feature_names),
        "safe_threshold": float(DEFAULT_SAFE_THRESHOLD),
        "hazard_threshold": float(DEFAULT_HAZARD_THRESHOLD),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    pipeline_path = MODELS_DIR / filename
    joblib.dump(payload, pipeline_path)
    return pipeline_path


def load_model_artifact(prefer_best=True):
    """Load the preferred production artifact and annotate it with path metadata."""
    preferred = [
        MODELS_DIR / "best_model.pkl",
        MODELS_DIR / "tuned_xgboost_model.pkl",
        MODELS_DIR / "baseline_best_model.pkl",
    ]
    if not prefer_best:
        preferred = [preferred[1], preferred[0], preferred[2]]

    for path in preferred:
        if path.exists():
            artifact = dict(joblib.load(path))
            artifact.setdefault("model_name", path.stem.replace("_", " ").title())
            artifact["artifact_path"] = str(path)
            artifact["display_name"] = artifact.get("model_name", path.stem.replace("_", " ").title())
            artifact["feature_names"] = list(artifact.get("feature_names", []))
            artifact.setdefault("safe_threshold", float(DEFAULT_SAFE_THRESHOLD))
            artifact.setdefault("hazard_threshold", float(DEFAULT_HAZARD_THRESHOLD))
            return artifact

    return None


def load_priority_toxin_dict() -> dict:
    """Load the offline priority toxin dictionary. Missing files return an empty dict."""
    data_path = DATA_DIR / "priority_toxins.json"
    try:
        with data_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def normalize_lookup_text(value: str) -> str:
    """Normalize free-text names for dictionary and metadata lookup."""
    if not isinstance(value, str):
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _iter_priority_toxin_aliases(entry: dict) -> set[str]:
    """Yield normalized aliases for one priority toxin metadata record."""
    aliases = set()
    for raw_value in (entry.get("name"), entry.get("chembl_id"), entry.get("source")):
        normalized = normalize_lookup_text(str(raw_value or ""))
        if normalized:
            aliases.add(normalized)

    name = str(entry.get("name", "") or "")
    for match in re.findall(r"\(([^)]+)\)", name):
        normalized = normalize_lookup_text(match)
        if normalized:
            aliases.add(normalized)

    return aliases


def canonicalize_smiles_input(smiles_or_mol) -> tuple[object | None, str | None]:
    """Normalize a SMILES string or RDKit Mol to a Mol + canonical SMILES."""
    if smiles_or_mol is None:
        return None, None

    mol = smiles_or_mol if hasattr(smiles_or_mol, "GetNumAtoms") else smiles_to_mol(smiles_or_mol)
    if mol is None:
        return None, None

    return mol, Chem.MolToSmiles(mol, canonical=True)


def lookup_priority_toxin(smiles_or_mol, toxin_dict=None):
    """Return the matched toxin entry for a canonicalized structure, else None."""
    toxin_dict = toxin_dict if toxin_dict is not None else load_priority_toxin_dict()
    mol, canonical_smiles = canonicalize_smiles_input(smiles_or_mol)
    if mol is None or canonical_smiles is None:
        return None

    entry = toxin_dict.get(canonical_smiles)
    if not isinstance(entry, dict):
        return None

    payload = dict(entry)
    payload["canonical_smiles"] = canonical_smiles
    return payload


def lookup_priority_toxin_by_name(name: str, toxin_dict=None):
    """Return a matched toxin entry for an offline name / alias lookup, else None."""
    toxin_dict = toxin_dict if toxin_dict is not None else load_priority_toxin_dict()
    normalized_query = normalize_lookup_text(name)
    if not normalized_query:
        return None

    for canonical_smiles, entry in toxin_dict.items():
        if not isinstance(entry, dict):
            continue
        aliases = _iter_priority_toxin_aliases(entry)
        if normalized_query in aliases:
            payload = dict(entry)
            payload["canonical_smiles"] = canonical_smiles
            return payload

    return None


def _build_raw_feature_row(descriptors: dict, fingerprint: np.ndarray, feature_names: list[str]) -> dict:
    """Build one raw feature row in artifact order from descriptors + fingerprint bits."""
    row = {}
    for name in feature_names:
        if name.startswith("FP_"):
            try:
                bit_idx = int(name.split("_")[1])
                row[name] = float(fingerprint[bit_idx]) if bit_idx < len(fingerprint) else 0.0
            except (IndexError, ValueError):
                row[name] = 0.0
        else:
            row[name] = float(descriptors.get(name, 0.0))
    return row


def transform_feature_frame(feature_frame: pd.DataFrame, artifact) -> np.ndarray:
    """Transform a raw feature frame into the exact matrix the artifact expects."""
    feature_names = list(artifact["feature_names"])
    scaler = artifact.get("scaler")
    ordered = feature_frame.reindex(columns=feature_names, fill_value=0.0).copy()
    ordered = ordered.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ordered.replace([np.inf, -np.inf], 0.0, inplace=True)

    expected = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    if scaler is None:
        return ordered.to_numpy(dtype=float)

    if expected == len(feature_names):
        return scaler.transform(ordered)

    continuous_names, fingerprint_names = get_feature_partitions(feature_names, scaler=scaler)
    continuous_frame = ordered.reindex(columns=continuous_names, fill_value=0.0)
    fingerprint_frame = ordered.reindex(columns=fingerprint_names, fill_value=0.0)
    scaled_continuous = scaler.transform(continuous_frame)
    return np.hstack([scaled_continuous, fingerprint_frame.to_numpy(dtype=float)])


def build_scaled_feature_vector(smiles_or_mol, artifact) -> dict:
    """Build the exact scaled feature vector that a saved artifact expects."""
    mol, canonical_smiles = canonicalize_smiles_input(smiles_or_mol)
    if mol is None or canonical_smiles is None:
        raise ValueError("Invalid SMILES string.")

    descriptors = stabilize_descriptor_dict(compute_descriptors(mol))
    fingerprint = compute_morgan_fingerprint(mol, radius=2, n_bits=1024)
    if descriptors is None or fingerprint is None:
        raise ValueError("Failed to compute chemical features.")

    raw_row = _build_raw_feature_row(descriptors, fingerprint, list(artifact["feature_names"]))
    raw_frame = pd.DataFrame([raw_row], columns=list(artifact["feature_names"]))
    feature_vector = transform_feature_frame(raw_frame, artifact)

    return {
        "feature_vector": feature_vector,
        "descriptors": descriptors,
        "fingerprint": fingerprint,
        "canonical_smiles": canonical_smiles,
        "molecule": mol,
    }


def predict_with_model(
    smiles_or_mol,
    artifact,
    safe_threshold=DEFAULT_SAFE_THRESHOLD,
    hazard_threshold=DEFAULT_HAZARD_THRESHOLD,
) -> dict:
    """Run pure model inference without any UI concerns."""
    payload = build_scaled_feature_vector(smiles_or_mol, artifact)
    model = artifact["model"]
    probability = model.predict_proba(payload["feature_vector"])[0]
    toxic_probability = float(probability[1])
    if toxic_probability >= hazard_threshold:
        verdict = "CRITICAL HAZARD"
        prediction = 1
    elif toxic_probability <= safe_threshold:
        verdict = "SAFE"
        prediction = 0
    else:
        verdict = "UNCERTAIN"
        prediction = 0

    payload.update(
        {
            "probability": probability,
            "prediction": prediction,
            "verdict": verdict,
            "safe_threshold": float(safe_threshold),
            "hazard_threshold": float(hazard_threshold),
        }
    )
    return payload
