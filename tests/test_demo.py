from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_utils import (  # noqa: E402
    build_scaled_feature_vector,
    load_model_artifact,
    load_priority_toxin_dict,
    lookup_priority_toxin,
    predict_with_model,
)


def test_priority_toxin_dictionary_contains_expected_entries():
    toxin_dict = load_priority_toxin_dict()

    formaldehyde = lookup_priority_toxin("C=O", toxin_dict)
    mic = lookup_priority_toxin("CN=C=O", toxin_dict)
    troglitazone_smiles = next(
        smiles for smiles, meta in toxin_dict.items()
        if "troglitazone" in str(meta.get("name", "")).lower()
    )
    troglitazone = lookup_priority_toxin(troglitazone_smiles, toxin_dict)

    assert formaldehyde is not None
    assert formaldehyde["name"] == "Formaldehyde"
    assert mic is not None
    assert "Methyl Isocyanate" in mic["name"]
    assert troglitazone is not None
    assert "troglitazone" in troglitazone["name"].lower()


def test_runtime_loader_prefers_best_model():
    artifact = load_model_artifact(prefer_best=True)

    assert artifact is not None
    assert Path(artifact["artifact_path"]).name == "best_model.pkl"
    assert len(artifact["feature_names"]) > 1000


def test_non_dictionary_compound_flows_into_ml_inference():
    artifact = load_model_artifact(prefer_best=True)
    assert artifact is not None

    toxin_dict = load_priority_toxin_dict()
    assert lookup_priority_toxin("CCO", toxin_dict) is None

    inference = predict_with_model("CCO", artifact)

    assert inference["verdict"] in {"SAFE", "UNCERTAIN", "CRITICAL HAZARD"}
    assert inference["safe_threshold"] < inference["hazard_threshold"]
    if inference["verdict"] == "SAFE":
        assert float(inference["probability"][1]) <= inference["safe_threshold"]
    elif inference["verdict"] == "CRITICAL HAZARD":
        assert float(inference["probability"][1]) >= inference["hazard_threshold"]
    else:
        assert inference["safe_threshold"] < float(inference["probability"][1]) < inference["hazard_threshold"]

    assert inference["prediction"] in {0, 1}
    assert inference["feature_vector"].shape == (1, len(artifact["feature_names"]))
    assert 0.0 <= float(inference["probability"][1]) <= 1.0


def test_scaled_feature_vector_matches_artifact_shape():
    artifact = load_model_artifact(prefer_best=True)
    assert artifact is not None

    payload = build_scaled_feature_vector("CCO", artifact)

    assert payload["feature_vector"].shape[1] == len(artifact["feature_names"])
    assert payload["canonical_smiles"] == "CCO"


def test_processed_labels_and_features_stay_aligned():
    labels = pd.read_csv(ROOT / "data" / "processed" / "labels.csv")
    features = pd.read_csv(ROOT / "data" / "processed" / "features.csv")

    assert len(labels) == len(features)
    assert list(labels.columns)[:2] == ["smiles", "toxicity"]
    assert sum(col.startswith("FP_") for col in features.columns) == 1024


def test_compare_chembl_experiment_imports_as_module_and_script():
    importlib.import_module("src.compare_chembl_experiment")
    result = runpy.run_path(str(ROOT / "src" / "compare_chembl_experiment.py"), run_name="__codex_test__")

    assert "run_one" in result
