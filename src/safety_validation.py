"""
safety_validation.py – External-style validation, calibration, and review-gating report.

This workflow complements the standard train/test reports by checking:
  - the promoted model on the current random split
  - a scaffold-based split to reduce chemical-series leakage
  - the applicability-domain gate used to force review outside the validated envelope
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

try:
    from .improve_model import calibrate_and_fit_safety_contract, tune_xgboost
    from .pipeline_utils import (
        DATA_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        apply_applicability_envelope,
        apply_runtime_decision_rule,
        build_sample_weights,
        compute_metrics_dict,
        load_model_artifact,
        resolve_label_column,
        stratified_train_val_test_split,
        transform_feature_frame,
    )
except ImportError:
    from improve_model import calibrate_and_fit_safety_contract, tune_xgboost  # type: ignore
    from pipeline_utils import (  # type: ignore
        DATA_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        apply_applicability_envelope,
        apply_runtime_decision_rule,
        build_sample_weights,
        compute_metrics_dict,
        load_model_artifact,
        resolve_label_column,
        stratified_train_val_test_split,
        transform_feature_frame,
    )


def load_processed_dataset(include_chembl=True):
    """Load processed features and labels with optional source filtering."""
    features = pd.read_csv(PROCESSED_DATA_DIR / "features.csv")
    labels = pd.read_csv(PROCESSED_DATA_DIR / "labels.csv")
    if (not include_chembl) and ("source" in labels.columns):
        keep_mask = labels["source"].astype(str).eq("tox21")
        features = features.loc[keep_mask].reset_index(drop=True)
        labels = labels.loc[keep_mask].reset_index(drop=True)
    return features, labels


def scaffold_group(smiles: str) -> str:
    """Build a scaffold grouping key for one SMILES string."""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        scaffold = ""
    scaffold = (scaffold or "").strip()
    return scaffold or f"acyclic::{smiles}"


def scaffold_train_val_test_split(features, labels, weights, random_state=42):
    """Split by Bemis-Murcko scaffold so related chemotypes stay together."""
    groups = labels["smiles"].astype(str).map(scaffold_group).values
    outer = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, temp_idx = next(outer.split(features, labels["toxicity"].values, groups=groups))

    temp_groups = groups[temp_idx]
    inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    val_sub_idx, test_sub_idx = next(
        inner.split(features.iloc[temp_idx], labels["toxicity"].iloc[temp_idx].values, groups=temp_groups)
    )
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

    return (
        features.iloc[train_idx].reset_index(drop=True),
        features.iloc[val_idx].reset_index(drop=True),
        features.iloc[test_idx].reset_index(drop=True),
        labels.iloc[train_idx].reset_index(drop=True),
        labels.iloc[val_idx].reset_index(drop=True),
        labels.iloc[test_idx].reset_index(drop=True),
        np.asarray(weights)[train_idx],
        np.asarray(weights)[val_idx],
        np.asarray(weights)[test_idx],
    )


def load_reference_scaler():
    """Load the shared unsupervised scaler contract."""
    scaler_artifact = joblib.load(MODELS_DIR / "zinc_chemical_space_scaler.pkl")
    return scaler_artifact["scaler"]


def evaluate_split_with_artifact(features_df, labels_df, artifact):
    """Evaluate one split using the saved artifact and its runtime decision rule."""
    label_col = resolve_label_column(labels_df)
    X_scaled = transform_feature_frame(features_df, artifact)
    y_true = labels_df[label_col].values
    y_proba = artifact["model"].predict_proba(X_scaled)[:, 1]
    in_envelope, distances = apply_applicability_envelope(X_scaled, artifact.get("validation_envelope"))
    metrics = compute_metrics_dict(
        y_true,
        y_proba,
        decision_threshold=float(artifact.get("hazard_threshold")),
        safe_threshold=float(artifact.get("safe_threshold")),
        in_validated_envelope=in_envelope,
    )
    runtime = apply_runtime_decision_rule(
        y_proba,
        in_validated_envelope=in_envelope,
        safe_threshold=float(artifact.get("safe_threshold")),
        hazard_threshold=float(artifact.get("hazard_threshold")),
    )
    return metrics, y_proba, runtime, in_envelope, distances


def build_failure_examples(labels_df, y_proba, runtime, in_envelope, top_n=5):
    """Collect concrete false-negative and out-of-envelope examples."""
    examples_df = labels_df[["smiles", "toxicity"]].copy()
    if "source" in labels_df.columns:
        examples_df["source"] = labels_df["source"]
    examples_df["toxicity_prob"] = y_proba
    examples_df["verdict"] = runtime["verdicts"]
    examples_df["in_validated_envelope"] = in_envelope
    examples_df["review_reason"] = runtime["review_reasons"]

    false_negatives = examples_df[
        (examples_df["toxicity"] == 1) & (examples_df["verdict"] != "CRITICAL HAZARD")
    ].sort_values("toxicity_prob", ascending=False)
    false_positives = examples_df[
        (examples_df["toxicity"] == 0) & (examples_df["verdict"] == "CRITICAL HAZARD")
    ].sort_values("toxicity_prob", ascending=False)
    out_of_envelope = examples_df[~examples_df["in_validated_envelope"]].sort_values(
        "toxicity_prob", ascending=False
    )

    def as_records(frame):
        cols = ["smiles", "toxicity", "toxicity_prob", "verdict", "in_validated_envelope", "review_reason"]
        if "source" in frame.columns:
            cols.insert(1, "source")
        return frame.head(top_n)[cols].to_dict(orient="records")

    return {
        "false_negatives": as_records(false_negatives),
        "false_positives": as_records(false_positives),
        "out_of_envelope_examples": as_records(out_of_envelope),
    }


def run_random_split_validation():
    """Evaluate the currently promoted artifact on the standard random split."""
    artifact = load_model_artifact(prefer_best=True)
    if artifact is None:
        raise FileNotFoundError("No promoted model artifact found. Run compare_chembl_experiment first.")

    features, labels = load_processed_dataset(include_chembl=True)
    label_col = resolve_label_column(labels)
    split = stratified_train_val_test_split(
        features,
        labels[label_col].values,
        extra_arrays=[labels["smiles"].values, labels.get("source", pd.Series(["unknown"] * len(labels))).values],
        random_state=42,
    )
    X_train_df, X_val_df, X_test_df = split[0], split[1], split[2]
    y_train, y_val, y_test = split[3], split[4], split[5]
    (smiles_train, source_train), (smiles_val, source_val), (smiles_test, source_test) = split[6], split[7], split[8]

    test_labels = pd.DataFrame({"smiles": smiles_test, "toxicity": y_test, "source": source_test})
    metrics, y_proba, runtime, in_envelope, distances = evaluate_split_with_artifact(X_test_df, test_labels, artifact)
    failures = build_failure_examples(test_labels, y_proba, runtime, in_envelope)
    return {
        "artifact_path": artifact["artifact_path"],
        "metrics": metrics,
        "failure_examples": failures,
    }


def run_scaffold_split_validation(chembl_weight=0.5, random_state=42):
    """Train and evaluate the model on a scaffold split for stronger leakage resistance."""
    features, labels = load_processed_dataset(include_chembl=True)
    scaler = load_reference_scaler()
    weights = build_sample_weights(labels, chembl_weight=chembl_weight)
    feature_names = features.columns.tolist()

    X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df, w_train, w_val, w_test = scaffold_train_val_test_split(
        features, labels, weights, random_state=random_state
    )

    transform_artifact = {"feature_names": feature_names, "scaler": scaler}
    X_train = transform_feature_frame(X_train_df, transform_artifact)
    X_val = transform_feature_frame(X_val_df, transform_artifact)
    X_test = transform_feature_frame(X_test_df, transform_artifact)

    y_train = y_train_df["toxicity"].values
    y_val = y_val_df["toxicity"].values
    y_test = y_test_df["toxicity"].values

    model = tune_xgboost(X_train, y_train, sample_weight=w_train, random_state=random_state)
    hazard_threshold, validation_envelope, calibration_summary, _, _ = calibrate_and_fit_safety_contract(
        model,
        X_train,
        X_val,
        y_val,
        feature_names,
        scaler,
    )
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "safe_threshold": 0.30,
        "hazard_threshold": hazard_threshold,
        "validation_envelope": validation_envelope,
        "calibration_summary": calibration_summary,
    }

    test_labels = y_test_df[["smiles", "toxicity"]].copy()
    if "source" in y_test_df.columns:
        test_labels["source"] = y_test_df["source"]
    metrics, y_proba, runtime, in_envelope, _ = evaluate_split_with_artifact(X_test_df, test_labels, artifact)
    failures = build_failure_examples(test_labels, y_proba, runtime, in_envelope)
    return {
        "metrics": metrics,
        "calibration_summary": calibration_summary,
        "failure_examples": failures,
    }


def evaluate_zinc_domain_shift():
    """Measure how often the ZINC demo sample falls outside the validated envelope."""
    artifact = load_model_artifact(prefer_best=True)
    zinc_path = PROCESSED_DATA_DIR / "zinc_demo_sample.csv"
    if artifact is None or not zinc_path.exists():
        return None

    zinc_df = pd.read_csv(zinc_path)
    if "smiles" not in zinc_df.columns:
        return None

    zinc_features = []
    valid_smiles = []
    for smiles in zinc_df["smiles"].astype(str).head(250):
        try:
            from .pipeline_utils import build_scaled_feature_vector
        except ImportError:
            from pipeline_utils import build_scaled_feature_vector  # type: ignore
        try:
            payload = build_scaled_feature_vector(smiles, artifact)
        except Exception:
            continue
        zinc_features.append(payload["feature_vector"][0])
        valid_smiles.append(smiles)

    if not zinc_features:
        return None

    zinc_matrix = np.asarray(zinc_features, dtype=float)
    in_envelope, distances = apply_applicability_envelope(zinc_matrix, artifact.get("validation_envelope"))
    return {
        "sampled_molecules": int(len(valid_smiles)),
        "validated_coverage_rate": float(in_envelope.mean()),
        "out_of_envelope_rate": float((~in_envelope).mean()),
        "example_smiles_out_of_envelope": [
            smiles for smiles, allowed in zip(valid_smiles, in_envelope) if not allowed
        ][:5],
    }


def write_safety_report(report: dict):
    """Persist JSON + Markdown safety reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "safety_validation.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    random_metrics = report["random_split"]["metrics"]
    scaffold_metrics = report["scaffold_split"]["metrics"]
    lines = [
        "# Safety Validation Report",
        "",
        "## Random Split",
        f"- Hazard precision: **{random_metrics['precision']:.4f}**",
        f"- Hazard recall: **{random_metrics['recall']:.4f}**",
        f"- Validated coverage rate: **{random_metrics.get('validated_coverage_rate', 1.0):.4f}**",
        f"- Out-of-envelope rate: **{random_metrics.get('out_of_envelope_rate', 0.0):.4f}**",
        "",
        "## Scaffold Split",
        f"- Hazard precision: **{scaffold_metrics['precision']:.4f}**",
        f"- Hazard recall: **{scaffold_metrics['recall']:.4f}**",
        f"- Validated coverage rate: **{scaffold_metrics.get('validated_coverage_rate', 1.0):.4f}**",
        f"- Out-of-envelope rate: **{scaffold_metrics.get('out_of_envelope_rate', 0.0):.4f}**",
        "",
        "## Safety Policy",
        "- No hard `SAFE` verdict is allowed outside the validated envelope.",
        "- Priority dictionary matches still bypass directly to `CRITICAL HAZARD`.",
        "",
        "## Failure Examples",
    ]
    for item in report["random_split"]["failure_examples"]["false_negatives"][:5]:
        lines.append(
            f"- False negative candidate: `{item['smiles']}` | prob={item['toxicity_prob']:.4f} | "
            f"verdict={item['verdict']} | in_envelope={item['in_validated_envelope']}"
        )
    markdown_path = REPORTS_DIR / "safety_validation.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved safety validation reports to {json_path} and {markdown_path}")


def main():
    report = {
        "random_split": run_random_split_validation(),
        "scaffold_split": run_scaffold_split_validation(),
        "zinc_domain_shift": evaluate_zinc_domain_shift(),
    }
    write_safety_report(report)


if __name__ == "__main__":
    main()
