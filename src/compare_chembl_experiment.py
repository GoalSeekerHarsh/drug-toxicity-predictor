"""
compare_chembl_experiment.py – Compare model performance with vs without ChEMBL

This script does NOT modify raw datasets.
It simply trains two XGBoost models from the already-built processed files:
  - data/processed/features.csv
  - data/processed/labels.csv  (must include a 'source' column: tox21 / chembl)

Outputs:
  - models/tuned_xgboost_model_with_chembl.pkl
  - models/tuned_xgboost_model_without_chembl.pkl
  - reports/chembl_ablation_with_chembl.json
  - reports/chembl_ablation_without_chembl.json

Usage:
  python src/compare_chembl_experiment.py
"""

import os

# Support BOTH execution styles:
#   - python src/compare_chembl_experiment.py
#   - python -m src.compare_chembl_experiment
try:
    # When executed as a module: python -m src.compare_chembl_experiment
    from .improve_model import prepare_data, tune_xgboost, evaluate_and_save  # type: ignore
    from .pipeline_utils import save_feature_pipeline_artifact  # type: ignore
except ImportError:
    # When executed as a script: python src/compare_chembl_experiment.py
    import sys

    SRC_DIR = os.path.dirname(__file__)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    from improve_model import prepare_data, tune_xgboost, evaluate_and_save  # type: ignore
    from pipeline_utils import save_feature_pipeline_artifact  # type: ignore


def run_one(include_chembl: bool, chembl_weight: float = 0.5, random_state: int = 42):
    tag = "with_chembl" if include_chembl else "without_chembl"
    print("\n" + "━" * 70)
    print(f"🧪 Experiment: {tag}")
    print("━" * 70)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        w_train,
        w_val,
        w_test,
        scaler,
        feature_names,
    ) = prepare_data(random_state=random_state, chembl_weight=chembl_weight, include_chembl=include_chembl)

    model = tune_xgboost(X_train, y_train, sample_weight=w_train, random_state=random_state)

    metrics = evaluate_and_save(
        model,
        X_test,
        y_test,
        scaler,
        feature_names,
        artifact_name=f"tuned_xgboost_model_{tag}.pkl",
        report_name=f"chembl_ablation_{tag}.json",
        extra_metadata={"include_chembl": bool(include_chembl), "chembl_weight": float(chembl_weight)},
        update_best_model=False,
    )
    return metrics


if __name__ == "__main__":
    # Ensure reports/ exists even if user hasn't run other scripts
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "reports"), exist_ok=True)

    # Default: conservative weighting
    m_with = run_one(include_chembl=True, chembl_weight=0.5)
    m_without = run_one(include_chembl=False, chembl_weight=0.5)

    # Pick the "best" model primarily by precision, then PR-AUC as tie-breaker.
    # This aligns with the "precision without corrupting label quality" goal.
    def score(m):
        return (m.get("precision", 0.0), m.get("pr_auc", 0.0))

    best_tag = "with_chembl" if score(m_with) >= score(m_without) else "without_chembl"
    print("\n" + "=" * 70)
    print(f"🏁 Selected best model: {best_tag} (by precision, then PR-AUC)")
    print("=" * 70)

    import joblib
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    best_src = os.path.join(models_dir, f"tuned_xgboost_model_{best_tag}.pkl")
    best_dst = os.path.join(models_dir, "best_model.pkl")
    tuned_alias = os.path.join(models_dir, "tuned_xgboost_model.pkl")
    artifact = joblib.load(best_src)
    joblib.dump(artifact, best_dst)
    joblib.dump(artifact, tuned_alias)
    pipeline_path = save_feature_pipeline_artifact(
        artifact["scaler"],
        artifact["feature_names"],
        filename="feature_pipeline.pkl",
        extra_metadata={"selected_model_tag": best_tag},
    )
    print(f"💾 Updated best model at {best_dst}")
    print(f"💾 Updated compatibility alias at {tuned_alias}")
    print(f"💾 Updated shared feature pipeline at {pipeline_path}")

    print("\n✅ Done. Compare `reports/chembl_ablation_*.json`.")
