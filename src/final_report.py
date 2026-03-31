"""
final_report.py – Generate a consolidated markdown report for the current pipeline.

This script pulls together:
  - dataset composition
  - baseline metrics
  - XGBoost metrics
  - ChEMBL ablation results
  - the currently selected production artifact
  - top SHAP features when available
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from .pipeline_utils import PROCESSED_DATA_DIR, REPORTS_DIR, load_model_artifact
except ImportError:
    from pipeline_utils import PROCESSED_DATA_DIR, REPORTS_DIR, load_model_artifact  # type: ignore


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_metric_block(metrics: dict | None) -> str:
    if not metrics:
        return "N/A"
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    parts = [f"{key}={metrics.get(key):.4f}" for key in keys if isinstance(metrics.get(key), (int, float))]
    return ", ".join(parts) if parts else str(metrics)


def build_report() -> str:
    labels = pd.read_csv(PROCESSED_DATA_DIR / "labels.csv")
    source_counts = labels["source"].value_counts().to_dict() if "source" in labels.columns else {}
    target_counts = labels["toxicity"].value_counts().to_dict() if "toxicity" in labels.columns else {}

    baseline = _load_json(REPORTS_DIR / "baseline_metrics.json")
    tuned = _load_json(REPORTS_DIR / "tuned_xgboost_metrics.json")
    with_chembl = _load_json(REPORTS_DIR / "chembl_ablation_with_chembl.json")
    without_chembl = _load_json(REPORTS_DIR / "chembl_ablation_without_chembl.json")
    top_features_path = REPORTS_DIR / "top_features.csv"
    top_features = pd.read_csv(top_features_path) if top_features_path.exists() else None
    artifact = load_model_artifact(prefer_best=True)

    lines = [
        "# ToxPredict Final Report",
        "",
        "## 1. Dataset Summary",
        f"- Total compounds in processed dataset: **{len(labels):,}**",
        f"- Source breakdown: **{source_counts}**",
        f"- Toxicity label breakdown: **{target_counts}**",
        "- Feature space: **217 RDKit descriptors + 1024 Morgan fingerprint bits = 1241 features**",
        "",
        "## 2. Model Comparison",
    ]

    if baseline:
        lines.extend(
            [
                f"- Baseline selection rule: **{baseline.get('selection_metric', 'N/A')}**",
                f"- Baseline toxic-call threshold: **{baseline.get('selection_threshold', 'N/A')}**",
                f"- Selected baseline: **{baseline.get('selected_model', 'N/A')}**",
                f"- Logistic Regression test metrics: **{baseline.get('test', {}).get('logistic_regression', {})}**",
                f"- Random Forest test metrics: **{baseline.get('test', {}).get('random_forest', {})}**",
            ]
        )
    else:
        lines.append("- Baseline comparison report not available yet. Run `python -m src.baseline_models`.")

    if tuned:
        lines.append(f"- Tuned XGBoost test metrics: **{_format_metric_block(tuned)}**")
    else:
        lines.append("- Tuned XGBoost metrics report not available yet. Run `python -m src.improve_model`.")

    lines.extend(["", "## 3. ChEMBL Ablation"])
    if with_chembl and without_chembl:
        better_tag = "with ChEMBL" if (
            with_chembl.get("precision", 0.0),
            with_chembl.get("pr_auc", 0.0),
        ) >= (
            without_chembl.get("precision", 0.0),
            without_chembl.get("pr_auc", 0.0),
        ) else "without ChEMBL"
        lines.extend(
            [
                f"- With ChEMBL: **{_format_metric_block(with_chembl)}**",
                f"- Without ChEMBL: **{_format_metric_block(without_chembl)}**",
                f"- Promotion rule outcome: **{better_tag}** currently wins by precision, with PR-AUC used as the tie-breaker.",
                "- Interpretation: ChEMBL remains an auxiliary signal. It should stay down-weighted and only be promoted when it genuinely improves held-out precision.",
            ]
        )
    else:
        lines.append("- ChEMBL ablation reports are not available yet. Run `python -m src.compare_chembl_experiment`.")

    lines.extend(["", "## 4. Production Artifact"])
    if artifact:
        lines.extend(
            [
                f"- Preferred production artifact: **{artifact.get('artifact_path')}**",
                f"- Display name: **{artifact.get('display_name')}**",
                f"- Feature count expected by runtime: **{len(artifact.get('feature_names', []))}**",
            ]
        )
    else:
        lines.append("- No production artifact available.")

    lines.extend(["", "## 5. Explainability"])
    if top_features is not None and not top_features.empty:
        lines.append("- Top SHAP features currently saved in `reports/top_features.csv`:")
        for _, row in top_features.head(10).iterrows():
            lines.append(
                f"  - Rank {int(row.iloc[0])}: **{row.iloc[1]}** ({row.iloc[2]})"
            )
    else:
        lines.append("- SHAP top features are not available yet. Run `python -m src.explainability` or `python -m src.shap_explain`.")

    lines.extend(
        [
            "",
            "## 6. Runtime Notes",
            "- The Streamlit app now prefers `models/best_model.pkl` and falls back only when needed.",
            "- The priority toxin dictionary is checked before ML feature generation.",
            "- Runtime predictions are triaged as SAFE, UNCERTAIN, or CRITICAL HAZARD so the app does not overstate certainty.",
            "- External hazard lists remain dictionary-only in this pass; they are not merged into training labels.",
        ]
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = build_report()
    report_path = REPORTS_DIR / "final_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved consolidated report to {report_path}")
