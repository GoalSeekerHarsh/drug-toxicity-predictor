# ToxPredict Final Report

## 1. Dataset Summary
- Total compounds in processed dataset: **7,853**
- Source breakdown: **{'tox21': 7823, 'chembl': 30}**
- Toxicity label breakdown: **{0: 4954, 1: 2899}**
- Feature space: **217 RDKit descriptors + 1024 Morgan fingerprint bits = 1241 features**

## 2. Model Comparison
- Baseline selection rule: **precision_then_pr_auc**
- Baseline toxic-call threshold: **0.62**
- Selected baseline: **Random Forest**
- Logistic Regression test metrics: **{'roc_auc': 0.6524280255565353, 'pr_auc': 0.4802141707939166, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'confusion_matrix': [[743, 0], [435, 0]], 'decision_threshold': 0.62}**
- Random Forest test metrics: **{'roc_auc': 0.793635618260856, 'pr_auc': 0.737476432399779, 'precision': 0.8155339805825242, 'recall': 0.38620689655172413, 'f1': 0.5241809672386896, 'confusion_matrix': [[705, 38], [267, 168]], 'decision_threshold': 0.62}**
- Tuned XGBoost test metrics: **roc_auc=0.7731, pr_auc=0.7119, precision=0.9487, recall=0.1701, f1=0.2885**

## 3. ChEMBL Ablation
- With ChEMBL: **roc_auc=0.7731, pr_auc=0.7119, precision=0.9487, recall=0.1701, f1=0.2885**
- Without ChEMBL: **roc_auc=0.7804, pr_auc=0.6979, precision=0.9057, recall=0.1114, f1=0.1983**
- Promotion rule outcome: **with ChEMBL** currently wins by precision, with PR-AUC used as the tie-breaker.
- Interpretation: ChEMBL remains an auxiliary signal. It should stay down-weighted and only be promoted when it genuinely improves held-out precision.

## 4. Production Artifact
- Preferred production artifact: **/Users/harshgupta/Documents/GoalSeeker/Quant/toxicity-project/models/best_model.pkl**
- Display name: **Tuned XGBoost**
- Feature count expected by runtime: **1241**

## 5. Explainability
- Top SHAP features currently saved in `reports/top_features.csv`:
  - Rank 1: **MolLogP** (0.16920298)
  - Rank 2: **BertzCT** (0.13488045)
  - Rank 3: **BCUT2D_LOGPHI** (0.08520752)
  - Rank 4: **HeavyAtomMolWt** (0.067650996)
  - Rank 5: **FractionCSP3** (0.06317009)
  - Rank 6: **SlogP_VSA2** (0.056536667)
  - Rank 7: **SlogP_VSA6** (0.03789076)
  - Rank 8: **SMR_VSA6** (0.031319994)
  - Rank 9: **SMR_VSA10** (0.029982258)
  - Rank 10: **fr_phenol_noOrthoHbond** (0.028632939)

## 6. Runtime Notes
- The Streamlit app now prefers `models/best_model.pkl` and falls back only when needed.
- The priority toxin dictionary is checked before ML feature generation.
- Runtime predictions are triaged as SAFE, UNCERTAIN, or CRITICAL HAZARD so the app does not overstate certainty.
- External hazard lists remain dictionary-only in this pass; they are not merged into training labels.
