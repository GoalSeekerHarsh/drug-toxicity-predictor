# ToxPredict Final Report

## 1. Dataset Summary
- Total compounds in processed dataset: **7,855**
- Source breakdown: **{'tox21': 7823, 'chembl': 32}**
- Toxicity label breakdown: **{0: 4954, 1: 2901}**
- Feature space: **217 RDKit descriptors + 1024 Morgan fingerprint bits = 1241 features**

## 2. Model Comparison
- Baseline selection rule: **precision_then_pr_auc**
- Selected baseline: **Random Forest**
- Logistic Regression test metrics: **{'roc_auc': 0.5, 'pr_auc': 0.36895674300254455, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'confusion_matrix': [[744, 0], [435, 0]]}**
- Random Forest test metrics: **{'roc_auc': 0.8000108144852305, 'pr_auc': 0.734156349657151, 'precision': 0.7677419354838709, 'recall': 0.5471264367816092, 'f1': 0.6389261744966444, 'confusion_matrix': [[672, 72], [197, 238]]}**
- Tuned XGBoost metrics: **{'roc_auc': 0.7816076504758374, 'pr_auc': 0.7006619730335109, 'precision': 0.7264957264957265, 'recall': 0.39080459770114945, 'f1': 0.5082212257100149, 'confusion_matrix': [[680, 64], [265, 170]], 'include_chembl': True}**

## 3. ChEMBL Ablation
- With ChEMBL: **{'roc_auc': 0.7816076504758374, 'pr_auc': 0.7006619730335109, 'precision': 0.7264957264957265, 'recall': 0.39080459770114945, 'f1': 0.5082212257100149, 'confusion_matrix': [[680, 64], [265, 170]], 'include_chembl': True, 'chembl_weight': 0.5}**
- Without ChEMBL: **{'roc_auc': 0.7191185792844585, 'pr_auc': 0.5968975225669053, 'precision': 0.71, 'recall': 0.16473317865429235, 'f1': 0.2674199623352166, 'confusion_matrix': [[714, 29], [360, 71]], 'include_chembl': False, 'chembl_weight': 0.5}**
- Interpretation: ChEMBL remains an auxiliary signal, and the weighted version should outperform the Tox21-only run on precision/PR-AUC before it is promoted.

## 4. Production Artifact
- Preferred production artifact: **/Users/harshgupta/Documents/GoalSeeker/Quant/toxicity-project/models/best_model.pkl**
- Display name: **Tuned XGBoost**
- Feature count expected by runtime: **1241**

## 5. Explainability
- Top SHAP features currently saved in `reports/top_features.csv`:
  - Rank 1: **MolLogP** (0.19356017)
  - Rank 2: **BertzCT** (0.13829164)
  - Rank 3: **SlogP_VSA2** (0.06806316)
  - Rank 4: **FractionCSP3** (0.05817404)
  - Rank 5: **HeavyAtomMolWt** (0.0567047)
  - Rank 6: **SMR_VSA10** (0.046551805)
  - Rank 7: **BCUT2D_LOGPHI** (0.031443633)
  - Rank 8: **VSA_EState3** (0.025616497)
  - Rank 9: **SMR_VSA3** (0.021791995)
  - Rank 10: **fr_phenol_noOrthoHbond** (0.019901654)

## 6. Runtime Notes
- The Streamlit app now prefers `models/best_model.pkl` and falls back only when needed.
- The priority toxin dictionary is checked before ML feature generation.
- Runtime predictions are triaged as SAFE, UNCERTAIN, or CRITICAL HAZARD so the app does not overstate certainty.
- External hazard lists remain dictionary-only in this pass; they are not merged into training labels.
