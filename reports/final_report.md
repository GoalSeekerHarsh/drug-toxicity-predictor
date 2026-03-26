# 🧪 Drug Toxicity Prediction – Final Report

## 1. Project Overview
This project builds a **binary classification model** to predict chemical toxicity using the **Tox21 dataset**. We framed the problem by predicting the `NR-AR` (Androgen Receptor) endpoint.

**Key Achievements:**
- Cleaned and validated 7,831 compounds using RDKit.
- Engineered a combined feature space of **1,241 molecular features** per compound (217 RDKit Descriptors + 1024-bit Morgan Fingerprints).
- Trained and tuned an **XGBoost Classifier** with SMOTE to handle class imbalance (only ~4.2% of compounds are toxic).
- Developed a **Streamlit Web Interface** for real-time predictions.

---

## 2. Methodology

### Data Pipeline
1. **Raw Data:** Tox21 CSV (`tox21.csv`)
2. **Cleaning (`src/data_loader.py`):** 
   - Filtered out missing/empty SMILES.
   - Validated SMILES strings using RDKit (dropped 8 unparseable compounds).
   - Removed duplicate compounds.
   - Extracted `NR-AR` endpoint: 7,258 valid compounds.
3. **Feature Engineering (`src/feature_engineering.py`):**
   - **Molecular Descriptors:** 217 numeric properties (Weight, LogP, TPSA, etc.)
   - **Morgan Fingerprints:** 1024-bit ECFP4 vectors capturing local structural motifs (radius=2).

### Model Training (`src/model.py`)
- **Split:** 70% Train / 15% Validation / 15% Test (Stratified to preserve the 4.2% toxicity ratio).
- **Preprocessing:** `StandardScaler` fitted on training data; `SMOTE` applied only to training data to synthesize toxic examples.
- **Algorithms Tested:** RandomForest vs. XGBoost. 
- **Selection:** XGBoost won on the validation set.

---

## 3. Results

### Final Model Performance (XGBoost)
Evaluated on the **unseen 15% Test Set** (1,089 compounds):

- **Test ROC-AUC:** `0.6913`
- **Test F1-Score:** `0.5143`

|               | Precision | Recall (Sensitivity) | F1-Score | Support |
|---------------|-----------|----------------------|----------|---------|
| **Non-toxic** | 0.97      | 0.99                 | 0.98     | 1,043   |
| **Toxic**     | 0.75      | 0.39                 | 0.51     | 46      |

*Interpretation: The model is highly accurate at identifying safe compounds. When it predicts a compound is toxic, it is correct 75% of the time (high precision), though it misses some toxic compounds (39% recall).*

---

## 4. Feature Importance (SHAP Analysis)

We used SHAP (SHapley Additive exPlanations) to interpret the XGBoost model natively (`src/explainability.py`).

**Top 5 Structural Drivers of Toxicity:**
1. `FP_519` (Morgan Fingerprint Bit 519) - Strongest individual structural motif linked to toxicity.
2. `NumAliphaticCarbocycles` - Number of aliphatic rings; strongly influences drug lipophilicity.
3. `FP_875` (Morgan Fingerprint Bit 875)
4. `FP_420` (Morgan Fingerprint Bit 420)
5. `EState_VSA5` - Electrotopological state metric related to molecular surface area and polarity.

*See `reports/feature_importance.png` and `reports/shap_summary.png` for visual breakdowns.*

---

## 5. Deliverables Checklist

- [x] **GitHub Repository:** Clean structure with `src/`, `notebooks/`, `models/`, and `app/`.
- [x] **ML Model:** XGBoost saved as `models/best_model.pkl`.
- [x] **Feature Importance:** SHAP plots generated in `reports/`.
- [x] **Visualizations:** EDA notebook (`notebooks/01_eda.ipynb`) containing class distributions, missing value heatmaps, and descriptor correlation analysis.
- [x] **Prediction Tool:** `app/streamlit_app.py` built for live evaluation.
