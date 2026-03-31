# 🧪 ToxPredict: ML-Driven Drug Toxicity Screening

**ToxPredict** is an end-to-end Machine Learning pipeline and web application designed to predict the toxicity of chemical compounds at the early stages of drug discovery. By analyzing a compound's molecular structure, the system predicts the likelihood of toxicity and explicitly explains *why* using advanced SHAP (SHapley Additive exPlanations) visual analytics.

Built for the **Drug Toxicity Prediction Hackathon**.

---

## 1. Problem Statement
Drug development is a billion-dollar process that frequently fails in late stages due to unexpected compound toxicity. Identifying toxic properties computationally *before* physical synthesis or animal testing can drastically reduce R&D costs, accelerate discovery, and improve patient safety. Our goal is to predict chemical toxicity strictly from structural notations (SMILES).

## 2. Dataset
This project uses two complementary sources:
- **Tox21 (primary labels):** ~7,800 validated compounds. A molecule is labelled `1` if it is positive in **any** tested Tox21 assay, otherwise `0`.
- **ChEMBL withdrawn drugs (auxiliary labels):** a conservative set of withdrawn compounds added as extra toxic examples. These rows are kept **down-weighted** during training because they are useful but noisier than Tox21.

Current processed dataset snapshot:
- **7,853 compounds total**
- **7,823 Tox21 rows**
- **30 ChEMBL auxiliary rows**

## 3. The Pipeline
1. **Data Cleaning (`src/data_loader.py`)**: Validates raw SMILES strings using RDKit, removes unparseable molecules, canonicalizes structures, deduplicates them chemically, aggregates all Tox21 assays into one binary label, and merges the auxiliary ChEMBL supplement.
2. **Feature Engineering (`src/feature_engineering.py`)**: Converts SMILES strings into a dense 1,241-dimensional feature matrix:
   - *217 Molecular Descriptors* (Weight, LogP, TPSA, etc.)
   - *1024-bit Morgan Fingerprints* (ECFP4 standard to capture structural sub-graphs).
3. **Baseline Models (`src/baseline_models.py`)**: Trains Logistic Regression and Random Forest on the same stratified `70/15/15` split and the same continuous-only scaling contract used everywhere else.
4. **Model Tuning (`src/improve_model.py`)**: Trains the production **XGBoost Classifier** with simple precision-first tuning. ChEMBL rows are down-weighted by default (`0.5`) so Tox21 remains dominant.
5. **Ablation (`src/compare_chembl_experiment.py`)**: Compares XGBoost performance with vs. without ChEMBL and promotes the better run to `models/best_model.pkl`.
6. **Explainability (`src/shap_explain.py`, `src/explainability.py`)**: Uses SHAP on the selected production artifact and the exact same feature scaling contract used at inference time.
7. **Priority Toxin Bypass (`data/priority_toxins.json`)**: The Streamlit app checks a canonicalized toxin dictionary before ML inference. Exact matches return **CRITICAL HAZARD** and bypass the model entirely.
8. **Precision-First Verdicts**: Non-dictionary predictions are triaged into **SAFE**, **UNCERTAIN**, or **CRITICAL HAZARD** using conservative thresholds so the app avoids overconfident binary calls.

## 4. Installation & Setup

Ensure you have **Python 3.9+** installed. We highly recommend using a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/toxicity-project.git
cd toxicity-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

*(Note for macOS users: XGBoost requires OpenMP. If you encounter installation errors, run `brew install libomp` first).*

## 5. How to Run the App
The project includes a production-ready, highly defensive **Streamlit Web Application** designed for non-technical users and domain experts.

To launch the app locally:
```bash
streamlit run app/streamlit_app.py
```
*Navigate to `http://localhost:8501`. Enter any valid SMILES string (e.g., `CCO`, `c1ccccc1`) to see real-time toxicity screening and SHAP breakdown plots.*

Recommended local workflow:
```bash
python -m src.data_loader
python -m src.feature_engineering
python -m src.baseline_models
python -m src.improve_model
python -m src.compare_chembl_experiment
python -m src.final_report
streamlit run app/streamlit_app.py
```

## 6. Model Results
Current ChEMBL ablation results on the held-out test split:

- **With ChEMBL:** ROC-AUC `0.7731`, PR-AUC `0.7119`, Precision `0.9487`
- **Without ChEMBL:** ROC-AUC `0.7804`, PR-AUC `0.6979`, Precision `0.9057`

Interpretation:
- The ChEMBL supplement remains an auxiliary signal and stays down-weighted.
- Promotion now uses the same hazard threshold as the production app, so the reported precision matches live `CRITICAL HAZARD` behavior.
- On the current rebuilt dataset snapshot, the weighted ChEMBL run is the promoted production model because it delivers the highest held-out hazard precision.
- Precision remains the primary promotion metric, with PR-AUC as the tie-breaker.
- The production app now prefers `models/best_model.pkl`, which is updated by the ablation workflow.
- The runtime verdict is intentionally tri-state so uncertain molecules do not get forced into a fake binary answer.

## 7. Feature Importance Summary
Using SHAP analysis, we identified the top structural drivers indicating toxicity:
1. Descriptor-driven global effects are still useful for interpretation.
2. Morgan fingerprint bits capture many of the strongest toxic substructure signals.
3. The most current top-feature list is written to `reports/top_features.csv` after running explainability.

## 8. Future Work
If given more time past this hackathon, we would explore:
- **Graph Neural Networks (GNNs):** Implementing Chemprop or DGL to learn directly from the molecular graph structure rather than relying entirely on pre-computed RDKit descriptors.
- **Multi-task Learning:** Predicting the 12 Tox21 endpoints as separate but related tasks instead of using the current aggregated "any-assay-positive" label.
- **Cloud Deployment:** Containerizing the Streamlit application using Docker and deploying it to AWS Fargate or Google Cloud Run for public access.
