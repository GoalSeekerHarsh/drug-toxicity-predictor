# 🧪 ToxPredict: ML-Driven Drug Toxicity Screening

**ToxPredict** is an end-to-end Machine Learning pipeline and web application designed to predict the toxicity of chemical compounds at the early stages of drug discovery. By analyzing a compound's molecular structure, the system predicts the likelihood of toxicity and explicitly explains *why* using advanced SHAP (SHapley Additive exPlanations) visual analytics.

Built for the **Drug Toxicity Prediction Hackathon**.

---

## 0. Project Status / What's Included

| Item | Status |
|------|--------|
| Trained XGBoost model (`models/tuned_xgboost_model.pkl`) | ✅ Included in repo |
| Processed dataset (`data/processed/`) | ✅ Included (features + labels CSVs) |
| ZINC-250k virtual screening demo data (`data/processed/zinc_demo_sample.csv`) | ✅ Included |
| Evaluation reports & plots (`reports/`) | ✅ Generated and committed |
| Priority toxin dictionary (`data/priority_toxins.json`) | ✅ Included |
| Streamlit web app (`app/streamlit_app.py`) | ✅ Ready to run |
| EDA notebook (`notebooks/01_eda.ipynb`) | ✅ Included |
| Full training pipeline (re-train from scratch) | ✅ Supported (see §5) |
| Automated tests (`tests/test_demo.py`) | ✅ Included |

**Quick start:** The repo ships with a pre-trained model. You only need to install dependencies and launch the app — no re-training required.

> **Model file note:** `models/tuned_xgboost_model.pkl` is the model trained and committed to this repository. Running the ablation script (`python -m src.compare_chembl_experiment`) additionally generates `models/best_model.pkl`, which the app prefers at runtime. The app automatically falls back to `tuned_xgboost_model.pkl` when `best_model.pkl` is not present, so the app works out of the box.

---

## 1. Problem Statement
Drug development is a billion-dollar process that frequently fails in late stages due to unexpected compound toxicity. Identifying toxic properties computationally *before* physical synthesis or animal testing can drastically reduce R&D costs, accelerate discovery, and improve patient safety. Our goal is to predict chemical toxicity strictly from structural notations (SMILES).

## 2. Dataset
This project uses two complementary sources:
- **Tox21 (primary labels):** ~7,800 validated compounds. A molecule is labelled `1` if it is positive in **any** tested Tox21 assay, otherwise `0`.
- **ChEMBL withdrawn drugs (auxiliary labels):** a conservative set of withdrawn compounds added as extra toxic examples. These rows are kept **down-weighted** during training because they are useful but noisier than Tox21.

Current processed dataset snapshot:
- **7,853 compounds total** (label breakdown: 4,954 non-toxic, 2,899 toxic)
- **7,823 Tox21 rows**
- **30 ChEMBL auxiliary rows**

> **Dataset licensing:** Tox21 data is publicly available from the [NIH Tox21 Data Challenge](https://tripod.nih.gov/tox21/challenge/). ChEMBL data is available under the [CC BY-SA 3.0 license](https://chembl.gitbook.io/chembl-interface-documentation/about#data-licensing). The raw Tox21 CSV (`tox21.csv`) is **not** committed to this repository and must be placed at `data/raw/tox21.csv` before running the data-loading step. The processed outputs and model artifacts are committed for convenience.

## 3. The Pipeline
1. **Data Cleaning (`src/data_loader.py`)**: Validates raw SMILES strings using RDKit, removes unparseable molecules, canonicalizes structures, deduplicates them chemically, aggregates all Tox21 assays into one binary label, and merges the auxiliary ChEMBL supplement.
2. **Feature Engineering (`src/feature_engineering.py`)**: Converts SMILES strings into a dense 1,241-dimensional feature matrix:
   - *217 Molecular Descriptors* (Weight, LogP, TPSA, etc.)
   - *1024-bit Morgan Fingerprints* (ECFP4 standard to capture structural sub-graphs).
3. **Baseline Models (`src/baseline_models.py`)**: Trains Logistic Regression and Random Forest on the same stratified `70/15/15` split and the same continuous-only scaling contract used everywhere else.
4. **Model Tuning (`src/improve_model.py`)**: Trains the production **XGBoost Classifier** with Grid Search hyperparameter tuning (exploring `max_depth` ∈ {3, 5, 7} and `learning_rate` ∈ {0.01, 0.1}). ChEMBL rows are down-weighted by default (`0.5`) so Tox21 remains dominant. Saves `models/tuned_xgboost_model.pkl`.
5. **Ablation (`src/compare_chembl_experiment.py`)**: Compares XGBoost performance with vs. without ChEMBL supplement and promotes the better-precision run to `models/best_model.pkl`.
6. **Explainability (`src/shap_explain.py`, `src/explainability.py`)**: Uses SHAP TreeExplainer on the selected production artifact and the same feature scaling contract used at inference time. Outputs global summary plots and per-compound waterfall charts to `reports/`.
7. **ZINC-250k Virtual Screening (`src/zinc_loader.py`, `src/zinc_screen.py`)**: Screens 1,000 molecules from the ZINC-250k library against the trained model. Results are saved to `reports/zinc_screen_results.csv` and `reports/zinc_screen_summary.txt`, and are browsable via the Streamlit Batch Upload tab.
8. **Priority Toxin Bypass (`data/priority_toxins.json`)**: The Streamlit app checks a canonicalized toxin dictionary before ML inference. Exact matches return **CRITICAL HAZARD** and bypass the model entirely.
9. **Precision-First Verdicts**: Non-dictionary predictions are triaged into **SAFE**, **UNCERTAIN**, or **CRITICAL HAZARD** using conservative thresholds (safe ≤ 0.30, hazard ≥ 0.62) so the app avoids overconfident binary calls.
10. **Final Report (`src/final_report.py`)**: Aggregates all metrics and writes `reports/final_report.md`.

### Helper Scripts
- `scripts/build_toxin_dictionary.py` — Rebuilds `data/priority_toxins.json` from a curated list of known priority hazards.
- `scripts/fetch_chembl_withdrawn.py` — Fetches the latest withdrawn drug list from the ChEMBL REST API and saves it to `data/raw/`.

## 4. Installation & Setup

Ensure you have **Python 3.9+** installed. We highly recommend using a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/GoalSeekerHarsh/drug-toxicity-predictor.git
cd drug-toxicity-predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt
```

*(Note for macOS users: XGBoost requires OpenMP. If you encounter installation errors, run `brew install libomp` first).*

> **Dev Container:** A `.devcontainer/devcontainer.json` is included for VS Code Dev Containers / GitHub Codespaces. Opening the repo in a Codespace automatically installs all dependencies.

## 5. How to Run

### 5a. Launch the Web App (quickest path — no re-training needed)
The project ships with a pre-trained model. Simply install dependencies and start the app:

```bash
streamlit run app/streamlit_app.py
```

*Navigate to `http://localhost:8501`. Enter any valid SMILES string (e.g., `CCO`, `c1ccccc1`) to get a real-time toxicity verdict and SHAP explanation.*

### 5b. Re-train the Full Pipeline from Scratch
If you want to re-train with your own data, run the pipeline steps in order. The raw Tox21 CSV must be present at `data/raw/tox21.csv` first.

```bash
# Step 1 – Clean & merge data  (writes data/processed/features.csv + labels.csv)
python -m src.data_loader

# Step 2 – Feature engineering  (updates data/processed/features.csv)
python -m src.feature_engineering

# Step 3 – Baseline models  (writes reports/baseline_metrics.json + evaluation plots)
python -m src.baseline_models

# Step 4 – Train tuned XGBoost  (writes models/tuned_xgboost_model.pkl + reports/tuned_xgboost_metrics.json)
python -m src.improve_model

# Step 5 – ChEMBL ablation  (writes models/best_model.pkl + reports/chembl_ablation_*.json)
python -m src.compare_chembl_experiment

# Step 6 – SHAP explainability  (writes reports/shap_*.png + reports/top_features.csv)
python -m src.shap_explain

# Step 7 – Final report  (writes reports/final_report.md)
python -m src.final_report

# Step 8 – Launch app
streamlit run app/streamlit_app.py
```

### 5c. ZINC-250k Virtual Screening
To run the batch virtual screen against ZINC-250k demo molecules:

```bash
python -m src.zinc_screen
```

Results are written to `reports/zinc_screen_results.csv` and `reports/zinc_screen_summary.txt`. They are also available in the **Batch Upload** tab of the Streamlit app.

### 5d. Exploratory Data Analysis Notebook
An EDA notebook is available at `notebooks/01_eda.ipynb`. Launch it with:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 5e. Running Tests
```bash
pytest tests/
```

## 6. Key Configuration Options

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `SAFE_THRESHOLD` | `src/pipeline_utils.py` | `0.30` | Probability below which a compound is SAFE |
| `HAZARD_THRESHOLD` | `src/pipeline_utils.py` | `0.62` | Probability at or above which a compound is CRITICAL HAZARD |
| `chembl_weight` | `src/improve_model.py` | `0.5` | Sample weight applied to auxiliary ChEMBL rows during training |
| `train_ratio / val_ratio / test_ratio` | `src/improve_model.py` | `0.70 / 0.15 / 0.15` | Stratified dataset split ratios |

## 7. Expected Inputs & Outputs

**Inputs:**
- A valid **SMILES string** (e.g., `CCO` for ethanol, `c1ccccc1` for benzene).
- For batch screening: a CSV file with a `smiles` column.

**Outputs:**
- **Verdict:** `SAFE`, `UNCERTAIN`, or `CRITICAL HAZARD`
- **Toxicity probability** (0.0 – 1.0)
- **2D molecular structure** rendering
- **SHAP waterfall chart** explaining which features drove the prediction
- Batch results: CSV with per-molecule probability and verdict

## 8. Where Are the Trained Model & Artifacts?

| Artifact | Path | Description |
|----------|------|-------------|
| Pre-trained model | `models/tuned_xgboost_model.pkl` | XGBoost model committed to the repo |
| Production model (after ablation) | `models/best_model.pkl` | Generated by `python -m src.compare_chembl_experiment` |
| Baseline evaluation | `reports/baseline_metrics.json` | Logistic Regression & Random Forest metrics |
| XGBoost metrics | `reports/tuned_xgboost_metrics.json` | Tuned XGBoost evaluation on held-out test set |
| Ablation results | `reports/chembl_ablation_with_chembl.json` | XGBoost + ChEMBL metrics |
| Ablation results | `reports/chembl_ablation_without_chembl.json` | XGBoost Tox21-only metrics |
| SHAP plots | `reports/shap_*.png` | Global summary and local waterfall charts |
| Feature importance | `reports/feature_importance.png` | Top-N feature importances bar chart |
| ZINC screen results | `reports/zinc_screen_results.csv` | Per-molecule prediction for 1,000 ZINC molecules |
| Final report | `reports/final_report.md` | Auto-generated summary of all metrics |

## 9. Model Results

### Baseline Comparison (held-out test set, threshold = 0.62)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|----|
| Logistic Regression | 0.6524 | 0.4802 | — | — | — |
| Random Forest | 0.7936 | 0.7375 | 0.8155 | 0.3862 | 0.5242 |
| **Tuned XGBoost** | **0.7731** | **0.7119** | **0.9487** | 0.1701 | 0.2885 |

### ChEMBL Ablation (held-out test set)

| Configuration | ROC-AUC | PR-AUC | Precision |
|---------------|---------|--------|-----------|
| XGBoost + ChEMBL (promoted) | 0.7731 | 0.7119 | **0.9487** |
| XGBoost Tox21-only | 0.7804 | 0.6979 | 0.9057 |

**Interpretation:**
- Precision is the primary promotion metric (minimising false CRITICAL HAZARD alarms matters most in drug screening).
- The ChEMBL-augmented run wins on precision (`0.9487` vs `0.9057`) and is therefore promoted to `models/best_model.pkl`.
- The runtime verdict is intentionally tri-state so uncertain molecules are not forced into a fake binary answer.

## 10. Feature Importance Summary
Using SHAP TreeExplainer, the top structural drivers of toxicity are:

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | MolLogP | 0.1692 |
| 2 | BertzCT | 0.1349 |
| 3 | BCUT2D_LOGPHI | 0.0852 |
| 4 | HeavyAtomMolWt | 0.0677 |
| 5 | FractionCSP3 | 0.0632 |

Morgan fingerprint bits (structural sub-graph patterns) also appear strongly in the top-50 list. The full ranked list is written to `reports/top_features.csv` after running `python -m src.shap_explain`.

## 11. ZINC-250k Virtual Screening Results
The trained model was applied to 1,000 molecules from the ZINC-250k drug-like library (compounds the model has never seen during training):

- **Molecules screened:** 1,000
- **Predicted TOXIC (CRITICAL HAZARD):** 32 (3.2%)
- **Predicted SAFE:** 968 (96.8%)

Full results are in `reports/zinc_screen_results.csv`. The highest-risk ZINC molecules (top toxicity probability > 0.70) include several steroidal structures, consistent with known hepatotoxic steroid scaffolds.

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: rdkit` | RDKit not installed or wrong Python env | `pip install rdkit` (Python ≥ 3.9 required) |
| `XGBoost install fails` on macOS | Missing OpenMP | `brew install libomp && pip install xgboost` |
| App shows "model not found" | Neither `best_model.pkl` nor `tuned_xgboost_model.pkl` present | Ensure `models/tuned_xgboost_model.pkl` is present (committed to repo), or re-run `python -m src.improve_model` |
| `FileNotFoundError: data/processed/features.csv` | Processed data missing | Run `python -m src.data_loader && python -m src.feature_engineering` |
| `FileNotFoundError: data/raw/tox21.csv` | Raw dataset not placed | Download `tox21.csv` from the NIH Tox21 challenge and place it at `data/raw/tox21.csv` |
| SHAP plots not generated | Explainability scripts not run | Run `python -m src.shap_explain` |
| Tests fail with `best_model.pkl not found` | Ablation step not yet run | Run `python -m src.compare_chembl_experiment` to generate `models/best_model.pkl` |

## 13. Future Work
- **Graph Neural Networks (GNNs):** Implementing Chemprop or DGL to learn directly from the molecular graph structure rather than relying entirely on pre-computed RDKit descriptors.
- **Multi-task Learning:** Predicting the 12 Tox21 endpoints as separate but related tasks instead of using the current aggregated "any-assay-positive" label.
- **Cloud Deployment:** Containerizing the Streamlit application using Docker and deploying it to AWS Fargate or Google Cloud Run for public access.
