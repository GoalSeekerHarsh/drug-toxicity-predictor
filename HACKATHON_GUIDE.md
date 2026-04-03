# 🏆 ToxPredict: Hackathon Winning Playbook

This document serves as your 3-Day Execution Plan, Architecture Overview, and Final Submission Checklist for the Drug Toxicity Prediction Hackathon.

---

## 🏗️ Simplest Winning Architecture

To win a hackathon, your architecture needs to be simple to explain, robust to demo, and technologically sound.

1. **The Data Layer:** Raw SMILES strings → RDKit Validation + Deduplication.
2. **The Feature Engineering Layer:** RDKit (MolWt, LogP, TPSA, etc.) + 1024-bit Morgan Fingerprints. This hybrid approach captures both physical properties and sub-graph structures.
3. **The ML Layer:** XGBoost Classifier. It inherently handles missing values, non-linear relationships, and most importantly, extreme class imbalance (95:5) via `scale_pos_weight`.
4. **The Interpretation Layer:** SHAP (TreeExplainer) to break open the black box and provide mathematical proof of *why* a compound is predicted toxic.
5. **The Presentation Layer:** A responsive, error-proof Streamlit web app that runs locally and handles bad inputs gracefully without crashing during your live demo.

### 📂 Folder Structure
Keep it industry-standard:
```text
drug-toxicity-predictor/
├── app/
│   └── streamlit_app.py              # Frontend web application
├── data/
│   ├── raw/                          # Original tox21.csv (excluded from git)
│   ├── processed/
│   │   ├── zinc_demo_sample.csv      # Demo ZINC molecules for batch tab
│   │   ├── features.npy              # Computed features (generated)
│   │   ├── labels.csv                # Cleaned labels (generated)
│   │   └── scaler.pkl                # Baseline scaler (generated)
│   └── priority_toxins.json          # 76 high-priority toxin dictionary
├── models/
│   ├── tuned_xgboost_model.pkl       # ✅ Pre-trained XGBoost (included)
│   └── best_model.pkl                # Promoted by ablation study (generated)
├── notebooks/
│   └── 01_eda.ipynb                  # Exploratory data analysis
├── reports/
│   ├── final_report.md               # Full technical write-up
│   ├── model_metrics.json            # Baseline metrics
│   ├── tuned_xgboost_metrics.json    # XGBoost metrics
│   ├── zinc_screen_results.csv       # ZINC screening output
│   ├── feature_importance.png        # Feature importance bar chart
│   ├── shap_global_summary.png       # SHAP beeswarm plot
│   ├── shap_local_waterfall.png      # SHAP waterfall for single molecule
│   └── ...                           # Other evaluation plots & JSON files
├── scripts/
│   ├── fetch_chembl_withdrawn.py     # Fetches withdrawn drugs from ChEMBL API
│   └── build_toxin_dictionary.py     # Builds priority_toxins.json
├── src/
│   ├── simple_load.py                # Basic dataset inspection
│   ├── data_loader.py                # Cleans missing/invalid SMILES + ChEMBL merge
│   ├── feature_engineering.py        # Extracts RDKit 2D descriptors + fingerprints
│   ├── baseline_models.py            # Logistic Regression & Random Forest baselines
│   ├── improve_model.py              # Trains & tunes XGBoost (saves tuned_xgboost_model.pkl)
│   ├── compare_chembl_experiment.py  # Ablation study; promotes best_model.pkl
│   ├── evaluate_model.py             # Metric calculation (ROC/AUC/F1/Precision)
│   ├── shap_explain.py               # Generates global & local SHAP explainability plots
│   ├── explainability.py             # Additional SHAP helpers
│   ├── zinc_screen.py                # Virtual screening on 1,000 ZINC molecules
│   ├── zinc_loader.py                # Loads ZINC-250k molecules
│   ├── zinc_baseline.py              # Builds chemical-space baseline scaler
│   ├── pipeline_utils.py             # Shared helpers (model load, metrics, split)
│   ├── final_report.py               # Generates final HTML/PDF report
│   └── inspect_data.py               # Data exploration utilities
├── tests/
│   └── test_demo.py                  # Smoke tests for core pipeline
├── FOR_BEGINNERS.md                  # Plain-language guide to every module
├── HACKATHON_GUIDE.md                # This file
├── DEMO_SCRIPT.md                    # 1-minute pitch & live demo flow
├── README.md                         # Main project documentation
└── requirements.txt                  # Python dependencies
```

---

## 📅 3-Day Execution Plan (Retrospective)

*If judges ask how you built this, here is your 3-day timeline:*

**Day 1: Data Mastery & Baselines**
*   Explored the Tox21 dataset, identified SMILES and the NR-AR endpoint.
*   Wrote `data_loader.py` to aggressively clean invalid molecules using RDKit.
*   Wrote `baseline_models.py` (Logistic Regression vs Random Forest) to establish a performance floor.

**Day 2: Advanced ML & Explainability**
*   Developed `feature_engineering.py` to extract 1,241 features (descriptors + ECFP4 fingerprints).
*   Upgraded to XGBoost (`improve_model.py`), running GridSearchCV to optimize `max_depth` and `learning_rate` while fiercely penalizing false negatives using `scale_pos_weight`.
*   Integrated SHAP (`shap_explain.py`) to extract the exact structural features driving toxicity.

**Day 3: Productization & Polish**
*   Built `streamlit_app.py`, bridging the raw XGBoost model into a user-friendly frontend.
*   Implemented extreme defensive programming (try-except blocks) so the app survives invalid judge inputs.
*   Polished `README.md` and visualizations for the final pitch.

---

## ✅ Final Verification Checklist (Pre-Submission)

Before finalizing your submission, double-check these critical items:

- [x] **Pre-trained Model Included:** `models/tuned_xgboost_model.pkl` is committed and the app loads it without any retraining step.
- [x] **Code Runs End-to-End:** A judge can run `pip install -r requirements.txt` followed by `streamlit run app/streamlit_app.py` without fatal crashes.
- [x] **Dependencies Match:** `requirements.txt` includes: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `rdkit`, `shap`, `streamlit`, `matplotlib`, `seaborn`, `imbalanced-learn`, `joblib`, `tqdm`, `requests`.
- [x] **No Hardcoded Absolute Paths:** All data loading in `src/` uses `os.path.join(os.path.dirname(__file__), ...)` so it works on any machine.
- [x] **Data Leakage Check:** `StandardScaler` in `improve_model.py` is only `fit()` on the training set, not the entire dataset. *(Verified: `fit_transform` on train, `transform` on test).*
- [x] **App Error Handling:** The Streamlit app handles an invalid SMILES string (e.g., `INVALID_CHEMICAL_xyz123`) with a friendly error card instead of a raw Python traceback. *(Verified).*
- [x] **Priority Toxin Dictionary:** `data/priority_toxins.json` (76 entries) is committed and loaded by the app.
- [x] **ZINC Screening Results:** `reports/zinc_screen_results.csv` is committed for the Batch Upload demo tab.
- [x] **SHAP Reports:** Explainability plots saved to `reports/` (feature_importance.png, shap_global_summary.png, shap_local_waterfall.png).
- [ ] **Live Demo Rehearsal:** Have 2 SMILES ready in your clipboard. One definitively SAFE (`CCO` - Ethanol), and one definitively TOXIC (`O=C(O)CCC(=O)c1ccc(-c2ccccc2)cc1` - sample).
- [ ] **README Polish:** Ensure your team's names are on the `README.md`.

**Good luck! You've built a technically rigorous, highly interpretable pipeline.**
