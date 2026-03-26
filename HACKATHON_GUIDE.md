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
toxicity-project/
├── data/
│   ├── raw/                 # Original tox21.csv
│   └── processed/           # Cleaned features and labels
├── src/
│   ├── simple_load.py       # Basic dataset inspection
│   ├── data_loader.py       # Cleans missing/invalid SMILES
│   ├── feature_engineering.py # Extracts RDKit 2D + Fingerprints
│   ├── improve_model.py     # Trains & tunes XGBoost
│   ├── evaluate_model.py    # Metric calculation (ROC/AUC/F1)
│   └── shap_explain.py      # Generates explainability plots
├── models/                  # Saved .pkl files
├── reports/                 # Saved PNG evaluation and SHAP plots
├── app/
│   └── streamlit_app.py     # Frontend web application
├── README.md                # Submission documentation
└── requirements.txt         # Python dependencies
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

- [ ] **Code Runs End-to-End:** Ensure a judge could run `pip install -r requirements.txt` followed by `streamlit run app/streamlit_app.py` without fatal crashes.
- [ ] **Dependencies Match:** Check if `requirements.txt` includes exactly: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `rdkit`, `shap`, `streamlit`, `matplotlib`, `seaborn`.
- [ ] **No Hardcoded Absolute Paths:** Ensure all data loading in `src/` uses `os.path.join(os.path.dirname(__file__), ...)` so it works on the judge's computer, not just your Mac. *(We have already verified this).*
- [ ] **Data Leakage Check:** Ensure that `StandardScaler` in `improve_model.py` is only `fit()` on the training set, not the entire dataset. *(Verified: `fit_transform` on train, `transform` on test).*
- [ ] **App Error Handling:** Test the Streamlit app with an invalid SMILES string (e.g., "invalid_chemical") to ensure it shows a friendly error card instead of a raw Python traceback. *(Verified).*
- [ ] **Live Demo Rehearsal:** Have 2 SMILES ready in your clipboard. One definitively SAFE (`CCO` - Ethanol), and one definitively TOXIC (`O=C(O)CCC(=O)c1ccc(-c2ccccc2)cc1` - sample).
- [ ] **README Polish:** Ensure your team's names are on the `README.md`.

**Good luck! You've built a technically rigorous, highly interpretable pipeline.**
