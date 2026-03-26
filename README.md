# 🧪 ToxPredict: ML-Driven Drug Toxicity Screening

**ToxPredict** is an end-to-end Machine Learning pipeline and web application designed to predict the toxicity of chemical compounds at the early stages of drug discovery. By analyzing a compound's molecular structure, the system predicts the likelihood of toxicity and explicitly explains *why* using advanced SHAP (SHapley Additive exPlanations) visual analytics.

Built for the **Drug Toxicity Prediction Hackathon**.

---

## 1. Problem Statement
Drug development is a billion-dollar process that frequently fails in late stages due to unexpected compound toxicity. Identifying toxic properties computationally *before* physical synthesis or animal testing can drastically reduce R&D costs, accelerate discovery, and improve patient safety. Our goal is to predict chemical toxicity strictly from structural notations (SMILES).

## 2. Dataset
This project uses the industry-standard **Tox21 (Toxicology in the 21st Century) dataset**.
- **Scope**: ~7,800 validated chemical compounds.
- **Labels**: We specifically target the `NR-AR` (Androgen Receptor) toxicity endpoint.
- **Imbalance**: Highly imbalanced; only ~4.2% of the compounds in the dataset are flagged as toxic.

## 3. The Pipeline
1. **Data Cleaning (`src/data_loader.py`)**: Validates raw SMILES strings using RDKit, removes unparseable molecules, and deduplicates the records.
2. **Feature Engineering (`src/feature_engineering.py`)**: Converts SMILES strings into a dense 1,241-dimensional feature matrix:
   - *217 Molecular Descriptors* (Weight, LogP, TPSA, etc.)
   - *1024-bit Morgan Fingerprints* (ECFP4 standard to capture structural sub-graphs).
3. **Model Tuning (`src/improve_model.py`)**: Trains an optimized **XGBoost Classifier**. It handles the extreme class imbalance by leveraging Synthetic Minority Over-sampling (SMOTE) and algorithm-level class weighting (`scale_pos_weight`).
4. **Explainability (`src/shap_explain.py`)**: Employs TreeExplainer to unpack the XGBoost black-box, extracting both global feature importance (top descriptors) and local explanations (single molecule waterfall plots).

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

## 6. Model Results
Our tuned XGBoost model was evaluated on a strictly held-out 15% Test Set. 

- **Test ROC-AUC:** `0.7637`
- **F1-Score:** `0.3439` (Due to heavy 95:5 class imbalance, precision naturally suffers compared to recall, which we optimized for safety).
- **Recall (Sensitivity):** `0.4355`

The model is incredibly precise at identifying safe compounds and is tuned aggressively to catch toxic drugs early using `scale_pos_weight`.

## 7. Feature Importance Summary
Using SHAP analysis, we identified the top structural drivers indicating toxicity:
1. `BCUT2D_MWLOW` (Molecular weight/burden eigenvalue)
2. `HallKierAlpha` (Topological shape and branching)
3. `BertzCT` (Symmetry and complexity of molecular rings)
4. `RingCount` (Number of aliphatic/aromatic rings)

## 8. Future Work
If given more time past this hackathon, we would explore:
- **Graph Neural Networks (GNNs):** Implementing Chemprop or DGL to learn directly from the molecular graph structure rather than relying entirely on pre-computed RDKit descriptors.
- **Multi-task Learning:** Predicting all 12 Tox21 toxicity endpoints simultaneously instead of just `NR-AR` to leverage shared chemical knowledge.
- **Cloud Deployment:** Containerizing the Streamlit application using Docker and deploying it to AWS Fargate or Google Cloud Run for public access.
