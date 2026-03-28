"""
streamlit_app.py – Drug Toxicity Prediction Interface (Hackathon Demo Version)

A clean, presentation-ready frontend built with Streamlit.
Designed to be easy to understand for judges and non-technical stakeholders.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Connect to src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from feature_engineering import smiles_to_mol, compute_descriptors, compute_morgan_fingerprint
except ImportError:
    st.error("❌ Critical Error: Could not find `src/feature_engineering.py`. Run from project root.")
    st.stop()

# Graceful Degradation Imports
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ── Configuration ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "tuned_xgboost_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_best_model.pkl")

@st.cache_resource
def load_model_file():
    """Load the trained model artifact safely."""
    if os.path.exists(TUNED_MODEL_PATH):
        return joblib.load(TUNED_MODEL_PATH), "Tuned XGBoost"
    elif os.path.exists(BASELINE_MODEL_PATH):
        return joblib.load(BASELINE_MODEL_PATH), "Baseline Model"
    else:
        return None, None

def predict_and_explain(smiles, artifact):
    """Safely compute features, predict, and generate SHAP values."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None, None, None, None, "Invalid SMILES string. RDKit could not parse it.", None
    
    try:
        desc = compute_descriptors(mol)
        fp = compute_morgan_fingerprint(mol, radius=2, n_bits=1024)
        if desc is None or fp is None:
            return None, None, None, None, "Failed to compute chemical features.", None
    except Exception as e:
        return None, None, None, None, f"Feature extraction error: {str(e)}", None
    
    feature_names = artifact["feature_names"]
    
    # Safely build feature vector matching EXACTLY what the model expects
    feature_vector = []
    for name in feature_names:
        if name.startswith("FP_"):
            try:
                bit_idx = int(name.split("_")[1])
                feature_vector.append(fp[bit_idx] if bit_idx < len(fp) else 0.0)
            except (ValueError, IndexError):
                feature_vector.append(0.0)
        else:
            feature_vector.append(desc.get(name, 0.0))
            
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_vector = np.nan_to_num(feature_vector, nan=0, posinf=0, neginf=0)
    
    try:
        feature_vector_scaled = artifact["scaler"].transform(feature_vector)
    except Exception as e:
        return None, None, None, None, f"Data scaling error: {str(e)}", None
    
    try:
        model = artifact["model"]
        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0]
    except Exception as e:
        return None, None, None, None, f"Model prediction error: {str(e)}", None

    metadata = {
        "risk_level": "N/A",
        "recommendation": "N/A",
        "confidence": "N/A",
        "top_features": "N/A"
    }

    shap_values = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shaps = explainer(feature_vector_scaled)
            if len(shaps.shape) == 3:
                shaps = shaps[:, :, 1]
            shaps.feature_names = feature_names
            shap_values = shaps[0]

            # Extract top features
            abs_shaps = np.abs(shap_values.values)
            top_indices = np.argsort(abs_shaps)[-3:][::-1]
            feat_list = []
            for idx in top_indices:
                feat_name = feature_names[idx]
                shap_val = shap_values.values[idx]
                if abs(shap_val) > 0.001:
                    impact_str = "↑ Risk" if shap_val > 0 else "↓ Risk"
                    feat_list.append(f"{feat_name} ({impact_str})")
            if feat_list:
                metadata["top_features"] = ", ".join(feat_list)
        except Exception:
            pass # Fail silently for SHAP, don't break the app
            
    # Calculate detailed metadata fields
    prob = probability[1]
    if prob < 0.34:
        metadata["risk_level"] = "Low"
        metadata["recommendation"] = "Proceed"
    elif prob < 0.67:
        metadata["risk_level"] = "Medium"
        metadata["recommendation"] = "Review carefully"
    else:
        metadata["risk_level"] = "High"
        metadata["recommendation"] = "Reject / High Risk"
        
    if prob >= 0.85 or prob <= 0.15:
        metadata["confidence"] = "High confidence"
    elif prob >= 0.70 or prob <= 0.30:
        metadata["confidence"] = "Moderate confidence"
    else:
        metadata["confidence"] = "Uncertain"
            
    return prediction, probability, desc, shap_values, None, metadata


# ── Streamlit UI ───────────────────────────────────────────────

st.set_page_config(
    page_title="ToxPredict API",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner hackathon look
st.markdown("""
    <style>
    .stAlert {border-radius: 10px;}
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--faded-text-color);
        color: var(--text-color);
    }
    .metric-card p {
        color: var(--text-color);
        opacity: 0.8;
    }
    h1, h2, h3 {font-family: 'Inter', sans-serif;}
    </style>
""", unsafe_allow_html=True)


# Sidebar layout
with st.sidebar:
    st.title("🧬 ToxPredict")
    st.markdown("Early-stage drug toxicity prediction tool built for the Hackathon.")
    st.markdown("---")
    
    st.markdown("### 📋 Try an Example")
    examples = {
        "Ethanol (Safe)": "CCO",
        "Aspirin (Safe)": "CC(=O)Oc1ccccc1C(=O)O",
        "Caffeine (Borderline)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Aniline (Toxic)": "Nc1ccccc1",
        "Dinitrobenzene (Toxic)": "O=[N+]([O-])c1cc(C(F)(F)F)cc([N+](=O)[O-])c1Cl",
    }
    
    selected_example = st.selectbox("Select a preset compound:", ["Custom Input"] + list(examples.keys()))
    
    st.markdown("---")
    artifact, model_type = load_model_file()
    if artifact:
        st.success(f"**Backend Status:** Online\n\nModel: {model_type}")
    else:
        st.error("**Backend Status:** Offline (Model Missing)")

# Main layout
st.title("Chemical Toxicity Screening")
st.write("Scan a molecule's SMILES representation to predict in-vitro toxicity across **12 Tox21 biological assays** (Nuclear Receptor + Stress Response pathways) and understand why the AI made its decision.")

with st.expander("⚠️ Model Scope & Limitations"):
    st.markdown("""
    **What this model measures:** This AI is trained on the [Tox21 dataset](https://tripod.nih.gov/tox21/), which tests molecules against 12 specific *in-vitro* biochemical assay endpoints:
    - **Nuclear Receptor disruption:** Androgen (NR-AR), Estrogen (NR-ER), Aromatase, PPAR-gamma, AhR
    - **Stress Response pathways:** ARE, ATAD5, HSE, MMP, p53

    **What this model does NOT measure:** Acute poisoning, respiratory toxicity, organ damage, or carcinogenicity via mechanisms not covered by the 12 Tox21 assays. For example, Formaldehyde is a known carcinogen, but it does not trigger any of the 12 Tox21 pathways, so our model (correctly per the training data) labels it as non-toxic.

    **Bottom line:** A "Non-toxic" prediction means the molecule is unlikely to disrupt these specific biological pathways, NOT that it's universally safe.
    """)

if artifact is None:
    st.stop()

# Determine input
if selected_example != "Custom Input":
    default_smiles = examples[selected_example]
else:
    default_smiles = "CCO"

tab1, tab2 = st.tabs(["Single Molecule Screening", "Batch Upload (CSV)"])

with tab1:
    st.markdown("### 1. Enter Molecule")
    smiles_input = st.text_input(
        label="SMILES Formula", 
        value=default_smiles,
        help="Simplified Molecular-Input Line-Entry System"
    )

    # Run Prediction
    if smiles_input:
        with st.spinner("Analyzing molecular structure against Tuned XGBoost Model..."):
            prediction, probability, descriptors, shap_values, error_msg, meta = predict_and_explain(smiles_input, artifact)
        
        if error_msg:
            st.error(f"**Failed to analyze molecule:** {error_msg}")
        else:
            st.markdown("---")
            st.markdown("### 2. Screening Report")
            
            # Display Comprehensive Fields
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Input SMILES:** `{smiles_input}`")
                st.markdown(f"**Validity Check:** Valid ✅")
                st.markdown(f"**Predicted Label:** {'**TOXIC** ⚠️' if prediction == 1 else '**Non-Toxic** ✅'}")
                st.markdown(f"**Toxicity Probability:** {probability[1]*100:.1f}%")
                
            with col_b:
                st.markdown(f"**Risk Level:** {meta['risk_level']}")
                st.markdown(f"**Model Confidence:** {meta['confidence']}")
                st.markdown(f"**Recommended Action:** {meta['recommendation']}")
                st.markdown(f"**Top Features:** {meta['top_features']}")

            st.write("") # Spacing
            
            # Visual Cards Row
            st.markdown("#### Key Metrics Overview")
            col1, col2, col3 = st.columns(3)
            toxic_prob = probability[1] * 100
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {'#FF4B4B' if prediction == 1 else '#00CC96'}; margin:0;">
                        {'⚠️ TOXIC' if prediction == 1 else '✅ SAFE'}
                    </h3>
                    <p style="font-size:14px; margin:0;">Model Verdict</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0;">{toxic_prob:.1f}%</h3>
                    <p style="font-size:14px; margin:0;">Toxicity Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0;">{meta['risk_level']}</h3>
                    <p style="font-size:14px; margin:0;">Risk Level</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("") # Spacing
        
            # Details Row
            bottom_left, bottom_right = st.columns([1, 2])
        
            with bottom_left:
                st.markdown("#### Molecular Structure")
                if HAS_RDKIT:
                    mol = Chem.MolFromSmiles(smiles_input)
                    if mol:
                        img = Draw.MolToImage(mol, size=(300, 300), fitImage=True)
                        st.image(img, use_container_width=True)
            
                st.markdown("#### Properties")
                props = {
                    "Weight (g/mol)": int(descriptors.get("MolWt", 0)),
                    "Lipophilicity (LogP)": round(descriptors.get("MolLogP", 0), 2),
                    "H-Donors": descriptors.get("NumHDonors", 0),
                    "H-Acceptors": descriptors.get("NumHAcceptors", 0)
                }
                st.dataframe(pd.DataFrame(list(props.items()), columns=["Property", "Value"]), hide_index=True, use_container_width=True)

            with bottom_right:
                st.markdown("#### AI Transparency (Why?)")
                st.write("The chart below (SHAP Waterfall) explains the math behind the prediction. **Red bars** represent molecular subsystems that push the drug toward toxicity. **Blue bars** pull it toward safety.")
            
                if shap_values is not None:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # Generate clean SHAP plot
                    shap.plots.waterfall(shap_values, max_display=10, show=False)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("SHAP Visualization not available for this molecule.")

with tab2:
    st.markdown("### 📁 Batch Prediction")
    st.write("Upload a CSV file containing a column named `smiles`. The model will predict toxicity for all compounds.")

    st.info("""
    💡 **Demo Tip:** A ready-to-use sample from the **ZINC-250k Drug Library** (1,000 molecules) 
    is available at `data/processed/zinc_demo_sample.csv`. Upload it here to see large-scale virtual screening in action!
    """, icon="🧪")

    # Show pre-computed ZINC screen results if available
    zinc_results_path = os.path.join(os.path.dirname(__file__), "..", "reports", "zinc_screen_results.csv")
    if os.path.exists(zinc_results_path):
        with st.expander("📊 View Pre-Computed ZINC-250k Screening Results (1,000 molecules)", expanded=False):
            zinc_df = pd.read_csv(zinc_results_path)
            n_toxic = (zinc_df["verdict"] == "TOXIC").sum()
            col_a, col_b = st.columns(2)
            col_a.metric("Molecules Screened", f"{len(zinc_df):,}")
            col_b.metric("Predicted Toxic", f"{n_toxic} ({n_toxic/len(zinc_df)*100:.1f}%)")
            st.dataframe(zinc_df.sort_values("toxicity_prob", ascending=False), use_container_width=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if "smiles" not in [c.lower() for c in batch_df.columns]:
                st.error("CSV must contain a column named 'smiles'.")
            else:
                # Find exact column name (case insensitive)
                s_col = [c for c in batch_df.columns if c.lower() == "smiles"][0]
                
                import datetime
                with st.spinner(f"Analyzing {len(batch_df)} compounds..."):
                    results = []
                    # Assign an ID column safely
                    if "ID" in batch_df.columns:
                        id_col = "ID"
                    elif "Molecule_ID" in batch_df.columns:
                        id_col = "Molecule_ID"
                    else:
                        batch_df["Molecule_ID"] = [f"MOL_{i+1:04d}" for i in range(len(batch_df))]
                        id_col = "Molecule_ID"

                    for _, row in batch_df.iterrows():
                        smi = row[s_col]
                        mol_id = row[id_col]
                        
                        if pd.isna(smi):
                            continue
                            
                        # Predict
                        pred, prob, desc, _, err, meta = predict_and_explain(str(smi), artifact)
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if err:
                            results.append({
                                "Timestamp": timestamp,
                                "Molecule_ID": mol_id,
                                "SMILES": smi,
                                "Valid": "No",
                                "Prediction Label": "ERROR",
                                "Toxicity Score": "N/A",
                                "Risk Level": "N/A",
                                "Confidence": "N/A",
                                "Top Features": "N/A",
                                "Recommended Action": "Failed to Parse"
                            })
                        else:
                            verdict = "Toxic" if pred == 1 else "Non-toxic"
                            results.append({
                                "Timestamp": timestamp,
                                "Molecule_ID": mol_id,
                                "SMILES": smi,
                                "Valid": "Yes",
                                "Prediction Label": verdict,
                                "Toxicity Score": round(prob[1], 4),
                                "Risk Level": meta["risk_level"],
                                "Confidence": meta["confidence"],
                                "Top Features": meta["top_features"],
                                "Recommended Action": meta["recommendation"]
                            })
                    
                    results_df = pd.DataFrame(results)
                    st.success("Batch Analysis Complete!")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Provide download button
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Full Screening Report (.csv)",
                        data=csv_export,
                        file_name="comprehensive_toxicity_report.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Failed to process file: {str(e)}")
