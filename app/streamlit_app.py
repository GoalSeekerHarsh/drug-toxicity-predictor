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
        return None, None, None, None, "Invalid SMILES string. RDKit could not parse it."
    
    try:
        desc = compute_descriptors(mol)
        fp = compute_morgan_fingerprint(mol, radius=2, n_bits=1024)
        if desc is None or fp is None:
            return None, None, None, None, "Failed to compute chemical features."
    except Exception as e:
        return None, None, None, None, f"Feature extraction error: {str(e)}"
    
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
        return None, None, None, None, f"Data scaling error: {str(e)}"
    
    try:
        model = artifact["model"]
        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0]
    except Exception as e:
        return None, None, None, None, f"Model prediction error: {str(e)}"

    shap_values = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            shaps = explainer(feature_vector_scaled)
            if len(shaps.shape) == 3:
                shaps = shaps[:, :, 1]
            shaps.feature_names = feature_names
            shap_values = shaps[0]
        except Exception:
            pass # Fail silently for SHAP, don't break the app
            
    return prediction, probability, desc, shap_values, None


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
    .main {background-color: #0E1117;}
    .stAlert {border-radius: 10px;}
    .metric-card {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #30363D;
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
        "Caffeine (Safe)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Toxic Example": "O=C(O)CCC(=O)c1ccc(-c2ccccc2)cc1",
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
st.write("Scan a molecule's SMILES representation to predict in-vitro toxicity (NR-AR endpoint) and understand why the AI made its decision.")

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
            prediction, probability, descriptors, shap_values, error_msg = predict_and_explain(smiles_input, artifact)
    
        if error_msg:
            st.error(f"**Failed to analyze molecule:** {error_msg}")
        else:
            st.markdown("---")
            st.markdown("### 2. Screening Results")
        
            # Top Metrics Row
            col1, col2, col3 = st.columns(3)
        
            toxic_prob = probability[1] * 100
        
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: {'#FF4B4B' if prediction == 1 else '#00CC96'}; margin:0;">
                        {'⚠️ TOXIC' if prediction == 1 else '✅ SAFE'}
                    </h3>
                    <p style="color:gray; font-size:14px; margin:0;">Model Verdict</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0;">{toxic_prob:.1f}%</h3>
                    <p style="color:gray; font-size:14px; margin:0;">Toxicity Confidence Range</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0;">{int(descriptors.get('MolWt', 0))} g/mol</h3>
                    <p style="color:gray; font-size:14px; margin:0;">Molecular Weight</p>
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
                        # RDKit drawing with a clean white background
                        img = Draw.MolToImage(mol, size=(300, 300), fitImage=True)
                        st.image(img, use_container_width=True)
            
                st.markdown("#### Key Properties")
                props = {
                    "Lipophilicity (LogP)": round(descriptors.get("MolLogP", 0), 2),
                    "Hydrogen Donors": descriptors.get("NumHDonors", 0),
                    "Hydrogen Acceptors": descriptors.get("NumHAcceptors", 0),
                    "Rotatable Bonds": descriptors.get("NumRotatableBonds", 0),
                }
                # Clean dataframe layout
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
                
                with st.spinner(f"Analyzing {len(batch_df)} compounds..."):
                    results = []
                    for smi in batch_df[s_col].dropna():
                        # We use predict_and_explain but ignore SHAP for batch speed
                        pred, prob, desc, _, err = predict_and_explain(str(smi), artifact)
                        if err:
                            results.append({"SMILES": smi, "Verdict": "ERROR", "Probability": "N/A"})
                        else:
                            verdict = "TOXIC" if pred == 1 else "SAFE"
                            results.append({"SMILES": smi, "Verdict": verdict, "Probability": f"{prob[1]*100:.1f}%"})
                    
                    results_df = pd.DataFrame(results)
                    st.success("Batch Analysis Complete!")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Provide download button
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_export,
                        file_name="toxicity_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Failed to process file: {str(e)}")
