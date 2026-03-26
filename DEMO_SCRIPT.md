# 🎙️ ToxPredict: 1-Minute Pitch & Demo Script

## ⏱️ 1-Minute Project Pitch (Opening)
*"Hi, judges. We built **ToxPredict**, an automated chemical toxicity screening platform. Evaluating drug toxicity physically in a lab takes months and millions of dollars per compound. We solve this by bringing the lab into the computer.*

*We engineered a machine learning pipeline trained on the Tox21 dataset. We extract 1,241 chemical properties—ranging from molecular weight to Morgan Fingerprints—from a single SMILES string. We run this through an XGBoost algorithm optimized specifically to aggressively catch rare toxic compounds without misdiagnosing safe ones.*

*But we don't just predict; we **explain**. We integrated game-theoretic SHAP mathematical explanations so researchers know exactly **why** a drug might be dangerous. Let us show you."*

---

## 💻 Live Demo Flow

### **Step A — Open app**
*(Make sure the app is running: `streamlit run app/streamlit_app.py`)*
**Say:** “This is our interface where researchers can instantly input a molecule.”

### **Step B — Show a valid example (Safe Compound)**
*(Paste the SMILES below in the Custom Input)*
**Input:** `CCO` *(Ethanol)*
**Say:** “Here is a very simple molecule. Let’s predict its toxicity.”
*(Click predict/hit Enter)*

### **Step C — Explain output & Transparency**
**Say clearly:**
*   “This is the model verdict and the exact toxicity probability range.”
*   “This means the compound has a low risk profile for this assay.”
*   “These intrinsic numerical properties instantly influenced the prediction.”
*(Point visually to the probability card, the label card, and the molecular weight)*

### **Step D — Show Explanation (Wait for SHAP Plot)**
**Say:** “However, in medicine, we are not just predicting—we **must** explain the decision. This is where we stand out. This SHAP chart provides total AI transparency, detailing the exact molecular substructures that influenced the AI’s math.”
*(Point to the red/blue horizontal bars)*

### **Step E — Edge Case Handling (The "Judge Impressor")**
*(Enter a fake, completely invalid SMILES string)*
**Input:** `INVALID_CHEMICAL_xyz123`
*(Hit enter)*
**Say:** “Since real-world inputs can be messy, we built the platform to handle invalid inputs gracefully. Notice how it instantly catches the structural error without crashing the system.”

### **Step F — Optional: Batch Upload (The Scale Play)**
*(Click the 'Batch Upload' Tab at the top)*
**Say:** “Finally, predicting one molecule is great, but researchers work with thousands. You can also upload entire CSV files and get structured predictions cleanly at scale via our Batch Processing engine.”

---

### 🧪 Pre-Rehearsed Test Molecules
Have these ready to copy-paste during the demo:
1. **Perfect Success Case (Safe):** `CCO` (Ethanol)
2. **Perfect Success Case (Toxic):** `O=C(O)CCC(=O)c1ccc(-c2ccccc2)cc1` (A known toxic assay hit)
3. **Failure/Edge Case (Invalid String):** `NOT_A_MOLECULE`
