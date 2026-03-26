import sys
import os
import io
import warnings

# Suppress warnings for clean testing output
warnings.filterwarnings("ignore")

# Add app to path to import the logic
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from streamlit_app import load_model_file, predict_and_explain

def run_tests():
    print("Loading model...")
    artifact, model_type = load_model_file()
    if artifact is None:
        print("ERROR: Model not found.")
        return

    print(f"Model loaded: {model_type}")
    
    test_cases = [
        ("Valid Safe", "CCO"),
        ("Valid Toxic", "O=C(O)CCC(=O)c1ccc(-c2ccccc2)cc1"),
        ("Invalid SMILES 1", "invalid_smiles"),
        ("Invalid SMILES 2", "12345"),
        ("Empty SMILES", ""),
        ("None", None),
        ("Extremely Long SMILES", "C" * 150),
        ("Salt/Disconnected", "[Na+].[Cl-]"),
    ]

    bugs_found = 0

    print("Running tests...\n")
    for name, smiles in test_cases:
        print(f"--- Test: {name} ({smiles}) ---")
        try:
            prediction, probability, descriptors, shap_values, error_msg = predict_and_explain(smiles, artifact)
            if error_msg:
                print(f"Gracefully handled with error message: {error_msg}")
            else:
                pred_label = "Toxic" if prediction == 1 else "Safe"
                print(f"Prediction: {pred_label} (Prob: {probability[1]:.2f})")
                print(f"Descriptors computed: {len(descriptors)}")
                print(f"SHAP values computed: {'Yes' if shap_values is not None else 'No'}")
        except Exception as e:
            print(f"💥 BUG FOUND: Unhandled Exception: {str(e)}")
            bugs_found += 1
        print("")
        
    print(f"Tests complete. Total crashing bugs found: {bugs_found}")

if __name__ == "__main__":
    run_tests()
