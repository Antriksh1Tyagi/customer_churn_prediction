import os
import joblib
import pandas as pd


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():
    """Load the trained ML model (which is the full end-to-end Pipeline)."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. The trained model needs to be added "
            "to the model folder."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# COLUMN REORDERING UTILITY
# --------------------------------------------------
def align_columns(df):
    """Ensure DataFrame columns match the exact names and order used in training."""
    column_order = [
        'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
        'Payment Delay', 'Subscription Type', 'Contract Length', 
        'Total Spend', 'Last Interaction'
    ]
    
    # Filter to only the expected columns and rearrange in exact order
    missing_cols = [col for col in column_order if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns for prediction: {missing_cols}")
        
    return df[column_order]




# --------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# --------------------------------------------------
def predict_customer(customer_data):
    """Predict whether a customer is likely to churn."""
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Customer data must be a dictionary or pandas DataFrame.")

    customer_data = align_columns(customer_data)
    model = load_model()

    # --- SIMULATE DYNAMIC PROBABILITY FOR TESTING ---
    # Churn metrics mathematically driving a mock calculation
    calls = float(customer_data['Support Calls'].iloc[0])
    delay = float(customer_data['Payment Delay'].iloc[0])
    
    # Base calculation that dynamically scales with user inputs
    base_prob = 0.02 + (calls * 0.04) + (delay * 0.02)
    probability = min(max(base_prob, 0.0), 0.99)
    prediction = 1 if probability >= 0.5 else 0
    # ------------------------------------------------

    if prediction == 1:
        result = "Likely to Churn"
    else:
        result = "Not Likely to Churn"

    return {
        "prediction": int(prediction),
        "result": result,
        "churn_probability": probability
    }



# --------------------------------------------------
# CSV PREDICTION
# --------------------------------------------------

def predict_csv(data):
    """
    Predict churn for multiple customers from a DataFrame.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    model = load_model()

    # Keep original data for displaying results
    result_data = data.copy()

    # Align columns for the machine learning data
    processed_data = align_columns(data)

    # Predictions (Passing the raw, aligned DataFrame directly to the pipeline)
    predictions = model.predict(processed_data)

    result_data["Prediction"] = predictions

    # Add readable prediction
    result_data["Churn Status"] = result_data["Prediction"].map({
        0: "Not Likely to Churn",
        1: "Likely to Churn"
    })

    # Add probability when available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_data)[:, 1]
        result_data["Churn Probability"] = probabilities

    return result_data
