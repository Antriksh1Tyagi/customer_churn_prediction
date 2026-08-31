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
    """
    Predict whether a customer is likely to churn.

    customer_data can be either:
    - a dictionary containing customer details
    - a pandas DataFrame
    """

    # Convert dictionary to DataFrame
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    # Make sure input is a DataFrame
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Customer data must be a dictionary or pandas DataFrame.")

    # Align the feature names and order exactly to match training data
    customer_data = align_columns(customer_data)

    # Load model pipeline only when prediction is requested
    model = load_model()

    # Pass the RAW data directly to the model pipeline.
    # The pipeline will automatically run the ColumnTransformer internally.
    prediction = model.predict(customer_data)[0]

    # Get probability if the model supports it
    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(customer_data)[0][1]

    # Convert prediction to readable result
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
