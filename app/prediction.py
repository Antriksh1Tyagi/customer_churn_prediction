import os
import joblib
import pandas as pd


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")


# --------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

def load_model():
    """Load the trained ML model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. The trained model needs to be added "
            "to the model folder."
        )

    return joblib.load(MODEL_PATH)


def load_preprocessor():
    """Load the saved data preprocessor."""

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "preprocessor.pkl not found. The preprocessor needs to be added "
            "to the model folder."
        )

    return joblib.load(PREPROCESSOR_PATH)


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

    # Load model and preprocessor only when prediction is requested
    model = load_model()
    preprocessor = load_preprocessor()

    # Apply preprocessing
    processed_data = preprocessor.transform(customer_data)

    # Make prediction
    prediction = model.predict(processed_data)[0]

    # Get probability if the model supports it
    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(processed_data)[0][1]

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
    preprocessor = load_preprocessor()

    # Keep original data for displaying results
    result_data = data.copy()

    # Apply preprocessing
    processed_data = preprocessor.transform(data)

    # Predictions
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