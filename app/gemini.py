# ---------------------- Phase 1 -----------------------

"""
Gemini AI integration for Customer Churn Prediction.

Phase 1:
- Gemini API client setup
- Secure API key handling
- Gemini 2.5 Flash connectivity test
- Basic API error handling
"""

import os
from typing import Optional

from google import genai


# Gemini model used by the project
GEMINI_MODEL = "gemini-3.6-flash"


def create_gemini_client() -> genai.Client:
    """
    Create and return a Gemini API client.

    Returns:
        genai.Client: Configured Gemini client.

    Raises:
        ValueError: If GEMINI_API_KEY is not configured.
        RuntimeError: If the client cannot be initialized.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set."
        )

    try:
        return genai.Client(api_key=api_key)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Gemini client: {exc}"
        ) from exc


def get_gemini_client() -> genai.Client:
    """
    Return a configured Gemini client.

    Returns:
        genai.Client: Gemini API client.
    """

    return create_gemini_client()


def test_gemini_connection() -> tuple[bool, Optional[str]]:
    """
    Test the connection to the Gemini API.

    Returns:
        tuple[bool, Optional[str]]:
            True and response text if successful.
            False and error message if unsuccessful.
    """

    try:
        client = get_gemini_client()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with exactly: Gemini connection successful."
        )

        if not response or not response.text:
            return False, "Gemini returned an empty response."

        return True, response.text.strip()

    except Exception as exc:
        return False, f"Gemini API error: {exc}"

# ---------------------- Phase 2 -----------------------

def generate_retention_advice(customer_data: dict) -> str:
    """
    Generate personalized customer retention recommendations.

    Args:
        customer_data: Dictionary containing customer details and
            churn prediction information.

    Returns:
        str: Business-friendly retention recommendations.

    Raises:
        ValueError: If customer_data is empty or invalid.
    """

    if not isinstance(customer_data, dict) or not customer_data:
        raise ValueError("customer_data must be a non-empty dictionary.")

    client = get_gemini_client()

    prompt = f"""
You are a professional Customer Success Manager helping a company
reduce customer churn.

Analyze the following customer information:

Age: {customer_data.get("Age", "Not provided")}
Gender: {customer_data.get("Gender", "Not provided")}
Tenure: {customer_data.get("Tenure", "Not provided")}
Usage Frequency: {customer_data.get("Usage Frequency", "Not provided")}
Support Calls: {customer_data.get("Support Calls", "Not provided")}
Payment Delay: {customer_data.get("Payment Delay", "Not provided")}
Subscription Type: {customer_data.get("Subscription Type", "Not provided")}
Contract Length: {customer_data.get("Contract Length", "Not provided")}
Total Spend: {customer_data.get("Total Spend", "Not provided")}
Last Interaction: {customer_data.get("Last Interaction", "Not provided")}
Churn Probability: {customer_data.get("Churn Probability", "Not provided")}
Risk Level: {customer_data.get("Risk Level", "Not provided")}

Based on these details, provide practical and personalized
customer retention advice.

Your response MUST contain exactly these four sections:

1. Risk Summary
Give a concise explanation of the customer's current churn risk.

2. Why the customer may churn
Explain the most relevant behavioral or account-related reasons
that could contribute to churn. Do not invent information that is
not supported by the provided customer data.

3. Top 3 Retention Actions
Provide exactly three practical actions that a customer success
team could take to improve the likelihood of retaining this customer.
Prioritize the actions based on the customer's specific situation.

4. Suggested communication tone
Recommend the appropriate tone for communicating with this customer
and briefly explain why.

Requirements:
- Use professional, business-friendly language.
- Be specific to the customer.
- Avoid technical machine-learning terminology.
- Do not mention that you are an AI.
- Do not use unnecessary introductions or conclusions.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        if not response or not response.text:
            return "Unable to generate retention advice."

        return response.text.strip()

    except Exception as exc:
        return f"Unable to generate retention advice: {exc}"