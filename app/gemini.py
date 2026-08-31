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