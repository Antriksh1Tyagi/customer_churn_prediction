import os
import sys
import pandas as pd
import streamlit as st
from prediction import predict_customer, predict_csv
from dashboard import show_dashboard
from export_utils import create_csv, create_pdf_report
# Add your Gemini module import here
from gemini import generate_retention_advice 


# ==================================================
# PROJECT PATHS
# ==================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


# ==================================================
# PROJECT IMPORTS
# ==================================================

from prediction import predict_customer, predict_csv
from dashboard import show_dashboard
from export_utils import create_csv, create_pdf_report


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background:
        radial-gradient(
            circle at 92% 0%,
            #dff4ff 0%,
            #f7fcff 25%,
            #ffffff 58%,
            #ffffff 100%
        );
}


/* Streamlit top header */
header[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e4eff7 !important;
    box-shadow: none !important;
}


/* Main content */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    padding-bottom: 3rem;
}


/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #ffffff 0%,
            #f3faff 100%
        ) !important;
    border-right: 1px solid #dcebf5;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

section[data-testid="stSidebar"] h2 {
    color: #0875d1 !important;
    font-size: 27px !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}


/* Headings */
h1 {
    color: #12385f !important;
    font-weight: 750 !important;
}

h2 {
    color: #173f68 !important;
    font-weight: 700 !important;
}

h3 {
    color: #1b4772 !important;
    font-weight: 650 !important;
}

p {
    color: #526b82;
}


/* Buttons */
.stButton > button {
    background:
        linear-gradient(
            135deg,
            #1689e8,
            #55b4f5
        ) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow:
        0 4px 12px rgba(22, 137, 232, 0.18);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow:
        0 7px 18px rgba(22, 137, 232, 0.25);
}


/* Download buttons */
.stDownloadButton > button {
    background: #edf8ff !important;
    color: #0875d1 !important;
    border: 1px solid #c7e7fb !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stDownloadButton > button:hover {
    background: #dff2ff !important;
}


/* Inputs */
.stTextInput input,
.stNumberInput input {
    border-radius: 9px !important;
    border: 1px solid #d5e7f4 !important;
}

div[data-baseweb="select"] {
    border-radius: 9px !important;
}


/* File uploader */
section[data-testid="stFileUploader"] {
    background: #f5fbff !important;
    border: 2px dashed #b7ddf7 !important;
    border-radius: 14px !important;
    padding: 12px;
}


/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #deedf7 !important;
    border-radius: 16px !important;
    padding: 18px !important;
    box-shadow:
        0 5px 18px rgba(30, 80, 120, 0.07);
}

div[data-testid="stMetricValue"] {
    color: #0875d1 !important;
    font-weight: 750 !important;
}


/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e1edf7;
}


/* Dividers */
hr {
    border-color: #e2edf5 !important;
}


/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* Sidebar navigation */
section[data-testid="stSidebar"]
div[role="radiogroup"] {
    gap: 6px;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label {
    border-radius: 9px;
    padding: 5px 8px;
}

section[data-testid="stSidebar"]
div[role="radiogroup"] label:hover {
    background: #eaf7ff;
}


/* Hide default menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## ◈ CHURN AI")

    st.caption("Customer Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            " Dashboard",
            " Single Prediction",
            " CSV Upload",
            " Export Reports"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.info(
        "🛡 **Churn Prediction**\n\n"
        "AI-powered customer retention insights."
    )


# ==================================================
# DATA PATH
# ==================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "customer_churn.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    return None


data = load_data()


# ==================================================
# SINGLE CUSTOMER FORM
# ==================================================

def prediction_form():

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=100,
            value=12
        )

        usage = st.number_input(
            "Usage Frequency",
            min_value=0,
            max_value=100,
            value=15
        )

    with col2:

        support = st.number_input(
            "Support Calls",
            min_value=0,
            max_value=100,
            value=3
        )

        payment = st.number_input(
            "Payment Delay",
            min_value=0,
            max_value=100,
            value=5
        )

        subscription = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"]
        )

    with col3:

        contract = st.selectbox(
            "Contract Length",
            ["Monthly", "Quarterly", "Annual"]
        )

        spend = st.number_input(
            "Total Spend",
            min_value=0.0,
            max_value=100000.0,
            value=500.0
        )

        interaction = st.number_input(
            "Last Interaction",
            min_value=0,
            max_value=365,
            value=15
        )

    return {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": payment,
        "Subscription Type": subscription,
        "Contract Length": contract,
        "Total Spend": spend,
        "Last Interaction": interaction
    }


# ==================================================
# DASHBOARD
# ==================================================

if page == " Dashboard":

    st.title("Customer Churn Analytics")

    st.caption(
        "Understand your customers and make "
        "smarter retention decisions."
    )

    if data is None:

        st.error(
            "customer_churn.csv was not found."
        )

    else:

        show_dashboard(data)


# ==================================================
# SINGLE PREDICTION
# ==================================================
elif page == " Single Prediction":

    st.title("Single Customer Prediction")

    st.caption(
        "Estimate the customer's probability of churn."
    )

    st.divider()

    customer = prediction_form()

    if st.button(
        " Predict Customer Churn",
        type="primary",
        use_container_width=True
    ):

        try:

            result = predict_customer(customer)

            st.divider()

            st.subheader("Prediction Result")

            if result["prediction"]:
                st.error(
                    " High Churn Risk — "
                    "Customer is likely to churn."
                )
            else:
                st.success(
                    " Low Churn Risk — "
                    "Customer is not likely to churn."
                )

            probability = result.get(
                "churn_probability"
            )

            if probability is not None:

                col1, col2 = st.columns(2)

                col1.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

                if probability >= 0.7:
                    risk = "High Risk"
                elif probability >= 0.4:
                    risk = "Medium Risk"
                else:
                    risk = "Low Risk"

                col2.metric(
                    "Risk Level",
                    risk
                )

                st.progress(
                    min(max(probability, 0), 1)
                )
                
            # --------------------------------------------------
            # INTEGRATED: GEMINI AI RETENTION ADVICE
            # --------------------------------------------------
            st.divider()
            st.subheader("🤖 AI Retention Insights")
            
            # Combine the user form inputs with the machine learning model output
            ai_input_data = customer.copy()
            ai_input_data["Churn Probability"] = f"{probability * 100:.2f}%" if probability is not None else "N/A"
            ai_input_data["Risk Level"] = risk if probability is not None else "N/A"
            
            # Use a spinner while generating the analysis response
            with st.spinner("Gemini is analyzing customer metrics..."):
                advice = generate_retention_advice(ai_input_data)
                
            # Render the AI markdown response beautifully
            st.markdown(advice)

        except FileNotFoundError:

            st.warning(
                "The trained ML model is not available yet."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ==================================================
# CSV UPLOAD
# ==================================================

elif page == "☁️ CSV Upload":

    st.title("CSV Upload & Prediction")

    st.caption(
        "Upload customer data and generate predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file:

        try:

            uploaded = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"✓ {len(uploaded):,} customers loaded."
            )

            st.subheader("Data Preview")

            st.dataframe(
                uploaded.head(10),
                use_container_width=True
            )

            required = [
                "Age",
                "Gender",
                "Tenure",
                "Usage Frequency",
                "Support Calls",
                "Payment Delay",
                "Subscription Type",
                "Contract Length",
                "Total Spend",
                "Last Interaction"
            ]

            missing = [
                column
                for column in required
                if column not in uploaded.columns
            ]

            if missing:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing)
                )

            elif st.button(
                " Generate Predictions",
                type="primary",
                use_container_width=True
            ):

                try:

                    results = predict_csv(
                        uploaded[required]
                    )

                    st.session_state["results"] = results

                    st.success(
                        "Predictions generated successfully."
                    )

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                except FileNotFoundError:

                    st.warning(
                        "The trained ML model is not available yet."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

        except Exception as e:

            st.error(
                f"Could not read the CSV: {e}"
            )


# ==================================================
# EXPORT REPORTS
# ==================================================

elif page == " Export Reports":

    st.title("Export Prediction Reports")

    st.caption(
        "Download your prediction results "
        "for analysis or reporting."
    )

    results = st.session_state.get("results")

    if results is None:

        st.info(
            "Generate predictions from CSV Upload first."
        )

    else:

        st.subheader("Report Preview")

        st.dataframe(
            results.head(10),
            use_container_width=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                " Download CSV",
                create_csv(results),
                "churn_predictions.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:

            st.download_button(
                " Download PDF",
                create_pdf_report(results),
                "churn_prediction_report.pdf",
                "application/pdf",
                use_container_width=True
            )
