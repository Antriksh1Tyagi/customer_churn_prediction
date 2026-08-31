import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

def show_dashboard(data):
    """Display the customer churn analytics dashboard."""

    st.title("Customer Churn Analytics")
    st.caption("Explore customer behavior and identify churn patterns.")

    # --------------------------------------------------
    # DATA VALIDATION
    # --------------------------------------------------

    if data is None or data.empty:
        st.warning("No customer data available.")
        return

    data = data.copy()

    # --------------------------------------------------
    # SIDEBAR FILTERS
    # --------------------------------------------------

    st.sidebar.subheader("Dashboard Filters")

    filtered_data = data.copy()

    if "Gender" in data.columns:
        gender_options = ["All"] + sorted(
            data["Gender"].dropna().astype(str).unique().tolist()
        )

        selected_gender = st.sidebar.selectbox(
            "Gender",
            gender_options
        )

        if selected_gender != "All":
            filtered_data = filtered_data[
                filtered_data["Gender"].astype(str) == selected_gender
            ]

    if "Subscription Type" in data.columns:
        subscription_options = ["All"] + sorted(
            data["Subscription Type"].dropna().astype(str).unique().tolist()
        )

        selected_subscription = st.sidebar.selectbox(
            "Subscription Type",
            subscription_options
        )

        if selected_subscription != "All":
            filtered_data = filtered_data[
                filtered_data["Subscription Type"].astype(str)
                == selected_subscription
            ]

    if "Contract Length" in data.columns:
        contract_options = ["All"] + sorted(
            data["Contract Length"].dropna().astype(str).unique().tolist()
        )

        selected_contract = st.sidebar.selectbox(
            "Contract Length",
            contract_options
        )

        if selected_contract != "All":
            filtered_data = filtered_data[
                filtered_data["Contract Length"].astype(str)
                == selected_contract
            ]

    # --------------------------------------------------
    # EMPTY FILTER RESULT
    # --------------------------------------------------

    if filtered_data.empty:
        st.warning("No customers match the selected filters.")
        return

    # --------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------

    total_customers = len(filtered_data)

    if "Churn" in filtered_data.columns:

        churned_customers = int(
            (filtered_data["Churn"] == 1).sum()
        )

        churn_rate = (
            churned_customers / total_customers * 100
        )

    else:
        churned_customers = 0
        churn_rate = 0

    if "Tenure" in filtered_data.columns:
        average_tenure = filtered_data["Tenure"].mean()
    else:
        average_tenure = 0

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Average Tenure",
        f"{average_tenure:.1f}"
    )

    st.divider()

    # --------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------

    if "Churn" in filtered_data.columns:

        st.subheader("Churn Overview")

        col1, col2 = st.columns(2)

        with col1:

            churn_counts = (
                filtered_data["Churn"]
                .value_counts()
                .sort_index()
            )

            values = [
                churn_counts.get(0, 0),
                churn_counts.get(1, 0)
            ]

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.bar(
                ["Not Churned", "Churned"],
                values
            )

            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Churn Distribution")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

        # --------------------------------------------------
        # CHURN RATE BY SUBSCRIPTION
        # --------------------------------------------------

        with col2:

            if "Subscription Type" in filtered_data.columns:

                subscription_churn = (
                    filtered_data
                    .groupby("Subscription Type")["Churn"]
                    .mean()
                    * 100
                )

                fig, ax = plt.subplots(figsize=(7, 4))

                subscription_churn.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_xlabel("Subscription Type")
                ax.set_ylabel("Churn Rate (%)")
                ax.set_title("Churn Rate by Subscription Type")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                plt.xticks(rotation=0)

                st.pyplot(fig, use_container_width=True)

                plt.close(fig)

    # --------------------------------------------------
    # CUSTOMER SEGMENTATION
    # --------------------------------------------------

    st.subheader("Customer Segmentation")

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # CONTRACT LENGTH
    # --------------------------------------------------

    with col1:

        if (
            "Contract Length" in filtered_data.columns
            and "Churn" in filtered_data.columns
        ):

            contract_churn = (
                filtered_data
                .groupby("Contract Length")["Churn"]
                .mean()
                * 100
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            contract_churn.plot(
                kind="bar",
                ax=ax
            )

            ax.set_xlabel("Contract Length")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_title("Churn Rate by Contract Length")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.xticks(rotation=0)

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

    # --------------------------------------------------
    # AGE DISTRIBUTION
    # --------------------------------------------------

    with col2:

        if "Age" in filtered_data.columns:

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.hist(
                filtered_data["Age"].dropna(),
                bins=10
            )

            ax.set_xlabel("Age")
            ax.set_ylabel("Number of Customers")
            ax.set_title("Customer Age Distribution")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

    # --------------------------------------------------
    # CUSTOMER BEHAVIOR
    # --------------------------------------------------

    st.subheader("Customer Behavior")

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # TENURE BY CHURN
    # --------------------------------------------------

    with col1:

        if (
            "Tenure" in filtered_data.columns
            and "Churn" in filtered_data.columns
        ):

            tenure_data = (
                filtered_data
                .groupby("Churn")["Tenure"]
                .mean()
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.bar(
                ["Not Churned", "Churned"],
                [
                    tenure_data.get(0, 0),
                    tenure_data.get(1, 0)
                ]
            )

            ax.set_xlabel("Customer Status")
            ax.set_ylabel("Average Tenure")
            ax.set_title("Average Tenure by Churn Status")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

    # --------------------------------------------------
    # SUPPORT CALLS BY CHURN
    # --------------------------------------------------

    with col2:

        if (
            "Support Calls" in filtered_data.columns
            and "Churn" in filtered_data.columns
        ):

            support_data = (
                filtered_data
                .groupby("Churn")["Support Calls"]
                .mean()
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.bar(
                ["Not Churned", "Churned"],
                [
                    support_data.get(0, 0),
                    support_data.get(1, 0)
                ]
            )

            ax.set_xlabel("Customer Status")
            ax.set_ylabel("Average Support Calls")
            ax.set_title("Support Calls by Churn Status")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

    # --------------------------------------------------
    # PAYMENT DELAY
    # --------------------------------------------------

    if (
        "Payment Delay" in filtered_data.columns
        and "Churn" in filtered_data.columns
    ):

        st.subheader("Payment Delay Analysis")

        payment_data = (
            filtered_data
            .groupby("Churn")["Payment Delay"]
            .mean()
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.bar(
            ["Not Churned", "Churned"],
            [
                payment_data.get(0, 0),
                payment_data.get(1, 0)
            ]
        )

        ax.set_xlabel("Customer Status")
        ax.set_ylabel("Average Payment Delay")
        ax.set_title("Average Payment Delay by Churn Status")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

    # --------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------

    with st.expander("View Filtered Customer Data"):

        st.dataframe(
            filtered_data,
            use_container_width=True
        )