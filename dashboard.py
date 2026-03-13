# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 8 of 8: Interactive Dashboard
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Presents all project results in an interactive
# web dashboard that audit teams can use to
# explore risk scores, filter companies, and
# identify high priority audit targets.
#
# Built using Streamlit - a Python library
# that turns data scripts into web applications
# without needing any web development skills.
#
# Academic modules:
#   STAT40800 - Data Programming with Python (Grade: A)
#   ACM40960  - Projects in Maths Modelling (Grade: A)
#
# To run this dashboard:
#   streamlit run dashboard.py
#
# This is the final file in the pipeline.
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Audit Risk Intelligence System",
    page_icon="[AUDIT]",
    layout="wide"
)

# Colour scheme
DELOITTE_GREEN  = "#86BC25"
DELOITTE_BLUE   = "#0076A8"
DELOITTE_RED    = "#DA291C"
DELOITTE_YELLOW = "#FFB81C"


# Load all data files
# st.cache_data means files are only loaded once - makes the dashboard faster

@st.cache_data
def load_data():
    """
    Loads all output files produced by the pipeline.
    Returns a dictionary containing each dataset.
    If a file is missing, that entry is set to None
    and the relevant dashboard page shows an error message.
    """
    data = {}

    try:
        data["risk"] = pd.read_csv("prathibha_risk_scores.csv")
    except Exception:
        data["risk"] = None

    try:
        data["anomaly"] = pd.read_csv("anomaly_detection_results.csv")
    except Exception:
        data["anomaly"] = None

    try:
        data["clustering"] = pd.read_csv("clustering_results.csv")
    except Exception:
        data["clustering"] = None

    try:
        data["model"] = pd.read_csv("predictive_model_results.csv")
    except Exception:
        data["model"] = None

    return data


data = load_data()


# Dashboard header

st.markdown(
    f"""
    <div style='background-color:{DELOITTE_GREEN};
                padding:20px;
                border-radius:8px;
                margin-bottom:20px'>
        <h1 style='color:white; margin:0; font-size:28px'>
            Audit Risk Intelligence System
        </h1>
        <p style='color:white; margin:5px 0 0 0; font-size:14px'>
            Author: Prathibha Kyatham  |
            UCD MSc Data Science  |
            731 Companies Analysed
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Sidebar navigation

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Deloitte.svg/2560px-Deloitte.svg.png",
    width=150
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Company Risk Table",
        "High Risk Alerts",
        "Anomaly Detection",
        "Predictive Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Academic Modules:**\n"
    "- STAT40800 Python \n"
    "- STAT40970 ML and AI \n"
    "- STAT40150 Multivariate \n"
    "- STAT40850 Bayesian \n"
    "- STAT40960 Networks \n"
    "- STAT40840 SAS \n"
    "- ACM40960 Modelling "
)


# ============================================
# PAGE 1: OVERVIEW
# ============================================

if page == "Overview":

    st.markdown("## Risk Score Overview")
    st.markdown(
        "Summary of the **Prathibha Risk Score** "
        "applied to 731 public companies using SEC EDGAR financial data."
    )

    if data["risk"] is not None:
        df = data["risk"]

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        total     = len(df)
        very_high = (df["Risk_Level"] == "Very High Risk").sum()
        high      = (df["Risk_Level"] == "High Risk").sum()
        medium    = (df["Risk_Level"] == "Medium Risk").sum()
        low       = (df["Risk_Level"] == "Low Risk").sum()

        col1.metric("Total Companies",  f"{total:,}")
        col2.metric("Very High Risk",   f"{very_high}", f"{very_high / total * 100:.1f}%")
        col3.metric("High Risk",        f"{high}",      f"{high / total * 100:.1f}%")
        col4.metric("Medium Risk",      f"{medium}",    f"{medium / total * 100:.1f}%")
        col5.metric("Low Risk",         f"{low}",       f"{low / total * 100:.1f}%")

        st.markdown("---")

        # Charts
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Risk Level Distribution")
            risk_counts = df["Risk_Level"].value_counts()
            colors_map  = {
                "Very High Risk" : DELOITTE_RED,
                "High Risk"      : DELOITTE_YELLOW,
                "Medium Risk"    : DELOITTE_BLUE,
                "Low Risk"       : DELOITTE_GREEN
            }
            colors = [colors_map.get(r, DELOITTE_BLUE) for r in risk_counts.index]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="white", alpha=0.85)
            ax.set_ylabel("Number of Companies")
            ax.set_title("Companies by Risk Level", fontweight="bold")
            ax.set_xticklabels(risk_counts.index, rotation=15, ha="right")
            ax.grid(axis="y", alpha=0.3)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        str(int(bar.get_height())), ha="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("#### PRS Score Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["PRS_Score"], bins=30, color=DELOITTE_GREEN, edgecolor="white", alpha=0.85)
            ax.axvline(x=65, color=DELOITTE_YELLOW, linestyle="--", linewidth=2, label="High Risk threshold (65)")
            ax.axvline(x=80, color=DELOITTE_RED,    linestyle="--", linewidth=2, label="Very High Risk threshold (80)")
            ax.set_xlabel("PRS Score")
            ax.set_ylabel("Number of Companies")
            ax.set_title("Prathibha Risk Score Distribution", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # Top 10 highest risk
        st.markdown("#### Top 10 Highest Risk Companies")
        top10 = df.nlargest(10, "PRS_Score")[["Company", "PRS_Score", "Risk_Level"]].reset_index(drop=True)
        top10.index = top10.index + 1
        st.dataframe(top10, use_container_width=True)

    else:
        st.error("Risk scores file not found. Run risk_score.py first.")


# ============================================
# PAGE 2: COMPANY RISK TABLE
# ============================================

elif page == "Company Risk Table":

    st.markdown("## Company Risk Table")
    st.markdown("Search and filter all 731 companies by risk level and score.")

    if data["risk"] is not None:
        df = data["risk"]

        col1, col2, col3 = st.columns(3)

        with col1:
            search = st.text_input("Search Company Name", "")

        with col2:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=df["Risk_Level"].unique().tolist(),
                default=df["Risk_Level"].unique().tolist()
            )

        with col3:
            score_range = st.slider("PRS Score Range", min_value=0, max_value=100, value=(0, 100))

        filtered = df.copy()

        if search:
            filtered = filtered[filtered["Company"].str.contains(search, case=False, na=False)]

        if risk_filter:
            filtered = filtered[filtered["Risk_Level"].isin(risk_filter)]

        filtered = filtered[
            (filtered["PRS_Score"] >= score_range[0]) &
            (filtered["PRS_Score"] <= score_range[1])
        ]

        filtered = filtered.sort_values("PRS_Score", ascending=False).reset_index(drop=True)
        filtered.index = filtered.index + 1

        st.markdown(f"**Showing {len(filtered)} companies**")
        st.dataframe(filtered, use_container_width=True, height=500)

    else:
        st.error("Risk scores file not found. Run risk_score.py first.")


# ============================================
# PAGE 3: HIGH RISK ALERTS
# ============================================

elif page == "High Risk Alerts":

    st.markdown("## High Risk Alerts")
    st.markdown("Companies flagged as High Risk or Very High Risk — priority targets for audit review.")

    if data["risk"] is not None:
        df = data["risk"]

        high_risk = df[
            df["Risk_Level"].isin(["High Risk", "Very High Risk"])
        ].sort_values("PRS_Score", ascending=False).reset_index(drop=True)
        high_risk.index = high_risk.index + 1

        st.error(f"{len(high_risk)} companies flagged for audit review.")

        # Very High Risk
        very_high = df[df["Risk_Level"] == "Very High Risk"].sort_values("PRS_Score", ascending=False)
        st.markdown("### Very High Risk Companies")
        st.markdown("These companies require immediate audit attention.")
        very_high_display = very_high[["Company", "PRS_Score", "Risk_Level"]].reset_index(drop=True)
        very_high_display.index += 1
        st.dataframe(very_high_display, use_container_width=True)

        st.markdown("---")

        # High Risk
        high_only = df[df["Risk_Level"] == "High Risk"].sort_values("PRS_Score", ascending=False)
        st.markdown("### High Risk Companies")
        high_display = high_only[["Company", "PRS_Score", "Risk_Level"]].reset_index(drop=True)
        high_display.index += 1
        st.dataframe(high_display, use_container_width=True)

    else:
        st.error("Risk scores file not found. Run risk_score.py first.")


# ============================================
# PAGE 4: ANOMALY DETECTION
# ============================================

elif page == "Anomaly Detection":

    st.markdown("## Anomaly Detection")
    st.markdown(
        "Companies flagged as financial anomalies using **Isolation Forest**. "
        "Academic module: STAT40970 Machine Learning and AI."
    )

    if data["anomaly"] is not None:
        df = data["anomaly"]

        anomalies = df[df["Anomaly_Flag"] == "ANOMALY"].reset_index(drop=True)
        anomalies.index += 1

        normal = df[df["Anomaly_Flag"] == "Normal"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Companies",    len(df))
        col2.metric("Anomalies Detected", len(anomalies), f"{len(anomalies) / len(df) * 100:.1f}%")
        col3.metric("Normal Companies",   len(normal))

        st.markdown("---")
        st.markdown("### Flagged Anomaly Companies")
        st.markdown("These companies have unusual financial patterns that deviate significantly from their peers.")

        if "Company" in anomalies.columns:
            cols_to_show = [c for c in ["Company", "Anomaly_Score", "Anomaly_Flag", "Debt_Ratio", "Profit_Margin"] if c in anomalies.columns]
            st.dataframe(anomalies[cols_to_show], use_container_width=True)

        st.markdown("---")
        st.markdown("### Anomaly Score Distribution")

        if "Anomaly_Score" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df[df["Anomaly_Flag"] == "Normal"]["Anomaly_Score"],
                    bins=30, color=DELOITTE_GREEN, alpha=0.7, label="Normal", edgecolor="white")
            ax.hist(df[df["Anomaly_Flag"] == "ANOMALY"]["Anomaly_Score"],
                    bins=30, color=DELOITTE_RED, alpha=0.7, label="Anomaly", edgecolor="white")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Number of Companies")
            ax.set_title("Anomaly Score Distribution", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    else:
        st.error("Anomaly detection file not found. Run anomaly_detection.py first.")


# ============================================
# PAGE 5: PREDICTIVE MODEL
# ============================================

elif page == "Predictive Model":

    st.markdown("## Predictive Model")
    st.markdown(
        "**Random Forest** model trained to predict audit risk. "
        "Academic module: STAT40970 Machine Learning and AI."
    )

    if data["model"] is not None:
        df = data["model"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy",       "98.0%")
        col2.metric("Decision Trees",       "200")
        col3.metric("Companies Predicted",  len(df))

        st.markdown("---")
        st.markdown("### Filter by Risk Probability")

        threshold = st.slider(
            "Show companies with High Risk Probability above:",
            min_value=0, max_value=100, value=50, step=5
        )

        if "RF_Probability" in df.columns:
            filtered = df[df["RF_Probability"] >= threshold].sort_values(
                "RF_Probability", ascending=False).reset_index(drop=True)
            filtered.index += 1

            st.markdown(f"**{len(filtered)} companies** with risk probability at or above {threshold}%")

            cols_to_show = [c for c in ["Company", "RF_Prediction", "RF_Probability", "PRS_Score", "Risk_Level"] if c in filtered.columns]
            st.dataframe(filtered[cols_to_show], use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("### Feature Importance")
        st.markdown("Which financial ratios are most important for predicting audit risk?")

        importance_data = {
            "Feature"    : ["Return on Assets", "Cash Ratio", "Leverage Ratio", "Debt Ratio", "Asset Efficiency", "Profit Margin"],
            "Importance" : [0.232, 0.225, 0.207, 0.173, 0.094, 0.069]
        }
        imp_df = pd.DataFrame(importance_data)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = [DELOITTE_RED if i == 0 else DELOITTE_BLUE for i in range(len(imp_df))]
        ax.barh(imp_df["Feature"], imp_df["Importance"], color=colors, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance - Random Forest", fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    else:
        st.error("Predictive model file not found. Run predictive_model.py first.")


# Footer

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align:center; color:gray; font-size:12px'>
        Audit Risk Intelligence System  |
        Prathibha Kyatham  |
        UCD MSc Data Science  |
        Built with Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)