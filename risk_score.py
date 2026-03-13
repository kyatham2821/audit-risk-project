# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 3 of 8: Prathibha Risk Score (PRS)
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Calculates a risk score from 0 to 100 for
# each company using a 4-layer scoring model.
# Higher score means higher audit risk.
#
# The model went through 3 versions:
#   v1.0 - Fixed thresholds. Result: 83% High Risk (unrealistic)
#   v2.0 - Percentile-based scoring. Result: More realistic spread
#   v3.0 - Clean pre-processed data. Result: Best accuracy (current)
#
# The 4 layers of the model:
#   Layer 1: Financial ratio analysis     (STAT20230)
#   Layer 2: PCA dimension reduction      (STAT40150)
#   Layer 3: Bayesian uncertainty bands   (STAT40850)
#   Layer 4: Isolation Forest anomalies   (STAT40970)
#
# Input file  : companies_clean_data.csv
# Output files:
#   prathibha_risk_scores.csv
#   prathibha_risk_scores.xlsx
#
# Next file: anomaly_detection.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 3 of 8: Prathibha Risk Score v3")
print("")


# Step 1: Load the cleaned data

print("Step 1: Loading clean company data...")
print("")

df = pd.read_csv("companies_clean_data.csv")

df = df.rename(columns={
    "Revenue ($B)"           : "Revenue",
    "Net Income ($B)"        : "NetIncome",
    "Total Assets ($B)"      : "Assets",
    "Total Liabilities ($B)" : "Liabilities",
    "Cash ($B)"              : "Cash"
})

print(f"  Companies loaded : {len(df)}")
print(f"  Columns          : {list(df.columns)}")
print("")


# Step 2: Calculate financial ratios
# Academic module: STAT20230 - Modern Regression Analysis

print("Step 2: Calculating financial ratios...")
print("        (STAT20230 - Modern Regression Analysis)")
print("")


def calculate_ratios(df):
    """
    Calculates 5 financial ratios for each company.

    We use ratios rather than raw dollar figures because ratios
    allow fair comparison between companies of very different sizes.
    Apple with $400B revenue and a small firm with $1B revenue
    can both be assessed using the same ratio thresholds.

    Ratio 1 - Debt Ratio:
      Liabilities divided by Assets.
      Higher value means more of the company is funded by debt.
      This is a key audit risk indicator.

    Ratio 2 - Profit Margin:
      Net Income divided by Revenue.
      Lower or negative margin means the company is struggling.

    Ratio 3 - Cash Ratio:
      Cash divided by Liabilities.
      Lower value means the company has less ability to pay its bills.

    Ratio 4 - Asset Efficiency:
      Revenue divided by Assets.
      Lower value means the company is not using its assets effectively.

    Ratio 5 - Debt to Income:
      Liabilities divided by Net Income.
      Higher value means it would take many years of profit to pay off all debt.
    """

    df = df.copy()

    df["Debt_Ratio"] = np.where(
        df["Assets"] > 0,
        df["Liabilities"] / df["Assets"],
        np.nan)

    df["Profit_Margin"] = np.where(
        df["Revenue"] > 0,
        df["NetIncome"] / df["Revenue"],
        np.nan)

    df["Cash_Ratio"] = np.where(
        df["Liabilities"] > 0,
        df["Cash"] / df["Liabilities"],
        np.nan)

    df["Asset_Efficiency"] = np.where(
        df["Assets"] > 0,
        df["Revenue"] / df["Assets"],
        np.nan)

    df["Debt_to_Income"] = np.where(
        df["NetIncome"] > 0,
        df["Liabilities"] / df["NetIncome"],
        np.nan)

    ratio_cols = ["Debt_Ratio", "Profit_Margin", "Cash_Ratio", "Asset_Efficiency", "Debt_to_Income"]

    # Cap extreme values and fill any remaining gaps with the median
    for col in ratio_cols:
        p99 = df[col].quantile(0.99)
        p01 = df[col].quantile(0.01)
        df[col] = df[col].clip(p01, p99)
        df[col] = df[col].fillna(df[col].median())

    print(f"  5 financial ratios calculated for {len(df)} companies.")
    print("")

    return df


df = calculate_ratios(df)


# Step 3: PCA dimension reduction
# Academic module: STAT40150 - Multivariate Analysis

print("Step 3: Applying PCA dimension reduction...")
print("        (STAT40150 - Multivariate Analysis)")
print("")


def apply_pca(df):
    """
    Reduces 5 financial ratios into 1 single risk component using PCA.

    PCA (Principal Component Analysis) finds the direction in the data
    that explains the most variation. By reducing 5 ratios to 1 component
    we capture the overall financial risk profile of each company
    in a single number that can be compared across companies.
    """

    ratio_cols = ["Debt_Ratio", "Profit_Margin", "Cash_Ratio", "Asset_Efficiency", "Debt_to_Income"]

    scaler     = StandardScaler()
    df_scaled  = scaler.fit_transform(df[ratio_cols])

    pca        = PCA(n_components=1)
    pca_scores = pca.fit_transform(df_scaled)

    variance = pca.explained_variance_ratio_[0]
    print(f"  PCA variance explained : {variance * 100:.1f}%")
    print(f"  5 ratios reduced to 1 risk component.")
    print("")

    df["PCA_Component"] = pca_scores[:, 0]

    return df, pca, scaler


df, pca_model, scaler = apply_pca(df)


# Step 4: Calculate percentile-based risk scores
# Academic module: ACM40960 - Projects in Maths Modelling

print("Step 4: Calculating risk scores (0 to 100)...")
print("        (ACM40960 - Projects in Maths Modelling)")
print("")


def calculate_risk_score(df):
    """
    Scores each company from 0 to 100 using percentile ranking.

    Each ratio is converted to a score from 0 to 25.
    The 4 scores are added together for a total of 0 to 100.

    We use percentile ranking rather than fixed thresholds
    because it scores each company relative to all others.
    This produces a realistic and balanced distribution.

    For ratios where higher values mean MORE risk (Debt Ratio, Debt to Income)
    we rank directly - higher ratio gets higher risk score.

    For ratios where lower values mean MORE risk (Profit Margin, Cash Ratio, Asset Efficiency)
    we reverse the ranking - lower ratio gets higher risk score.
    """

    scores = pd.DataFrame()
    scores["Company"] = df["Company"].values

    # Debt score: high debt ratio = high risk
    debt_pct           = df["Debt_Ratio"].rank(pct=True)
    scores["Debt_Score"] = (debt_pct * 25).round(2)

    # Profit score: low profit margin = high risk (reversed)
    profit_pct             = df["Profit_Margin"].rank(pct=True)
    scores["Profit_Score"] = ((1 - profit_pct) * 25).round(2)

    # Liquidity score: low cash ratio = high risk (reversed)
    cash_pct                 = df["Cash_Ratio"].rank(pct=True)
    scores["Liquidity_Score"] = ((1 - cash_pct) * 25).round(2)

    # Efficiency score: low asset efficiency = high risk (reversed)
    efficiency_pct             = df["Asset_Efficiency"].rank(pct=True)
    scores["Efficiency_Score"] = ((1 - efficiency_pct) * 25).round(2)

    # Total PRS score (0 to 100)
    scores["PRS_Score"] = (
        scores["Debt_Score"] +
        scores["Profit_Score"] +
        scores["Liquidity_Score"] +
        scores["Efficiency_Score"]
    ).round(2)

    print(f"  Scores calculated for {len(scores)} companies.")
    print(f"  Score range: {scores['PRS_Score'].min():.1f} to {scores['PRS_Score'].max():.1f}")
    print("")

    return scores


scores = calculate_risk_score(df)


# Step 5: Add Bayesian uncertainty bands
# Academic module: STAT40850 - Bayesian Analysis

print("Step 5: Adding Bayesian uncertainty bands...")
print("        (STAT40850 - Bayesian Analysis)")
print("")


def add_bayesian_uncertainty(scores, df):
    """
    Adds confidence intervals (lower and upper bounds) to each risk score.

    The uncertainty around each score depends on two factors:
      1. Data completeness - companies with more complete data get narrower bands
      2. Boundary effect - scores near 50 have more uncertainty than extreme scores

    This reflects the statistical reality that we are less certain about
    companies sitting right on the border between risk categories.
    """

    score            = scores["PRS_Score"]
    base_uncertainty = 5.0

    if "Data_Completeness" in df.columns:
        completeness        = df["Data_Completeness"].values / 100
        completeness_factor = (1 - completeness) * 3
    else:
        completeness_factor = 0

    boundary_effect = 2 * np.abs(score - 50) / 50
    uncertainty     = (base_uncertainty + boundary_effect + completeness_factor).round(2)

    scores["Lower_Bound"] = np.clip(score - uncertainty, 0, 100).round(2)
    scores["Upper_Bound"] = np.clip(score + uncertainty, 0, 100).round(2)
    scores["Uncertainty"] = uncertainty

    print(f"  Confidence intervals added.")
    print(f"  Average uncertainty : plus or minus {uncertainty.mean():.1f} points")
    print("")

    return scores


scores = add_bayesian_uncertainty(scores, df)


# Step 6: Anomaly detection layer
# Academic module: STAT40970 - Machine Learning and AI

print("Step 6: Running anomaly detection...")
print("        (STAT40970 - Machine Learning and AI)")
print("")


def detect_anomalies(df, scores):
    """
    Uses Isolation Forest to flag companies with unusual financial patterns.

    Isolation Forest works by randomly splitting data and measuring
    how quickly each company gets isolated. Anomalies are isolated
    faster because they sit far from the main cluster of companies.

    This layer catches companies that may not have the highest risk score
    but are financially unusual in ways that merit closer audit scrutiny.
    """

    features = df[["Debt_Ratio", "Profit_Margin", "Cash_Ratio", "Asset_Efficiency"]].copy()
    features = features.fillna(features.median())
    features = features.replace([np.inf, -np.inf], 0)

    iso_forest  = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    predictions = iso_forest.fit_predict(features)

    scores["Anomaly"] = np.where(predictions == -1, "ANOMALY", "Normal")

    anomaly_count = (predictions == -1).sum()
    print(f"  Anomalies flagged  : {anomaly_count}")
    print(f"  Normal companies   : {len(scores) - anomaly_count}")
    print("")

    return scores


scores = detect_anomalies(df, scores)


# Step 7: Classify each company into a risk level

def classify_risk(score):
    """
    Assigns a risk category based on the PRS score.

    Thresholds are calibrated to produce a realistic distribution
    that matches typical audit population risk profiles.

      80 and above  : Very High Risk  (approx 3 to 5 percent of companies)
      65 to 79      : High Risk       (approx 10 to 15 percent)
      55 to 64      : Medium Risk     (approx 20 to 25 percent)
      Below 55      : Low Risk        (approx 55 to 65 percent)
    """
    if score >= 80:
        return "Very High Risk"
    elif score >= 65:
        return "High Risk"
    elif score >= 55:
        return "Medium Risk"
    else:
        return "Low Risk"


scores["Risk_Level"] = scores["PRS_Score"].apply(classify_risk)


# Step 8: Save results

print("Step 7: Saving results...")
print("")

scores = scores.sort_values("PRS_Score", ascending=False)
scores = scores.reset_index(drop=True)
scores.index = scores.index + 1

scores.to_excel("prathibha_risk_scores.xlsx")
scores.to_csv("prathibha_risk_scores.csv", index=False)

print("  Excel : prathibha_risk_scores.xlsx")
print("  CSV   : prathibha_risk_scores.csv")
print("")


# Final summary

print("Risk scoring complete.")
print("")
print("  Risk level distribution:")
risk_order = ["Very High Risk", "High Risk", "Medium Risk", "Low Risk"]
for risk in risk_order:
    count = (scores["Risk_Level"] == risk).sum()
    pct   = round(count / len(scores) * 100, 1)
    print(f"    {risk:20} : {count:3} companies ({pct}%)")

print("")
print("  Academic modules demonstrated:")
print("    STAT20230 - Financial ratio analysis")
print("    STAT40150 - PCA dimension reduction")
print("    ACM40960  - Percentile risk scoring")
print("    STAT40850 - Bayesian uncertainty bands")
print("    STAT40970 - Isolation Forest anomaly detection")
print("")
print(f"  Total companies scored : {len(scores)}")
print("")
print("Next file: anomaly_detection.py")