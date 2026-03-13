# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 4 of 8: Anomaly Detection
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Detects companies with unusual financial
# patterns that warrant closer audit scrutiny.
# Uses the Isolation Forest algorithm which
# finds companies that behave very differently
# from all others in the dataset.
#
# In an audit context, an anomaly could mean:
#   - Potential fraud or earnings manipulation
#   - Unusual business model or structure
#   - Accounting irregularities
#   - A data error worth investigating
#
# Academic module: STAT40970 - Machine Learning and AI
#
# Input files:
#   companies_clean_data.csv
#   prathibha_risk_scores.csv
#
# Output files:
#   anomaly_detection_results.csv
#   anomaly_detection_results.xlsx
#   anomaly_detection_chart.png
#
# Next file: clustering.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 4 of 8: Anomaly Detection")
print("")


# Step 1: Load the cleaned data

print("Step 1: Loading clean data...")
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
print("")


# Step 2: Engineer financial ratio features

print("Step 2: Engineering financial ratio features...")
print("")


def engineer_features(df):
    """
    Creates 6 financial ratios as input features for the anomaly detector.

    We use ratios rather than raw values because ratios are scale-independent.
    This means a small company and a large company can be compared fairly.

    Ratio 1 - Debt Ratio        : Liabilities / Assets
    Ratio 2 - Profit Margin     : Net Income / Revenue
    Ratio 3 - Cash Ratio        : Cash / Liabilities
    Ratio 4 - Asset Efficiency  : Revenue / Assets
    Ratio 5 - Return on Assets  : Net Income / Assets
    Ratio 6 - Leverage Ratio    : Liabilities / Net Income
    """

    features = pd.DataFrame()
    features["Company"] = df["Company"]

    features["Debt_Ratio"] = np.where(
        df["Assets"] > 0,
        df["Liabilities"] / df["Assets"],
        np.nan)

    features["Profit_Margin"] = np.where(
        df["Revenue"] > 0,
        df["NetIncome"] / df["Revenue"],
        np.nan)

    features["Cash_Ratio"] = np.where(
        df["Liabilities"] > 0,
        df["Cash"] / df["Liabilities"],
        np.nan)

    features["Asset_Efficiency"] = np.where(
        df["Assets"] > 0,
        df["Revenue"] / df["Assets"],
        np.nan)

    features["Return_on_Assets"] = np.where(
        df["Assets"] > 0,
        df["NetIncome"] / df["Assets"],
        np.nan)

    features["Leverage_Ratio"] = np.where(
        df["NetIncome"] > 0,
        df["Liabilities"] / df["NetIncome"],
        np.nan)

    ratio_cols = ["Debt_Ratio", "Profit_Margin", "Cash_Ratio",
                  "Asset_Efficiency", "Return_on_Assets", "Leverage_Ratio"]

    # Cap extreme values and fill any gaps with the column median
    for col in ratio_cols:
        p99              = features[col].quantile(0.99)
        p01              = features[col].quantile(0.01)
        features[col]    = features[col].clip(p01, p99)
        features[col]    = features[col].fillna(features[col].median())

    print(f"  6 financial ratios created for {len(features)} companies.")
    print("")

    return features, ratio_cols


features, ratio_cols = engineer_features(df)


# Step 3: Standardise the features

print("Step 3: Standardising features...")
print("")
print("  Standardisation puts all ratios on the same scale.")
print("  Without this, ratios with larger ranges would dominate the model.")
print("")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features[ratio_cols])

print(f"  All features scaled to mean=0, standard deviation=1.")
print("")


# Step 4: Train the Isolation Forest model
# Academic module: STAT40970 - Machine Learning and AI

print("Step 4: Training Isolation Forest...")
print("        (STAT40970 - Machine Learning and AI)")
print("")
print("  How Isolation Forest works:")
print("  The algorithm builds 200 decision trees.")
print("  Each tree randomly splits the data into smaller groups.")
print("  Normal companies are hard to isolate because they sit")
print("  close together in the middle of the data.")
print("  Anomalies are easy to isolate because they sit far")
print("  away from the rest - they get separated quickly.")
print("")
print("  We set contamination to 0.05 which means we expect")
print("  approximately 5 percent of companies to be anomalies.")
print("")

iso_forest = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=200,
    max_samples="auto"
)

predictions    = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

# The model returns -1 for anomalies and 1 for normal companies
features["Anomaly_Flag"]  = np.where(predictions == -1, "ANOMALY", "Normal")
features["Anomaly_Score"] = anomaly_scores.round(4)

anomaly_count = (predictions == -1).sum()
normal_count  = (predictions == 1).sum()

print(f"  Anomalies detected : {anomaly_count}")
print(f"  Normal companies   : {normal_count}")
print(f"  Anomaly rate       : {round(anomaly_count / len(features) * 100, 1)}%")
print("")


# Step 5: Review the top anomalies

print("Step 5: Reviewing the most anomalous companies...")
print("")

anomalies = features[features["Anomaly_Flag"] == "ANOMALY"].copy()
anomalies = anomalies.sort_values("Anomaly_Score", ascending=True)

print("  Top 10 most anomalous companies:")
print("")

display_cols = ["Company", "Anomaly_Score", "Debt_Ratio", "Profit_Margin", "Return_on_Assets"]
print(anomalies[display_cols].head(10).to_string(index=False))
print("")


# Step 6: Apply PCA for visualisation purposes
# We reduce 6 features to 2 components so we can plot the results in 2D.

print("Step 6: Applying PCA for visualisation...")
print("")

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

variance_explained = pca.explained_variance_ratio_ * 100
print(f"  PCA Component 1 : {variance_explained[0]:.1f}% variance")
print(f"  PCA Component 2 : {variance_explained[1]:.1f}% variance")
print(f"  Total explained : {sum(variance_explained):.1f}%")
print("")


# Step 7: Combine with Prathibha Risk Scores

print("Step 7: Merging with risk scores...")
print("")

results = df[["Company"]].copy()
results = results.merge(
    features[["Company", "Anomaly_Flag", "Anomaly_Score",
               "Debt_Ratio", "Profit_Margin", "Cash_Ratio",
               "Asset_Efficiency", "Return_on_Assets", "Leverage_Ratio"]],
    on="Company",
    how="left"
)

try:
    prs     = pd.read_csv("prathibha_risk_scores.csv")
    results = results.merge(prs[["Company", "PRS_Score", "Risk_Level"]], on="Company", how="left")
    print(f"  Merged with Prathibha Risk Scores.")

    # Find companies flagged as anomalies AND high risk by the PRS model
    dual_flagged = results[
        (results["Anomaly_Flag"] == "ANOMALY") &
        (results["Risk_Level"].isin(["High Risk", "Very High Risk"]))
    ]
    print(f"  Flagged by both models : {len(dual_flagged)} companies")
    print(f"  These are the highest priority companies for audit review.")

except FileNotFoundError:
    print("  Risk scores file not found. Run risk_score.py first.")

print("")


# Step 8: Save results

print("Step 8: Saving results...")
print("")

results = results.sort_values(["Anomaly_Flag", "Anomaly_Score"], ascending=[False, True])
results = results.reset_index(drop=True)
results.index = results.index + 1

results.to_excel("anomaly_detection_results.xlsx")
results.to_csv("anomaly_detection_results.csv", index=False)

print("  Excel : anomaly_detection_results.xlsx")
print("  CSV   : anomaly_detection_results.csv")
print("")


# Step 9: Generate charts

print("Step 9: Generating charts...")
print("")


def generate_charts(X_pca, features, predictions, anomaly_scores, variance_explained):
    """
    Produces 3 charts to visualise the anomaly detection results.

    Chart 1: PCA scatter plot showing normal vs anomaly companies
    Chart 2: Distribution of anomaly scores
    Chart 3: Box plot comparing debt ratios for normal vs anomaly companies
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Audit Risk Intelligence System\nAnomaly Detection - Isolation Forest",
        fontsize=14,
        fontweight="bold"
    )

    # Chart 1: PCA scatter
    ax1 = axes[0]
    ax1.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1],
                c="#0076A8", alpha=0.4, s=15, label=f"Normal ({normal_count})")
    ax1.scatter(X_pca[predictions == -1, 0], X_pca[predictions == -1, 1],
                c="#DA291C", alpha=0.8, s=40, marker="X", label=f"Anomaly ({anomaly_count})")
    ax1.set_title(f"PCA Scatter Plot\n({variance_explained[0]:.0f}% + {variance_explained[1]:.0f}% variance)",
                  fontweight="bold")
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Chart 2: Anomaly score distribution
    ax2 = axes[1]
    ax2.hist(anomaly_scores[predictions == 1], bins=30, color="#0076A8", alpha=0.7, label="Normal", edgecolor="white")
    ax2.hist(anomaly_scores[predictions == -1], bins=10, color="#DA291C", alpha=0.9, label="Anomaly", edgecolor="white")
    ax2.set_title("Anomaly Score Distribution", fontweight="bold")
    ax2.set_xlabel("Anomaly Score")
    ax2.set_ylabel("Number of Companies")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Chart 3: Debt ratio comparison
    ax3 = axes[2]
    normal_debt  = features[features["Anomaly_Flag"] == "Normal"]["Debt_Ratio"]
    anomaly_debt = features[features["Anomaly_Flag"] == "ANOMALY"]["Debt_Ratio"]
    ax3.boxplot([normal_debt, anomaly_debt], labels=["Normal", "Anomaly"],
                patch_artist=True,
                boxprops=dict(facecolor="#0076A8", alpha=0.6),
                medianprops=dict(color="white", linewidth=2))
    ax3.set_title("Debt Ratio: Normal vs Anomaly", fontweight="bold")
    ax3.set_ylabel("Debt Ratio")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("anomaly_detection_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Chart saved : anomaly_detection_chart.png")
    print("")


generate_charts(X_pca, features, predictions, anomaly_scores, variance_explained)


# Final summary

print("Anomaly detection complete.")
print("")
print(f"  Total companies analysed : {len(results)}")
print(f"  Anomalies flagged        : {anomaly_count}")
print(f"  Normal companies         : {normal_count}")
print("")
print("  Files created:")
print("    - anomaly_detection_results.csv")
print("    - anomaly_detection_results.xlsx")
print("    - anomaly_detection_chart.png")
print("")
print("  Academic module demonstrated:")
print("    STAT40970 - Isolation Forest, unsupervised anomaly detection, PCA visualisation")
print("")
print("Next file: clustering.py")