# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 6 of 8: Predictive Model - Random Forest
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Trains a Random Forest classifier to predict
# whether a company is High Risk or Low Risk
# based on its financial ratios.
#
# The model learns patterns from the Prathibha
# Risk Scores and can then predict risk for
# new companies not seen during training.
#
# Academic modules:
#   STAT20230 - Modern Regression Analysis (model evaluation)
#   STAT40970 - Machine Learning and AI (Random Forest)
#   STAT30270 - Statistical Machine Learning (train/test split)
#
# Input files:
#   companies_clean_data.csv
#   prathibha_risk_scores.csv
#
# Output files:
#   predictive_model_results.csv
#   predictive_model_results.xlsx
#   predictive_model_chart.png
#
# Next file: network_graph.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 6 of 8: Predictive Model - Random Forest")
print("")


# Step 1: Load the data

print("Step 1: Loading data...")
print("")

df = pd.read_csv("companies_clean_data.csv")
df = df.rename(columns={
    "Revenue ($B)"           : "Revenue",
    "Net Income ($B)"        : "NetIncome",
    "Total Assets ($B)"      : "Assets",
    "Total Liabilities ($B)" : "Liabilities",
    "Cash ($B)"              : "Cash"
})

prs = pd.read_csv("prathibha_risk_scores.csv")
df  = df.merge(prs[["Company", "PRS_Score", "Risk_Level"]], on="Company", how="inner")

print(f"  Companies loaded : {len(df)}")
print("")


# Step 2: Prepare features and labels

print("Step 2: Preparing features and labels...")
print("")


def prepare_data(df):
    """
    Creates the feature matrix (X) and label vector (y) for the model.

    Features are 6 financial ratios calculated from the raw data.
    Labels are binary:
      1 = High Risk or Very High Risk
      0 = Medium Risk or Low Risk

    We use binary labels rather than 4 categories because
    audit teams need a clear flag - either a company needs
    priority attention or it does not.
    """

    df["Debt_Ratio"] = np.where(df["Assets"] > 0, df["Liabilities"] / df["Assets"], np.nan)
    df["Profit_Margin"] = np.where(df["Revenue"] > 0, df["NetIncome"] / df["Revenue"], np.nan)
    df["Cash_Ratio"] = np.where(df["Liabilities"] > 0, df["Cash"] / df["Liabilities"], np.nan)
    df["Asset_Efficiency"] = np.where(df["Assets"] > 0, df["Revenue"] / df["Assets"], np.nan)
    df["Return_on_Assets"] = np.where(df["Assets"] > 0, df["NetIncome"] / df["Assets"], np.nan)
    df["Leverage_Ratio"] = np.where(df["NetIncome"] > 0, df["Liabilities"] / df["NetIncome"], np.nan)

    feature_cols = ["Debt_Ratio", "Profit_Margin", "Cash_Ratio",
                    "Asset_Efficiency", "Return_on_Assets", "Leverage_Ratio"]

    for col in feature_cols:
        p99      = df[col].quantile(0.99)
        p01      = df[col].quantile(0.01)
        df[col]  = df[col].clip(p01, p99)
        df[col]  = df[col].fillna(df[col].median())

    # Assign binary labels
    # High Risk and Very High Risk = 1 (needs audit attention)
    # Medium Risk and Low Risk = 0 (lower priority)
    df["Label"] = np.where(
        df["Risk_Level"].isin(["High Risk", "Very High Risk"]),
        1,
        0
    )

    X = df[feature_cols]
    y = df["Label"]

    high_risk = y.sum()
    low_risk  = (y == 0).sum()

    print(f"  Feature columns    : {len(feature_cols)}")
    print(f"  High Risk labels   : {high_risk}")
    print(f"  Low Risk labels    : {low_risk}")
    print(f"  Total companies    : {len(df)}")
    print("")

    return X, y, feature_cols, df


X, y, feature_cols, df = prepare_data(df)


# Step 3: Split data into training and testing sets
# Academic module: STAT30270 - Statistical Machine Learning

print("Step 3: Splitting data into training and testing sets...")
print("        (STAT30270 - Statistical Machine Learning)")
print("")
print("  80 percent of companies used for training the model.")
print("  20 percent held back to test how accurate the model is.")
print("  The model never sees the test set during training.")
print("")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"  Training set : {len(X_train)} companies")
print(f"  Testing set  : {len(X_test)} companies")
print("")


# Step 4: Train the Random Forest model
# Academic module: STAT40970 - Machine Learning and AI

print("Step 4: Training Random Forest model...")
print("        (STAT40970 - Machine Learning and AI)")
print("")
print("  Random Forest builds 200 decision trees.")
print("  Each tree independently votes on whether a company is High Risk.")
print("  The majority vote across all 200 trees is the final prediction.")
print("  Using many trees makes the model more stable and accurate.")
print("")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

print(f"  Random Forest trained. 200 decision trees built.")
print("")


# Step 5: Evaluate model performance
# Academic module: STAT20230 - Modern Regression Analysis

print("Step 5: Evaluating model performance...")
print("        (STAT20230 - Modern Regression Analysis)")
print("")

y_pred = rf_model.predict(X_test)

# Handle the case where only one class appears in predictions
proba = rf_model.predict_proba(X_test)
if proba.shape[1] == 1:
    y_prob = proba[:, 0]
else:
    y_prob = proba[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"  Model accuracy : {accuracy * 100:.1f}%")
print("")

print("  Detailed performance report:")
report = classification_report(y_test, y_pred, target_names=["Low/Med Risk", "High Risk"])
print(report)

cm = confusion_matrix(y_test, y_pred)
print("  Confusion matrix:")
print(f"    Correctly identified Low Risk  : {cm[0][0]}")
print(f"    Incorrectly flagged High Risk  : {cm[0][1]}")
print(f"    Incorrectly missed High Risk   : {cm[1][0]}")
print(f"    Correctly identified High Risk : {cm[1][1]}")
print("")


# Step 6: Feature importance

print("Step 6: Calculating feature importance...")
print("")
print("  Which financial ratios matter most for predicting audit risk?")
print("")

importances   = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature"    : feature_cols,
    "Importance" : importances
}).sort_values("Importance", ascending=False)

for _, row in importance_df.iterrows():
    bar = "=" * int(row["Importance"] * 100)
    print(f"  {row['Feature']:20} : {row['Importance']:.3f}  {bar}")
print("")


# Step 7: Predict for all companies

print("Step 7: Predicting risk for all companies...")
print("")

all_predictions   = rf_model.predict(X)
all_probabilities = rf_model.predict_proba(X)[:, 1]

df["RF_Prediction"]  = np.where(all_predictions == 1, "High Risk", "Low Risk")
df["RF_Probability"] = (all_probabilities * 100).round(1)

high_count = (all_predictions == 1).sum()
low_count  = (all_predictions == 0).sum()

print(f"  Predicted High Risk : {high_count}")
print(f"  Predicted Low Risk  : {low_count}")
print("")


# Step 8: Save results

print("Step 8: Saving results...")
print("")

results = df[["Company", "PRS_Score", "Risk_Level", "RF_Prediction", "RF_Probability",
              "Debt_Ratio", "Profit_Margin", "Cash_Ratio", "Asset_Efficiency", "Return_on_Assets"]].copy()

results = results.sort_values("RF_Probability", ascending=False)
results = results.reset_index(drop=True)
results.index = results.index + 1

results.to_excel("predictive_model_results.xlsx")
results.to_csv("predictive_model_results.csv", index=False)

print("  Excel : predictive_model_results.xlsx")
print("  CSV   : predictive_model_results.csv")
print("")


# Step 9: Generate charts

print("Step 9: Generating charts...")
print("")


def generate_charts(importance_df, cm, y_test, y_prob, all_probabilities):
    """
    Produces 3 charts to visualise the model results.

    Chart 1: Feature importance showing which ratios drive audit risk
    Chart 2: Confusion matrix showing model accuracy
    Chart 3: Distribution of high risk probabilities across all companies
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Audit Risk Intelligence System\nRandom Forest Predictive Model",
        fontsize=14,
        fontweight="bold"
    )

    # Chart 1: Feature importance
    ax1    = axes[0]
    colors = ["#DA291C" if i == 0 else "#0076A8" for i in range(len(importance_df))]
    bars   = ax1.barh(importance_df["Feature"], importance_df["Importance"], color=colors, edgecolor="white", alpha=0.85)
    ax1.set_title("Feature Importance\n(Most Influential Ratios)", fontweight="bold")
    ax1.set_xlabel("Importance Score")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, importance_df["Importance"]):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    # Chart 2: Confusion matrix
    ax2 = axes[1]
    ax2.imshow(cm, interpolation="nearest", cmap="Blues")
    ax2.set_title("Confusion Matrix\n(Model Accuracy)", fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Low Risk", "High Risk"])
    ax2.set_yticklabels(["Low Risk", "High Risk"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Chart 3: Risk probability distribution
    ax3 = axes[2]
    ax3.hist(all_probabilities[all_predictions == 0] * 100, bins=30, color="#0076A8", alpha=0.7, label="Low Risk", edgecolor="white")
    ax3.hist(all_probabilities[all_predictions == 1] * 100, bins=30, color="#DA291C", alpha=0.7, label="High Risk", edgecolor="white")
    ax3.set_title("Risk Probability Distribution", fontweight="bold")
    ax3.set_xlabel("High Risk Probability (%)")
    ax3.set_ylabel("Number of Companies")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictive_model_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Chart saved : predictive_model_chart.png")
    print("")


generate_charts(importance_df, cm, y_test, y_prob, all_probabilities)


# Final summary

print("Predictive model complete.")
print("")
print(f"  Companies analysed : {len(results)}")
print(f"  Model accuracy     : {accuracy * 100:.1f}%")
print("")
print("  Files created:")
print("    - predictive_model_results.csv")
print("    - predictive_model_results.xlsx")
print("    - predictive_model_chart.png")
print("")
print("  Academic modules demonstrated:")
print("    STAT20230 - Model evaluation and classification report")
print("    STAT40970 - Random Forest with 200 decision trees")
print("    STAT30270 - Train/test split and model validation")
print("")
print("Next file: network_graph.py")