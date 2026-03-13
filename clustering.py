# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 5 of 8: PCA and K-Means Clustering
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Groups companies into risk clusters based on
# financial similarity. Companies in the same
# cluster share similar financial profiles and
# therefore similar audit risk characteristics.
#
# This helps auditors prioritise their work by
# focusing on entire risk clusters rather than
# reviewing companies one by one.
#
# Academic modules:
#   STAT40150 - Multivariate Analysis (PCA)
#   STAT40970 - Machine Learning and AI (K-Means)
#
# Input files:
#   companies_clean_data.csv
#   prathibha_risk_scores.csv
#
# Output files:
#   clustering_results.csv
#   clustering_results.xlsx
#   clustering_chart.png
#
# Next file: predictive_model.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 5 of 8: PCA and K-Means Clustering")
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


# Step 2: Build financial ratio features

print("Step 2: Engineering financial ratio features...")
print("")


def engineer_features(df):
    """
    Creates 5 financial ratios for clustering.

    We cluster on ratios rather than raw financial figures because
    ratios allow fair comparison between companies of different sizes.
    """

    features = pd.DataFrame()
    features["Company"] = df["Company"]

    features["Debt_Ratio"] = np.where(
        df["Assets"] > 0, df["Liabilities"] / df["Assets"], np.nan)

    features["Profit_Margin"] = np.where(
        df["Revenue"] > 0, df["NetIncome"] / df["Revenue"], np.nan)

    features["Cash_Ratio"] = np.where(
        df["Liabilities"] > 0, df["Cash"] / df["Liabilities"], np.nan)

    features["Asset_Efficiency"] = np.where(
        df["Assets"] > 0, df["Revenue"] / df["Assets"], np.nan)

    features["Return_on_Assets"] = np.where(
        df["Assets"] > 0, df["NetIncome"] / df["Assets"], np.nan)

    ratio_cols = ["Debt_Ratio", "Profit_Margin", "Cash_Ratio", "Asset_Efficiency", "Return_on_Assets"]

    for col in ratio_cols:
        p99           = features[col].quantile(0.99)
        p01           = features[col].quantile(0.01)
        features[col] = features[col].clip(p01, p99)
        features[col] = features[col].fillna(features[col].median())

    print(f"  5 financial ratios created for {len(features)} companies.")
    print("")

    return features, ratio_cols


features, ratio_cols = engineer_features(df)


# Step 3: Standardise features

print("Step 3: Standardising features...")
print("")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features[ratio_cols])

print(f"  Features standardised to mean=0, standard deviation=1.")
print("")


# Step 4: Reduce to 2 dimensions using PCA
# Academic module: STAT40150 - Multivariate Analysis

print("Step 4: Applying PCA dimension reduction...")
print("        (STAT40150 - Multivariate Analysis)")
print("")
print("  We reduce 5 financial ratios to 2 components")
print("  so we can visualise the clusters in a 2D scatter plot.")
print("")

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

var = pca.explained_variance_ratio_ * 100
print(f"  Component 1 : {var[0]:.1f}% variance explained")
print(f"  Component 2 : {var[1]:.1f}% variance explained")
print(f"  Total       : {sum(var):.1f}% variance explained")
print("")


# Step 5: Find the optimal number of clusters using the Elbow Method
# Academic module: STAT40970 - Machine Learning and AI

print("Step 5: Finding optimal number of clusters...")
print("        (STAT40970 - Machine Learning and AI)")
print("")
print("  Testing K from 2 to 8 clusters...")
print("")

inertias    = []
silhouettes = []
k_range     = range(2, 9)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_)
    silhouettes.append(sil)
    print(f"  K={k} : Inertia={kmeans.inertia_:.1f}  Silhouette={sil:.3f}")

best_k = k_range[silhouettes.index(max(silhouettes))]
print("")
print(f"  Optimal clusters : K = {best_k}")
print(f"  Best silhouette  : {max(silhouettes):.3f}")
print("")


# Step 6: Apply K-Means clustering with the optimal K

print("Step 6: Applying K-Means clustering...")
print("")

kmeans        = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
features["Cluster"] = cluster_labels

print(f"  K-Means complete. {best_k} clusters created.")
print("")
print("  Cluster sizes:")
for c in sorted(features["Cluster"].unique()):
    count = (features["Cluster"] == c).sum()
    pct   = round(count / len(features) * 100, 1)
    print(f"    Cluster {c} : {count:3} companies ({pct}%)")
print("")


# Step 7: Profile each cluster by average financial ratios

print("Step 7: Profiling each cluster...")
print("")


def profile_clusters(features, ratio_cols):
    """
    Calculates the average financial ratios for each cluster.
    This tells us what kind of companies are in each cluster
    and lets us assign a meaningful risk label.

    A cluster is High Risk if:
      - Average debt ratio is above 0.7
      - Average profit margin is below 0.05

    A cluster is Medium Risk if:
      - Average debt ratio is above 0.5 OR profit margin is below 0.10
    """

    profiles = features.groupby("Cluster")[ratio_cols].mean().round(3)

    print("  Average financial ratios per cluster:")
    print("")
    print(profiles.to_string())
    print("")

    cluster_labels_map = {}
    for cluster in profiles.index:
        debt   = profiles.loc[cluster, "Debt_Ratio"]
        profit = profiles.loc[cluster, "Profit_Margin"]
        roa    = profiles.loc[cluster, "Return_on_Assets"]

        if debt > 0.7 and profit < 0.05:
            label = "High Risk Cluster"
        elif debt > 0.5 or profit < 0.10:
            label = "Medium Risk Cluster"
        else:
            label = "Low Risk Cluster"

        cluster_labels_map[cluster] = label
        print(f"  Cluster {cluster} : {label}")
        print(f"    Average Debt Ratio    : {debt:.3f}")
        print(f"    Average Profit Margin : {profit:.3f}")
        print(f"    Average Return/Assets : {roa:.3f}")
        print("")

    return cluster_labels_map


cluster_labels_map = profile_clusters(features, ratio_cols)
features["Cluster_Label"] = features["Cluster"].map(cluster_labels_map)


# Step 8: Combine with Prathibha Risk Scores

print("Step 8: Combining with risk scores...")
print("")

results = df[["Company"]].copy()
results = results.merge(
    features[["Company", "Cluster", "Cluster_Label",
               "Debt_Ratio", "Profit_Margin", "Cash_Ratio",
               "Asset_Efficiency", "Return_on_Assets"]],
    on="Company",
    how="left"
)

try:
    prs     = pd.read_csv("prathibha_risk_scores.csv")
    results = results.merge(prs[["Company", "PRS_Score", "Risk_Level"]], on="Company", how="left")
    print(f"  Merged with Prathibha Risk Scores.")
except FileNotFoundError:
    print("  Risk scores file not found. Run risk_score.py first.")

print("")


# Step 9: Save results

print("Step 9: Saving results...")
print("")

results = results.sort_values(["Cluster", "Company"])
results = results.reset_index(drop=True)
results.index = results.index + 1

results.to_excel("clustering_results.xlsx")
results.to_csv("clustering_results.csv", index=False)

print("  Excel : clustering_results.xlsx")
print("  CSV   : clustering_results.csv")
print("")


# Step 10: Generate charts

print("Step 10: Generating charts...")
print("")


def generate_charts(X_pca, features, inertias, silhouettes, k_range, best_k, var, cluster_labels_map):
    """
    Produces 3 charts to visualise the clustering results.

    Chart 1: Elbow method showing how we chose K
    Chart 2: PCA scatter plot coloured by cluster
    Chart 3: Bar chart showing companies per cluster
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Audit Risk Intelligence System\nPCA and K-Means Clustering",
        fontsize=14,
        fontweight="bold"
    )

    cluster_colors = ["#0076A8", "#86BC25", "#DA291C", "#FFB81C", "#26890D", "#6D2077"]

    # Chart 1: Elbow method
    ax1      = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(list(k_range), inertias, "o-", color="#0076A8", linewidth=2, label="Inertia")
    ax1_twin.plot(list(k_range), silhouettes, "s--", color="#DA291C", linewidth=2, label="Silhouette")
    ax1.axvline(x=best_k, color="#86BC25", linestyle=":", linewidth=2, label=f"Best K={best_k}")
    ax1.set_title("Elbow Method\n(Finding Optimal Number of Clusters)", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia", color="#0076A8")
    ax1_twin.set_ylabel("Silhouette Score", color="#DA291C")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Chart 2: PCA scatter by cluster
    ax2 = axes[1]
    for cluster in sorted(features["Cluster"].unique()):
        mask        = features["Cluster"] == cluster
        label       = cluster_labels_map.get(cluster, f"Cluster {cluster}")
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=cluster_colors[cluster % len(cluster_colors)],
                    alpha=0.6, s=20, label=f"C{cluster}: {label}")
    ax2.set_title(f"PCA Clusters (K={best_k})\n({var[0]:.0f}% + {var[1]:.0f}% variance)", fontweight="bold")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Chart 3: Companies per cluster
    ax3         = axes[2]
    cluster_counts = features["Cluster"].value_counts().sort_index()
    clean_labels   = [cluster_labels_map.get(c, f"C{c}") for c in cluster_counts.index]
    bar_colors     = []
    for c in cluster_counts.index:
        lbl = cluster_labels_map.get(c, "")
        if "High" in lbl:
            bar_colors.append("#DA291C")
        elif "Medium" in lbl:
            bar_colors.append("#FFB81C")
        else:
            bar_colors.append("#86BC25")

    bars = ax3.bar(clean_labels, cluster_counts.values, color=bar_colors, edgecolor="white", alpha=0.85)
    ax3.set_title("Companies per Cluster", fontweight="bold")
    ax3.set_ylabel("Number of Companies")
    ax3.set_xticklabels(clean_labels, rotation=20, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(int(bar.get_height())), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("clustering_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Chart saved : clustering_chart.png")
    print("")


generate_charts(X_pca, features, inertias, silhouettes, k_range, best_k, var, cluster_labels_map)


# Final summary

print("Clustering complete.")
print("")
print(f"  Companies analysed : {len(results)}")
print(f"  Clusters created   : {best_k}")
print("")
print("  Files created:")
print("    - clustering_results.csv")
print("    - clustering_results.xlsx")
print("    - clustering_chart.png")
print("")
print("  Academic modules demonstrated:")
print("    STAT40150 - PCA dimension reduction")
print("    STAT40970 - K-Means clustering, Elbow method, Silhouette scoring")
print("")
print("Next file: predictive_model.py")