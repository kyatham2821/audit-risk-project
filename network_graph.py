# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 7 of 8: Network Graph Analysis
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Builds a network graph connecting companies
# that have similar financial risk profiles.
# Companies with the most connections represent
# systemic risks - if one faces financial
# distress, connected companies may follow.
#
# This analysis identifies which companies
# should be prioritised for coordinated
# audit review across an entire sector.
#
# Academic module:
#   STAT40960 - Statistical Network Analysis (Grade: A-)
#
# Input files:
#   companies_clean_data.csv
#   prathibha_risk_scores.csv
#
# Output files:
#   network_graph_results.csv
#   network_graph_results.xlsx
#   network_graph_chart.png
#
# Next file: dashboard.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 7 of 8: Network Graph Analysis")
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


# Step 2: Calculate financial ratios

print("Step 2: Calculating financial ratios...")
print("")


def calculate_ratios(df):
    """
    Calculates 5 financial ratios for each company.
    These ratios are used to measure how similar two companies are.
    Companies with very similar ratio profiles are connected in the network.
    """

    df = df.copy()

    df["Debt_Ratio"] = np.where(df["Assets"] > 0, df["Liabilities"] / df["Assets"], np.nan)
    df["Profit_Margin"] = np.where(df["Revenue"] > 0, df["NetIncome"] / df["Revenue"], np.nan)
    df["Cash_Ratio"] = np.where(df["Liabilities"] > 0, df["Cash"] / df["Liabilities"], np.nan)
    df["Asset_Efficiency"] = np.where(df["Assets"] > 0, df["Revenue"] / df["Assets"], np.nan)
    df["Return_on_Assets"] = np.where(df["Assets"] > 0, df["NetIncome"] / df["Assets"], np.nan)

    ratio_cols = ["Debt_Ratio", "Profit_Margin", "Cash_Ratio", "Asset_Efficiency", "Return_on_Assets"]

    for col in ratio_cols:
        p99       = df[col].quantile(0.99)
        p01       = df[col].quantile(0.01)
        df[col]   = df[col].clip(p01, p99)
        df[col]   = df[col].fillna(df[col].median())

    print(f"  Financial ratios calculated for {len(df)} companies.")
    print("")

    return df, ratio_cols


df, ratio_cols = calculate_ratios(df)


# Step 3: Build a similarity matrix
# Academic module: STAT40960 - Statistical Network Analysis

print("Step 3: Building financial similarity matrix...")
print("        (STAT40960 - Statistical Network Analysis)")
print("")
print("  Calculating how similar each company is to every other company.")
print("  We use cosine similarity which measures the angle between")
print("  two companies in financial ratio space.")
print("  A similarity of 1.0 means identical financial profiles.")
print("")

scaler            = StandardScaler()
X_scaled          = scaler.fit_transform(df[ratio_cols])
similarity_matrix = cosine_similarity(X_scaled)

print(f"  Similarity matrix built. Size: {similarity_matrix.shape}")
print(f"  Every company compared to every other company.")
print("")


# Step 4: Build the network graph using top 50 highest risk companies
# We focus on the highest risk companies as they are most relevant for audit.
# Using all 742 companies would make the graph too crowded to read.

print("Step 4: Building network graph...")
print("")

top_n        = 50
top_companies = df.nlargest(top_n, "PRS_Score").reset_index(drop=True)
top_indices   = df.nlargest(top_n, "PRS_Score").index.tolist()
sub_similarity = similarity_matrix[np.ix_(top_indices, top_indices)]

print(f"  Using the top {top_n} highest risk companies for the network.")
print("")

G = nx.Graph()

# Add nodes
for i, row in top_companies.iterrows():
    G.add_node(row["Company"], risk_score=row["PRS_Score"], risk_level=row["Risk_Level"])

# Add edges - only connect companies above the similarity threshold
threshold = 0.95
for i in range(top_n):
    for j in range(i + 1, top_n):
        sim = sub_similarity[i][j]
        if sim >= threshold:
            company_i = top_companies.iloc[i]["Company"]
            company_j = top_companies.iloc[j]["Company"]
            G.add_edge(company_i, company_j, weight=sim)

print(f"  Network built.")
print(f"  Nodes (companies)    : {G.number_of_nodes()}")
print(f"  Edges (connections)  : {G.number_of_edges()}")
print(f"  Similarity threshold : {threshold}")
print("")


# Step 5: Calculate network metrics

print("Step 5: Calculating network metrics...")
print("")

degree_centrality = nx.degree_centrality(G)

most_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("  Top 10 most connected companies:")
print("  Companies with many connections represent systemic risks.")
print("  If one fails, connected companies may be affected too.")
print("")
for company, centrality in most_connected:
    connections = G.degree(company)
    print(f"    {company[:35]:35} : {connections:3} connections  (centrality: {centrality:.3f})")
print("")

density    = nx.density(G)
components = list(nx.connected_components(G))

print(f"  Network density : {density:.3f}")
print(f"  A density of {density:.3f} means {density*100:.0f}% of all possible connections exist.")
print("")
print(f"  Risk clusters found : {len(components)}")
for i, component in enumerate(components[:5]):
    print(f"    Cluster {i + 1} : {len(component)} companies")
print("")


# Step 6: Save results

print("Step 6: Saving results...")
print("")

network_results = []
for company in G.nodes():
    row = top_companies[top_companies["Company"] == company].iloc[0]
    network_results.append({
        "Company"     : company,
        "PRS_Score"   : row["PRS_Score"],
        "Risk_Level"  : row["Risk_Level"],
        "Connections" : G.degree(company),
        "Centrality"  : round(degree_centrality[company], 4)
    })

network_df = pd.DataFrame(network_results)
network_df = network_df.sort_values("Connections", ascending=False)
network_df = network_df.reset_index(drop=True)
network_df.index = network_df.index + 1

network_df.to_excel("network_graph_results.xlsx")
network_df.to_csv("network_graph_results.csv", index=False)

print("  Excel : network_graph_results.xlsx")
print("  CSV   : network_graph_results.csv")
print("")


# Step 7: Generate charts

print("Step 7: Generating charts...")
print("")


def generate_network_chart(G, top_companies, degree_centrality):
    """
    Produces 2 charts to visualise the network analysis.

    Chart 1: Network graph showing connections between high risk companies
    Chart 2: Bar chart of the top 15 most connected companies
    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        "Audit Risk Intelligence System\nNetwork Graph - Risk Connections",
        fontsize=14,
        fontweight="bold"
    )

    # Chart 1: Network graph
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=2, seed=42) if G.number_of_edges() > 0 else nx.circular_layout(G)

    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        company_row = top_companies[top_companies["Company"] == node]
        if len(company_row) > 0:
            risk = company_row.iloc[0]["Risk_Level"]
            if "Very High" in risk:
                node_colors.append("#DA291C")
                node_sizes.append(300)
            elif "High" in risk:
                node_colors.append("#FFB81C")
                node_sizes.append(200)
            else:
                node_colors.append("#0076A8")
                node_sizes.append(100)
        else:
            node_colors.append("#0076A8")
            node_sizes.append(100)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax1)
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#999999", ax=ax1)

    top10_nodes = {c for c, _ in most_connected[:10]}
    labels      = {node: node[:15] for node in G.nodes() if node in top10_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax1)

    ax1.set_title(f"Risk Network (Top {top_n} Highest Risk Companies)", fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DA291C", label="Very High Risk"),
        Patch(facecolor="#FFB81C", label="High Risk"),
        Patch(facecolor="#0076A8", label="Medium Risk")
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax1.axis("off")

    # Chart 2: Most connected companies
    ax2       = axes[1]
    top15     = network_df.head(15)
    bar_colors = []
    for risk in top15["Risk_Level"]:
        if "Very High" in risk:
            bar_colors.append("#DA291C")
        elif "High" in risk:
            bar_colors.append("#FFB81C")
        else:
            bar_colors.append("#0076A8")

    bars = ax2.barh(top15["Company"].str[:25], top15["Connections"], color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.set_title("Top 15 Most Connected Companies\n(Highest Systemic Risk)", fontweight="bold")
    ax2.set_xlabel("Number of Connections")
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)
    for bar in bars:
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(int(bar.get_width())), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("network_graph_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Chart saved : network_graph_chart.png")
    print("")


generate_network_chart(G, top_companies, degree_centrality)


# Final summary

print("Network graph analysis complete.")
print("")
print(f"  Companies in network : {G.number_of_nodes()}")
print(f"  Connections mapped   : {G.number_of_edges()}")
print(f"  Network density      : {density:.3f}")
print(f"  Risk clusters found  : {len(components)}")
print("")
print("  Files created:")
print("    - network_graph_results.csv")
print("    - network_graph_results.xlsx")
print("    - network_graph_chart.png")
print("")
print("  Academic module demonstrated:")
print("    STAT40960 - Cosine similarity, degree centrality, connected components, network visualisation")
print("")
print("Next file: dashboard.py")