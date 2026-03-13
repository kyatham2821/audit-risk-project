# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 2 of 8: Data Preprocessing
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Cleans and validates the raw financial data
# before any models are built. Handles missing
# values, removes duplicates, caps outliers,
# validates business rules, and scores each
# company for data completeness.
#
# Academic modules:
#   STAT41040 - Principles of Probability and Stats
#   STAT20230 - Modern Regression Analysis
#   ACM40960  - Projects in Maths Modelling
#
# Input file  : companies_financial_data.csv
# Output files:
#   companies_clean_data.csv
#   companies_clean_data.xlsx
#   data_quality_report.png
#
# Next file: risk_score.py
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 2 of 8: Data Preprocessing")
print("")


# Step 1: Load the raw data file

print("Step 1: Loading raw data...")
print("")

df_raw = pd.read_csv("companies_financial_data.csv")

print(f"  Companies loaded : {len(df_raw)}")
print(f"  Columns available: {list(df_raw.columns)}")
print("")

# Keep a copy of the original for comparison
df = df_raw.copy()

# Rename columns to simpler names for internal use
df = df.rename(columns={
    "Revenue ($B)"           : "Revenue",
    "Net Income ($B)"        : "NetIncome",
    "Total Assets ($B)"      : "Assets",
    "Total Liabilities ($B)" : "Liabilities",
    "Cash ($B)"              : "Cash"
})

numeric_cols = ["Revenue", "NetIncome", "Assets", "Liabilities", "Cash"]


# Step 2: Run a full data quality check

print("Step 2: Running data quality checks...")
print("")


def generate_quality_report(df):
    """
    Checks the data for common problems before cleaning.

    We check for:
      - Missing values per column
      - Duplicate company entries
      - Negative values where not expected
      - Data ranges and distributions
      - Statistical outliers using the IQR method

    Returns an overall quality score from 0 to 100.
    """

    print(f"  Total companies : {len(df)}")
    print(f"  Total columns   : {len(df.columns)}")
    print("")

    # Check 1: Missing values
    print("  Missing values check:")
    total_missing = 0
    for col in numeric_cols:
        missing = df[col].isna().sum()
        pct     = round(missing / len(df) * 100, 1)
        total_missing += missing
        status  = "[PASS]" if missing == 0 else "[WARNING]"
        print(f"    {status} {col:20} : {missing:3} missing ({pct}%)")
    print(f"    Total missing values : {total_missing}")
    print("")

    # Check 2: Duplicate companies
    print("  Duplicate company check:")
    dupes  = df["Company"].duplicated().sum()
    status = "[PASS]" if dupes == 0 else "[WARNING]"
    print(f"    {status} Duplicate companies : {dupes}")
    print("")

    # Check 3: Negative values
    # Revenue, Assets and Liabilities should always be positive.
    # Net Income can be negative - that just means the company made a loss.
    print("  Negative values check:")
    always_positive = ["Revenue", "Assets", "Liabilities"]
    for col in always_positive:
        neg    = (df[col] < 0).sum()
        status = "[PASS]" if neg == 0 else "[WARNING]"
        print(f"    {status} {col:20} : {neg:3} negative values")
    neg_income = (df["NetIncome"] < 0).sum()
    print(f"    [INFO] NetIncome : {neg_income} negative (companies making losses - this is normal)")
    print("")

    # Check 4: Data ranges
    print("  Data ranges (in billions):")
    for col in numeric_cols:
        mn   = df[col].min()
        mx   = df[col].max()
        mean = df[col].mean()
        print(f"    {col:20} : min={mn:.1f}  max={mx:.1f}  mean={mean:.1f}")
    print("")

    # Check 5: Outliers using the IQR method
    # A value is an outlier if it falls more than 1.5x the IQR above Q3 or below Q1.
    print("  Outlier check (IQR method):")
    for col in numeric_cols:
        Q1   = df[col].quantile(0.25)
        Q3   = df[col].quantile(0.75)
        IQR  = Q3 - Q1
        low  = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        out  = ((df[col] < low) | (df[col] > high)).sum()
        status = "[PASS]" if out == 0 else "[WARNING]"
        print(f"    {status} {col:20} : {out:3} outliers detected")
    print("")

    # Overall quality score
    total_cells   = len(df) * len(numeric_cols)
    quality_score = round((1 - total_missing / total_cells) * 100, 1)

    print(f"  Data quality score : {quality_score}%")
    if quality_score >= 90:
        print("  Status             : EXCELLENT")
    elif quality_score >= 75:
        print("  Status             : GOOD")
    else:
        print("  Status             : NEEDS ATTENTION")
    print("")

    return quality_score


quality_score = generate_quality_report(df)


# Step 3: Fill missing values

print("Step 3: Handling missing values...")
print("")


def handle_missing_values(df):
    """
    Fills missing values using appropriate strategies.

    Strategy for each column:
      Revenue, Assets, Liabilities, NetIncome -> median
        We use median rather than mean because financial data
        is heavily skewed by very large companies like Apple.
        Mean would be pulled too high by these outliers.

      Cash -> lower quartile (25th percentile)
        We use a conservative estimate for cash
        to avoid overstating liquidity.
    """

    before = df[numeric_cols].isna().sum().sum()

    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if col == "Cash":
                fill_val = df[col].quantile(0.25)
                strategy = "lower quartile"
            else:
                fill_val = df[col].median()
                strategy = "median"
            df[col] = df[col].fillna(fill_val)
            print(f"  {col:20} : filled {missing_count} values using {strategy} ({fill_val:.2f})")
        else:
            print(f"  {col:20} : no missing values found")

    after = df[numeric_cols].isna().sum().sum()
    print("")
    print(f"  Missing before : {before}")
    print(f"  Missing after  : {after}")
    print("")

    return df


df = handle_missing_values(df)


# Step 4: Remove duplicate companies

print("Step 4: Removing duplicate companies...")
print("")


def handle_duplicates(df):
    """
    Removes duplicate company entries.
    Where duplicates exist, we keep the first occurrence.
    """

    before     = len(df)
    duplicates = df[df["Company"].duplicated(keep=False)]

    if len(duplicates) > 0:
        print(f"  Found {len(duplicates)} duplicate entries:")
        for company in duplicates["Company"].unique():
            print(f"    - {company}")
        df = df.drop_duplicates(subset="Company", keep="first")
        print(f"  Duplicates removed.")
    else:
        print(f"  No duplicate companies found.")

    after = len(df)
    print(f"  Companies before : {before}")
    print(f"  Companies after  : {after}")
    print("")

    return df


df = handle_duplicates(df)


# Step 5: Cap extreme outliers using Winsorisation

print("Step 5: Capping outliers using Winsorisation...")
print("")
print("  Winsorisation caps extreme values at the 1st and 99th percentile.")
print("  This limits the distorting effect of extreme values")
print("  without removing those companies from the dataset entirely.")
print("")


def handle_outliers(df):
    """
    Winsorisation: caps values at the 1st and 99th percentile.

    This is better than removing outliers completely because:
      - We keep all companies in the dataset
      - We reduce the influence of extreme values
      - The model treats all companies fairly
    """

    for col in numeric_cols:
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        outliers_before = ((df[col] < p01) | (df[col] > p99)).sum()
        df[col] = df[col].clip(p01, p99)
        print(f"  {col:20} : {outliers_before:3} values capped  [{p01:.2f} to {p99:.2f}]")

    print("")
    print("  Winsorisation complete.")
    print("")

    return df


df = handle_outliers(df)


# Step 6: Validate business rules

print("Step 6: Validating business rules...")
print("")


def validate_data(df):
    """
    Checks that all data satisfies basic financial logic.

    Rules we enforce:
      Rule 1: Total Assets must be positive (a company must own something)
      Rule 2: Revenue must be positive (a company must generate some income)
      Rule 3: Liabilities cannot exceed 10 times total assets
      Rule 4: Company name cannot be missing
    """

    issues = []

    rule1 = (df["Assets"] <= 0).sum()
    if rule1 > 0:
        issues.append(f"  {rule1} companies with zero or negative assets removed")
        df = df[df["Assets"] > 0]

    rule2 = (df["Revenue"] <= 0).sum()
    if rule2 > 0:
        issues.append(f"  {rule2} companies with zero or negative revenue removed")
        df = df[df["Revenue"] > 0]

    rule3 = (df["Liabilities"] > df["Assets"] * 10).sum()
    if rule3 > 0:
        issues.append(f"  {rule3} companies with liabilities more than 10x their assets (flagged for review)")

    rule4 = df["Company"].isna().sum()
    if rule4 > 0:
        issues.append(f"  {rule4} companies with no name removed")
        df = df[df["Company"].notna()]

    if issues:
        print("  Issues found and handled:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All business rules passed.")

    print(f"  Final company count : {len(df)}")
    print("")

    return df


df = validate_data(df)


# Step 7: Add a data completeness score per company

print("Step 7: Calculating data completeness scores...")
print("")


def add_completeness_score(df_clean, df_original):
    """
    Scores each company on how complete its original data was.

    A company with all 5 financial figures present scores 100%.
    A company missing 2 figures scores 60%.

    This score is used later in the Bayesian uncertainty calculation
    to give less confident risk scores to companies with incomplete data.
    """

    original_cols = ["Revenue ($B)", "Net Income ($B)", "Total Assets ($B)", "Total Liabilities ($B)", "Cash ($B)"]

    completeness = []
    for _, row in df_original.iterrows():
        non_null = sum(1 for col in original_cols if pd.notna(row.get(col)))
        score    = round(non_null / len(original_cols) * 100, 0)
        completeness.append(score)

    df_clean["Data_Completeness"] = completeness[:len(df_clean)]

    avg  = np.mean(completeness)
    full = sum(1 for s in completeness if s == 100)

    print(f"  Average completeness : {avg:.1f}%")
    print(f"  Fully complete       : {full} companies")
    print(f"  Partial data         : {len(df_clean) - full} companies")
    print("")

    return df_clean


df = add_completeness_score(df, df_raw)


# Step 8: Save the cleaned data

print("Step 8: Saving clean data...")
print("")

df.to_csv("companies_clean_data.csv", index=False)
df.to_excel("companies_clean_data.xlsx", index=False)

print("  CSV   : companies_clean_data.csv")
print("  Excel : companies_clean_data.xlsx")
print("")


# Step 9: Generate visualisation charts

print("Step 9: Generating data quality charts...")
print("")


def generate_charts(df):
    """
    Produces a 6-panel chart summarising the data quality
    and distribution of the cleaned dataset.
    """

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Audit Risk Intelligence System\nData Quality and Distribution Report",
        fontsize=16,
        fontweight="bold"
    )

    gs  = gridspec.GridSpec(2, 3, figure=fig)
    clr = ["#86BC25", "#0076A8", "#26890D", "#DA291C", "#FFB81C"]

    # Chart 1: Revenue distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["Revenue"].dropna(), bins=30, color=clr[0], edgecolor="white", alpha=0.85)
    ax1.set_title("Revenue Distribution (Billions)", fontweight="bold")
    ax1.set_xlabel("Revenue ($B)")
    ax1.set_ylabel("Number of Companies")
    ax1.grid(axis="y", alpha=0.3)

    # Chart 2: Net income distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["NetIncome"].dropna(), bins=30, color=clr[1], edgecolor="white", alpha=0.85)
    ax2.set_title("Net Income Distribution (Billions)", fontweight="bold")
    ax2.set_xlabel("Net Income ($B)")
    ax2.set_ylabel("Number of Companies")
    ax2.grid(axis="y", alpha=0.3)

    # Chart 3: Assets vs Liabilities scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df["Assets"], df["Liabilities"], color=clr[2], alpha=0.5, s=20)
    ax3.set_title("Assets vs Liabilities (Billions)", fontweight="bold")
    ax3.set_xlabel("Total Assets ($B)")
    ax3.set_ylabel("Total Liabilities ($B)")
    ax3.grid(alpha=0.3)

    # Chart 4: Missing values percentage
    ax4 = fig.add_subplot(gs[1, 0])
    cols = ["Revenue", "NetIncome", "Assets", "Liabilities", "Cash"]
    df_raw_renamed = df_raw.rename(columns={
        "Revenue ($B)"           : "Revenue",
        "Net Income ($B)"        : "NetIncome",
        "Total Assets ($B)"      : "Assets",
        "Total Liabilities ($B)" : "Liabilities",
        "Cash ($B)"              : "Cash"
    })
    miss_pct = [df_raw_renamed[c].isna().mean() * 100 for c in cols]
    bars = ax4.bar(cols, miss_pct, color=clr[3], edgecolor="white", alpha=0.85)
    ax4.set_title("Missing Values by Column (%)", fontweight="bold")
    ax4.set_ylabel("Missing (%)")
    ax4.set_xticklabels(cols, rotation=45, ha="right")
    ax4.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, miss_pct):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}%", ha="center", fontsize=9)

    # Chart 5: Data completeness distribution
    ax5 = fig.add_subplot(gs[1, 1])
    comp_counts = df["Data_Completeness"].value_counts().sort_index()
    ax5.bar(comp_counts.index.astype(str), comp_counts.values, color=clr[4], edgecolor="white", alpha=0.85)
    ax5.set_title("Data Completeness Distribution", fontweight="bold")
    ax5.set_xlabel("Completeness Score (%)")
    ax5.set_ylabel("Number of Companies")
    ax5.grid(axis="y", alpha=0.3)

    # Chart 6: Top 10 companies by revenue
    ax6 = fig.add_subplot(gs[1, 2])
    top10 = df.nlargest(10, "Revenue")
    ax6.barh(top10["Company"].str[:20], top10["Revenue"], color=clr[0], edgecolor="white", alpha=0.85)
    ax6.set_title("Top 10 Companies by Revenue (Billions)", fontweight="bold")
    ax6.set_xlabel("Revenue ($B)")
    ax6.invert_yaxis()
    ax6.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("data_quality_report.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Chart saved : data_quality_report.png")
    print("")


generate_charts(df)


# Final summary

print("Preprocessing complete.")
print("")
print("  What was done:")
print("  - Missing values filled (median and lower quartile)")
print("  - Duplicate companies removed")
print("  - Outliers capped using Winsorisation (1st to 99th percentile)")
print("  - Business rules validated")
print("  - Data completeness scores added per company")
print("  - Quality charts generated")
print("")
print("  Files created:")
print("  - companies_clean_data.csv")
print("  - companies_clean_data.xlsx")
print("  - data_quality_report.png")
print("")
print(f"  Clean dataset : {len(df)} companies ready for modelling")
print("")
print("Next file: risk_score.py")