# ============================================
# AUDIT RISK INTELLIGENCE SYSTEM
# File 1 of 8: Data Pipeline
# Author: Prathibha Kyatham
# University College Dublin - MSc Data Science
#
# PURPOSE:
# Downloads financial data for 700+ public
# companies directly from the SEC EDGAR
# database. Extracts 5 key financial figures
# per company and saves to CSV and Excel.
#
# Output files:
#   companies_financial_data.csv
#   companies_financial_data.xlsx
#
# Next file: data_preprocessing.py
# ============================================

import requests
import pandas as pd
from tqdm import tqdm
import time


# Step 1: Download the full company list from SEC EDGAR

print("AUDIT RISK INTELLIGENCE SYSTEM")
print("File 1 of 8: Data Pipeline")
print("")
print("Step 1: Downloading company list from SEC EDGAR...")
print("")


def get_all_companies():
    """
    Downloads the complete list of companies
    registered with the SEC EDGAR database.

    SEC provides this as a free public file.
    Returns a dictionary of company name to CIK number.
    CIK is the unique ID the SEC assigns to each company.
    """

    url = "https://www.sec.gov/files/company_tickers.json"

    headers = {
        "User-Agent": "Prathibha Kyatham prathibha@ucd.ie"
    }

    print("  Connecting to SEC EDGAR...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("  Company list downloaded successfully.")
        data = response.json()

        companies = {}
        for key, value in data.items():
            company_name = value["title"]
            cik = str(value["cik_str"]).zfill(10)
            companies[company_name] = cik

        print(f"  Total companies available : {len(companies)}")
        return companies

    else:
        print(f"  Connection failed. Status code: {response.status_code}")
        return None


all_companies = get_all_companies()


# Step 2: Select the top 700+ companies to analyse

print("")
print("Step 2: Selecting top 700+ companies...")
print("")

selected_companies = dict(list(all_companies.items())[:1000])

print(f"  Companies selected : {len(selected_companies)}")
print("")


# Step 3: Define functions to download financial data


def get_company_facts(cik):
    """
    Downloads all financial filings for one company
    from the SEC EDGAR API.

    Returns the raw JSON data or None if unavailable.
    """

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    headers = {
        "User-Agent": "Prathibha Kyatham prathibha@ucd.ie"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        # SEC rate limits requests - wait and retry
        time.sleep(10)
        return None
    else:
        return None


def extract_financials(company_name, cik):
    """
    Extracts 5 key financial figures from SEC data.

    The 5 figures we extract:
      Revenue      - total income from operations
      Net Income   - profit after all expenses
      Total Assets - everything the company owns
      Liabilities  - everything the company owes
      Cash         - liquid cash available

    We look for the most recent annual (10-K) filing.
    Returns a dictionary or None if data unavailable.
    """

    data = get_company_facts(cik)

    if data is None:
        return None

    try:
        facts = data["facts"]["us-gaap"]

        def get_latest_value(metric_name):
            if metric_name not in facts:
                return None
            units = facts[metric_name]["units"]
            if "USD" not in units:
                return None
            values = units["USD"]
            annual = [v for v in values if v.get("form") == "10-K"]
            if not annual:
                return None
            sorted_values = sorted(annual, key=lambda x: x["end"], reverse=True)
            return sorted_values[0]["val"]

        revenue     = get_latest_value("Revenues")
        net_income  = get_latest_value("NetIncomeLoss")
        assets      = get_latest_value("Assets")
        liabilities = get_latest_value("Liabilities")
        cash        = get_latest_value("CashAndCashEquivalentsAtCarryingValue")

        # Try alternative names for revenue if first attempt fails
        if revenue is None:
            revenue = get_latest_value("RevenueFromContractWithCustomerExcludingAssessedTax")
        if revenue is None:
            revenue = get_latest_value("SalesRevenueNet")

        # Skip companies with no financial data at all
        if all(v is None for v in [revenue, net_income, assets, liabilities, cash]):
            return None

        return {
            "Company"           : company_name,
            "CIK"               : cik,
            "Revenue"           : revenue,
            "Net Income"        : net_income,
            "Total Assets"      : assets,
            "Total Liabilities" : liabilities,
            "Cash"              : cash,
        }

    except Exception:
        return None


# Step 3: Download financial data for all selected companies

print("Step 3: Downloading financial data for all companies...")
print("")

all_companies_data = []
successful = []
failed = []

for company_name, cik in tqdm(selected_companies.items(), desc="  Downloading", unit=" company"):
    financials = extract_financials(company_name, cik)
    if financials:
        all_companies_data.append(financials)
        successful.append(company_name)
    else:
        failed.append(company_name)
    time.sleep(1)

print("")
print(f"  Successfully downloaded : {len(successful)} companies")
print(f"  No data available       : {len(failed)} companies")
print("")


# Step 4: Save the downloaded data to files

print("Step 4: Processing and saving data...")
print("")

df = pd.DataFrame(all_companies_data)

# Convert raw dollar values to billions for readability
money_columns = ["Revenue", "Net Income", "Total Assets", "Total Liabilities", "Cash"]
for col in money_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: round(x / 1_000_000_000, 2) if x is not None else None)

# Rename columns to include the unit
df = df.rename(columns={
    "Revenue"           : "Revenue ($B)",
    "Net Income"        : "Net Income ($B)",
    "Total Assets"      : "Total Assets ($B)",
    "Total Liabilities" : "Total Liabilities ($B)",
    "Cash"              : "Cash ($B)"
})

# Add a simple debt-based risk level for reference
def calculate_risk(row):
    try:
        liabilities = row["Total Liabilities ($B)"]
        assets      = row["Total Assets ($B)"]
        if liabilities and assets and assets > 0:
            debt_ratio = liabilities / assets
            if debt_ratio > 0.8:
                return "High Risk"
            elif debt_ratio > 0.5:
                return "Medium Risk"
            else:
                return "Low Risk"
    except Exception:
        pass
    return "Unknown"

df["Risk Level"] = df.apply(calculate_risk, axis=1)

# Sort by revenue, largest companies first
df = df.sort_values("Revenue ($B)", ascending=False, na_position="last")
df = df.reset_index(drop=True)
df.index = df.index + 1

# Save to both Excel and CSV
df.to_excel("companies_financial_data.xlsx")
df.to_csv("companies_financial_data.csv")

print(f"  Excel saved : companies_financial_data.xlsx")
print(f"  CSV saved   : companies_financial_data.csv")
print(f"  Total companies saved : {len(df)}")
print("")

# Risk level breakdown
print("  Risk level breakdown:")
risk_counts = df["Risk Level"].value_counts()
for risk, count in risk_counts.items():
    pct = round(count / len(df) * 100, 1)
    print(f"    {risk:15} : {count:3} companies ({pct}%)")

print("")
print("Data pipeline complete.")
print("")
print("Next file: data_preprocessing.py")