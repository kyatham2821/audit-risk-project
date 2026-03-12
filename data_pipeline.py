# ============================================
# DELOITTE AUDIT RISK PROJECT
# Data Pipeline — 600 Companies Version
# Author: Prathibha Kyatham
# University College Dublin — MSc Data Science
# ============================================

# Import all libraries
import requests
import pandas as pd
from tqdm import tqdm
import time

# ============================================
# PART 1: DOWNLOAD FULL COMPANY LIST FROM SEC
# ============================================

print("=====================================")
print("  DELOITTE AUDIT RISK PROJECT")
print("  Data Pipeline — 600 Companies")
print("=====================================")
print("")
print("Step 1: Downloading full company")
print("        list from SEC EDGAR...")
print("")

def get_all_companies():
    """
    Downloads the complete list of all
    companies registered with SEC EDGAR

    Output: dictionary of company name → CIK
    """

    # SEC provides a free file with
    # every single registered company
    url = "https://www.sec.gov/files/company_tickers.json"

    headers = {
        "User-Agent": "Prathibha Kyatham prathibha@ucd.ie"
    }

    print("  Connecting to SEC EDGAR...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("  ✅ Company list downloaded!")
        data = response.json()

        # Convert to simple dictionary
        # company name → CIK number
        companies = {}
        for key, value in data.items():
            company_name = value["title"]
            cik = str(value["cik_str"]).zfill(10)
            companies[company_name] = cik

        print(f"  Total companies available: {len(companies)}")
        return companies

    else:
        print(f"  ❌ Error: {response.status_code}")
        return None

# Download full company list
all_companies = get_all_companies()

# ============================================
# PART 2: SELECT TOP 600 COMPANIES
# ============================================

print("")
print("Step 2: Selecting top 600 companies...")
print("")

# Simply take the first 600 from SEC list
# SEC lists companies in order of filing activity
# Most active/largest companies appear first!
selected_companies = dict(
    list(all_companies.items())[:1000]
)

print(f"  ✅ Selected {len(selected_companies)} companies")
print("")
print("  First 10 selected companies:")
for i, company in enumerate(
        list(selected_companies.keys())[:10]):
    print(f"    {i+1}. {company}")
print(f"    ... and "
      f"{len(selected_companies) - 10} more!")
print("")

# ============================================
# PART 3: DOWNLOAD FUNCTION
# ============================================

def get_company_facts(cik):
    """
    Downloads all financial data for one company
    from SEC EDGAR website

    Input  : cik = company ID number
    Output : financial data as dictionary
    """

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    headers = {
        "User-Agent": "Prathibha Kyatham prathibha@ucd.ie"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        print("  ⚠️ Rate limited - waiting 10 seconds...")
        time.sleep(10)
        return None
    else:
        return None

# ============================================
# PART 4: EXTRACT FINANCIAL NUMBERS
# ============================================

def extract_financials(company_name, cik):
    """
    Extracts 5 key financial numbers from SEC data

    Input  : company_name = example "Apple Inc."
             cik          = example "0000320193"
    Output : dictionary of 5 financial numbers
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
            annual = [v for v in values
                     if v.get("form") == "10-K"]
            if not annual:
                return None
            sorted_values = sorted(
                annual,
                key=lambda x: x["end"],
                reverse=True)
            return sorted_values[0]["val"]

        # Extract our 5 key numbers
        revenue     = get_latest_value("Revenues")
        net_income  = get_latest_value("NetIncomeLoss")
        assets      = get_latest_value("Assets")
        liabilities = get_latest_value("Liabilities")
        cash        = get_latest_value(
                      "CashAndCashEquivalentsAtCarryingValue")

        # Try alternative revenue names
        if revenue is None:
            revenue = get_latest_value(
                      "RevenueFromContractWithCustomerExcludingAssessedTax")

        if revenue is None:
            revenue = get_latest_value(
                      "SalesRevenueNet")

        # Skip companies with no data at all
        if all(v is None for v in
               [revenue, net_income,
                assets, liabilities, cash]):
            return None

        financials = {
            "Company"          : company_name,
            "CIK"              : cik,
            "Revenue"          : revenue,
            "Net Income"       : net_income,
            "Total Assets"     : assets,
            "Total Liabilities": liabilities,
            "Cash"             : cash,
        }

        return financials

    except Exception:
        return None

# ============================================
# PART 5: PROCESS ALL 600 COMPANIES
# ============================================

print("Step 3: Downloading financial data...")
print("        This will take ~10 minutes")
print("        Grab a cup of tea! ☕")
print("")

# Empty list to store results
all_companies_data = []

# Track results
successful = []
failed     = []

# Loop through all 600 companies
for company_name, cik in tqdm(
        selected_companies.items(),
        desc="📥 Downloading",
        unit=" company"):

    financials = extract_financials(
                 company_name, cik)

    if financials:
        all_companies_data.append(financials)
        successful.append(company_name)
    else:
        failed.append(company_name)

    # Be polite to SEC server
    time.sleep(1)

# Show download summary
print("")
print(f"✅ Successfully downloaded : {len(successful)}")
print(f"❌ No financial data found : {len(failed)}")
print("")

# ============================================
# PART 6: SAVE THE DATA
# ============================================

print("=====================================")
print("Step 4: Saving Data")
print("=====================================")
print("")

# Convert to table
df = pd.DataFrame(all_companies_data)

# Convert raw numbers to billions
money_columns = [
    "Revenue",
    "Net Income",
    "Total Assets",
    "Total Liabilities",
    "Cash"
]

for col in money_columns:
    if col in df.columns:
        df[col] = df[col].apply(
            lambda x: round(x / 1_000_000_000, 2)
            if x is not None else None
        )

# Rename columns
df = df.rename(columns={
    "Revenue"          : "Revenue ($B)",
    "Net Income"       : "Net Income ($B)",
    "Total Assets"     : "Total Assets ($B)",
    "Total Liabilities": "Total Liabilities ($B)",
    "Cash"             : "Cash ($B)"
})

# Add risk level
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
    except:
        pass
    return "Unknown"

df["Risk Level"] = df.apply(
                   calculate_risk, axis=1)

# Sort by Revenue biggest first
df = df.sort_values(
    "Revenue ($B)",
    ascending=False,
    na_position="last")

# Reset index starting from 1
df = df.reset_index(drop=True)
df.index = df.index + 1

# Save files
excel_filename = "companies_financial_data.xlsx"
csv_filename   = "companies_financial_data.csv"

df.to_excel(excel_filename)
df.to_csv(csv_filename)

print(f"✅ Excel saved : {excel_filename}")
print(f"✅ CSV saved   : {csv_filename}")
print("")

# ============================================
# PART 7: SHOW FINAL RESULTS
# ============================================

print("=====================================")
print("  TOP 20 COMPANIES BY REVENUE")
print("=====================================")
print("")
print(df.head(20).to_string())
print("")

# Risk summary
print("=====================================")
print("  RISK SUMMARY")
print("=====================================")
print("")
risk_counts = df["Risk Level"].value_counts()
for risk, count in risk_counts.items():
    if risk == "High Risk":
        emoji = "🔴"
    elif risk == "Medium Risk":
        emoji = "🟡"
    elif risk == "Low Risk":
        emoji = "🟢"
    else:
        emoji = "⚪"
    pct = round(count / len(df) * 100, 1)
    print(f"  {emoji} {risk:15} : "
          f"{count:3} companies ({pct}%)")

print("")
print("=====================================")
print(f"  TOTAL COMPANIES ANALYSED: {len(df)}")
print("=====================================")
print("")
print("✅ DATA PIPELINE COMPLETE!")
print("")
print("Next Steps:")
print("  → Compute Beneish M-Score")
print("  → Run Anomaly Detection")
print("  → Build Risk Dashboard")
print("=====================================")
