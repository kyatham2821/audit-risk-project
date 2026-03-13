/********************************************
* AUDIT RISK INTELLIGENCE SYSTEM
* SAS Pipeline — Financial Risk Analysis
* Author: Prathibha Kyatham
* University College Dublin
* MSc Data Science
*
* Academic module:
* STAT40840 — Data Programming with SAS
* 
*
* PURPOSE:
* Replicate the audit risk pipeline
* in SAS  
********************************************/

/********************************************
* PART 1: SET UP
********************************************/

/* Clear any previous work */
proc datasets library=work kill nolist;
run;

/* Print a header */
data _null_;
    put "=====================================";
    put "  AUDIT RISK INTELLIGENCE SYSTEM";
    put "  SAS Financial Risk Pipeline";
    put "  Author: Prathibha Kyatham";
    put "  STAT40840 — SAS Programming ";
    put "=====================================";
run;

/********************************************
* PART 2: IMPORT CLEAN DATA FROM CSV
*
* Importing our pre-cleaned dataset
* directly from the CSV file!
********************************************/

proc import
    datafile="/home/u64262052/companies_clean_data.csv"
    out=company_financials
    dbms=csv
    replace;
    getnames=yes;
run;

/* Rename columns to remove spaces */
data company_financials;
    set company_financials;
    rename
        'Revenue ($B)'n      = Revenue
        'Net Income ($B)'n   = NetIncome
        'Total Assets ($B)'n = Assets
        'Total Liabilities ($B)'n = Liabilities
        'Cash ($B)'n         = Cash;
run;

/* Check how many companies loaded */
proc sql;
    select count(*) as Total_Companies
    from company_financials;
quit;

/********************************************
* PART 3: DATA QUALITY CHECK
*
* Professional audit practice —
* always check data quality first!
********************************************/

title "DATA QUALITY REPORT";
title2 "Deloitte Audit Risk Project";

/* Check for missing values */
proc means data=company_financials
    n nmiss mean median min max;
    var Revenue NetIncome Assets
        Liabilities Cash;
run;

/* Check data ranges */
proc univariate data=company_financials
    noprint;
    var Revenue Assets Liabilities;
    output out=outlier_check
        pctlpts=1 5 95 99
        pctlpre=P;
run;

/********************************************
* PART 4: CALCULATE FINANCIAL RATIOS
*
* Creating the same 5 ratios as our
* Python pipeline!
********************************************/

title "FINANCIAL RATIO CALCULATION";

data company_ratios;
    set company_financials;

    /* Ratio 1: Debt Ratio */
    /* Liabilities / Assets */
    if Assets > 0 then
        Debt_Ratio = Liabilities / Assets;
    else
        Debt_Ratio = .;

    /* Ratio 2: Profit Margin */
    /* Net Income / Revenue */
    if Revenue > 0 then
        Profit_Margin = NetIncome / Revenue;
    else
        Profit_Margin = .;

    /* Ratio 3: Cash Ratio */
    /* Cash / Liabilities */
    if Liabilities > 0 then
        Cash_Ratio = Cash / Liabilities;
    else
        Cash_Ratio = .;

    /* Ratio 4: Asset Efficiency */
    /* Revenue / Assets */
    if Assets > 0 then
        Asset_Efficiency = Revenue / Assets;
    else
        Asset_Efficiency = .;

    /* Ratio 5: Return on Assets */
    /* Net Income / Assets */
    if Assets > 0 then
        Return_on_Assets = NetIncome / Assets;
    else
        Return_on_Assets = .;

    /* Round all ratios to 3 decimal places */
    Debt_Ratio       = round(Debt_Ratio,       0.001);
    Profit_Margin    = round(Profit_Margin,    0.001);
    Cash_Ratio       = round(Cash_Ratio,       0.001);
    Asset_Efficiency = round(Asset_Efficiency, 0.001);
    Return_on_Assets = round(Return_on_Assets, 0.001);

run;

/* Print ratios */
proc print data=company_ratios (obs=30) noobs;
    var Company Debt_Ratio Profit_Margin
        Cash_Ratio Asset_Efficiency
        Return_on_Assets;
    title "Financial Ratios — All Companies";
run;

/********************************************
* PART 5: CALCULATE RISK SCORE
*
* Implementing the Prathibha Risk Score
* in SAS!
********************************************/

title "PRATHIBHA RISK SCORE — SAS VERSION";

/* Step 1: Rank each ratio */
/* PROC RANK = percentile ranking */
/* Same as our Python percentile scoring! */
proc rank data=company_ratios
          out=company_ranked
          fraction;

    /* High debt = high risk */
    var Debt_Ratio;
    ranks Debt_Rank;
run;

proc rank data=company_ranked
          out=company_ranked
          fraction descending;

    /* Low profit = high risk (descending!) */
    var Profit_Margin;
    ranks Profit_Rank;
run;

proc rank data=company_ranked
          out=company_ranked
          fraction descending;

    /* Low cash = high risk (descending!) */
    var Cash_Ratio;
    ranks Cash_Rank;
run;

proc rank data=company_ranked
          out=company_ranked
          fraction descending;

    /* Low efficiency = high risk */
    var Asset_Efficiency;
    ranks Efficiency_Rank;
run;

/* Step 2: Calculate PRS Score */
data company_scored;
    set company_ranked;

    /* Each rank score = 0 to 25 */
    Debt_Score       = Debt_Rank       * 25;
    Profit_Score     = Profit_Rank     * 25;
    Liquidity_Score  = Cash_Rank       * 25;
    Efficiency_Score = Efficiency_Rank * 25;

    /* Total PRS Score = 0 to 100 */
    PRS_Score = Debt_Score +
                Profit_Score +
                Liquidity_Score +
                Efficiency_Score;

    PRS_Score = round(PRS_Score, 0.01);

    /* Risk Classification */
    /* Same thresholds as Python version! */
    if PRS_Score >= 80 then
        Risk_Level = "Very High Risk";
    else if PRS_Score >= 65 then
        Risk_Level = "High Risk";
    else if PRS_Score >= 55 then
        Risk_Level = "Medium Risk";
    else
        Risk_Level = "Low Risk";

run;

/* Sort by risk score highest first */
proc sort data=company_scored;
    by descending PRS_Score;
run;

/* Print risk scores */
proc print data=company_scored (obs=30) noobs;
    var Company PRS_Score Risk_Level
        Debt_Score Profit_Score
        Liquidity_Score Efficiency_Score;
    title "Prathibha Risk Score — SAS Results";
run;

/********************************************
* PART 6: RISK SUMMARY STATISTICS
*
* Professional summary report
********************************************/

title "RISK SUMMARY REPORT";

/* Count companies per risk level */
proc freq data=company_scored;
    tables Risk_Level /
        nocum
        nopercent;
    title "Risk Level Distribution";
run;

/* Average scores per risk level */
proc means data=company_scored
    mean median min max;
    class Risk_Level;
    var PRS_Score Debt_Ratio
        Profit_Margin Return_on_Assets;
    title "Average Ratios by Risk Level";
run;

/********************************************
* PART 7: IDENTIFY HIGH RISK COMPANIES
*
* Flag companies needing audit attention
********************************************/

title "HIGH PRIORITY AUDIT FLAGS";

/* Extract high risk companies */
data high_risk_companies;
    set company_scored;
    where Risk_Level in (
        "High Risk",
        "Very High Risk");
run;

proc print data=high_risk_companies noobs;
    var Company PRS_Score Risk_Level
        Debt_Ratio Profit_Margin;
    title "Companies Flagged for Audit Review";
run;

/********************************************
* PART 8: CORRELATION ANALYSIS
*
* Which ratios are most correlated
* with high risk?
********************************************/

title "CORRELATION ANALYSIS";
title2 "Which ratios drive audit risk?";

proc corr data=company_scored
    nosimple;
    var PRS_Score Debt_Ratio
        Profit_Margin Cash_Ratio
        Asset_Efficiency Return_on_Assets;
    title "Correlation Between Risk Score and Ratios";
run;

/********************************************
* PART 9: VISUALISATIONS
*
* Professional charts in SAS
********************************************/

title "RISK VISUALISATIONS";

/* Chart 1: Risk Score Distribution */
proc sgplot data=company_scored;
    histogram PRS_Score /
        fillattrs=(color=CX0076A8)
        binwidth=10;
    refline 65 / axis=x
        lineattrs=(color=red
        thickness=2)
        label="High Risk Threshold";
    refline 80 / axis=x
        lineattrs=(color=darkred
        thickness=2)
        label="Very High Risk";
    xaxis label="PRS Score (0-100)";
    yaxis label="Number of Companies";
    title "Prathibha Risk Score Distribution";
run;

/* Chart 2: Debt Ratio vs Profit Margin */
proc sgplot data=company_scored;
    scatter x=Debt_Ratio
            y=Profit_Margin /
        colorresponse=PRS_Score
        colormodel=(green yellow red)
        markerattrs=(size=10 symbol=circlefilled)
        datalabel=Company;
    xaxis label="Debt Ratio";
    yaxis label="Profit Margin";
    title "Debt Ratio vs Profit Margin";
    title2 "Coloured by Risk Score";
run;

/* Chart 3: Top 10 Highest Risk */
proc sgplot data=company_scored(obs=10);
    hbar Company /
        response=PRS_Score
        fillattrs=(color=CxDA291C)
        datalabel;
    xaxis label="PRS Score";
    title "Top 10 Highest Risk Companies";
run;

/********************************************
* PART 10: EXPORT RESULTS
*
* Save results to CSV
********************************************/

proc export data=company_scored
    outfile="/home/u64262052/sas_risk_scores.csv"
    dbms=csv
    replace;
run;

/********************************************
* FINAL SUMMARY
********************************************/

data _null_;
    put "=====================================";
    put "  SAS PIPELINE COMPLETE!";
    put "=====================================";
    put "  What was done:";
    put "  Data quality check";
    put "  Financial ratios calculated";
    put "  Prathibha Risk Score in SAS";
    put "  Risk classification";
    put "  Summary statistics";
    put "  High risk flags";
    put "  Correlation analysis";
    put "  Visualisations generated";
run;




 