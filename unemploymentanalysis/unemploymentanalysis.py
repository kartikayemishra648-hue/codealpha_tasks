# ============================================================
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

print("=" * 60)
print("UNEMPLOYMENT ANALYSIS WITH PYTHON")
print("=" * 60)


# -----------------------------
# 2. FIND AND LOAD CSV FILE
# -----------------------------

# Automatically finds CSV file in the same folder
csv_files = glob.glob("*.csv")

if len(csv_files) == 0:
    print("ERROR: No CSV file found in the project folder.")
    print("Please keep your unemployment CSV file in the same folder.")
    exit()

file_path = csv_files[0]

print("\nDataset found:", file_path)

df = pd.read_csv(file_path)

print("\nOriginal Dataset Shape:", df.shape)


# -----------------------------
# 3. CLEAN COLUMN NAMES
# -----------------------------

df.columns = df.columns.str.strip()

print("\nColumns in dataset:")
print(df.columns.tolist())


# -----------------------------
# 4. DATA CLEANING
# -----------------------------

# Remove completely empty rows
df.dropna(how="all", inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert Date column
df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

# Convert numerical columns
numeric_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Remove rows with missing important values
df.dropna(
    subset=[
        "Region",
        "Date",
        "Estimated Unemployment Rate (%)"
    ],
    inplace=True
)

print("\nDataset after cleaning:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 5. BASIC DATA EXPLORATION
# -----------------------------

print("\n" + "=" * 60)
print("BASIC DATA EXPLORATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nNumber of Regions:", df["Region"].nunique())

print("\nRegions:")
print(df["Region"].unique())

print("\nArea distribution:")
print(df["Area"].value_counts())


# -----------------------------
# 6. CREATE OUTPUT FOLDER
# -----------------------------

if not os.path.exists("graphs"):
    os.makedirs("graphs")

print("\nGraphs will be saved inside the 'graphs' folder.")


# ============================================================
# 7. OVERALL UNEMPLOYMENT TREND
# ============================================================

monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment["Date"],
    monthly_unemployment["Estimated Unemployment Rate (%)"],
    marker="o"
)

plt.title("Overall Unemployment Rate Trend")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("graphs/01_overall_unemployment_trend.png")
plt.show()


# ============================================================
# 8. COVID-19 IMPACT
# ============================================================

# Define periods
def classify_period(date):

    if date < pd.Timestamp("2020-03-01"):
        return "Before COVID"

    elif date <= pd.Timestamp("2020-05-31"):
        return "COVID Period"

    else:
        return "June 2020"


df["COVID Period"] = df["Date"].apply(classify_period)


covid_analysis = (
    df.groupby("COVID Period")
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

# Correct order
period_order = [
    "Before COVID",
    "COVID Period",
    "June 2020"
]

covid_analysis = covid_analysis.reindex(period_order)

print("\n" + "=" * 60)
print("COVID-19 IMPACT")
print("=" * 60)

print(covid_analysis)


# COVID bar chart

plt.figure(figsize=(9, 6))

covid_analysis.plot(kind="bar")

plt.title("Impact of COVID-19 on Unemployment")
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("graphs/02_covid_impact.png")
plt.show()


# ============================================================
# 9. COVID PERIOD HIGHLIGHTED TREND
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment["Date"],
    monthly_unemployment["Estimated Unemployment Rate (%)"],
    marker="o",
    label="Unemployment Rate"
)

plt.axvspan(
    pd.Timestamp("2020-03-01"),
    pd.Timestamp("2020-05-31"),
    alpha=0.2,
    label="COVID-19 Period"
)

plt.title("Unemployment Trend During COVID-19")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("graphs/03_covid_trend.png")
plt.show()


# ============================================================
# 10. REGION-WISE UNEMPLOYMENT
# ============================================================

region_unemployment = (
    df.groupby("Region")
    ["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\n" + "=" * 60)
print("REGION-WISE UNEMPLOYMENT")
print("=" * 60)

print(region_unemployment)


# Top 10 regions

plt.figure(figsize=(12, 7))

region_unemployment.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Regions with Highest Average Unemployment")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.grid(axis="x")
plt.tight_layout()

plt.savefig("graphs/04_top_10_regions.png")
plt.show()


# Bottom 10 regions

plt.figure(figsize=(12, 7))

region_unemployment.tail(10).sort_values(
    ascending=False
).plot(kind="barh")

plt.title("10 Regions with Lowest Average Unemployment")
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.grid(axis="x")
plt.tight_layout()

plt.savefig("graphs/05_lowest_10_regions.png")
plt.show()


# ============================================================
# 11. RURAL VS URBAN
# ============================================================

area_unemployment = (
    df.groupby("Area")
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\n" + "=" * 60)
print("RURAL VS URBAN ANALYSIS")
print("=" * 60)

print(area_unemployment)


plt.figure(figsize=(8, 6))

area_unemployment.plot(kind="bar")

plt.title("Rural vs Urban Unemployment Rate")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("graphs/06_rural_vs_urban.png")
plt.show()


# ============================================================
# 12. RURAL VS URBAN COVID IMPACT
# ============================================================

area_covid = (
    df.groupby(
        ["COVID Period", "Area"]
    )["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
)

area_covid["COVID Period"] = pd.Categorical(
    area_covid["COVID Period"],
    categories=period_order,
    ordered=True
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=area_covid,
    x="COVID Period",
    y="Estimated Unemployment Rate (%)",
    hue="Area"
)

plt.title("COVID-19 Impact: Rural vs Urban")
plt.xlabel("Period")
plt.ylabel("Average Unemployment Rate (%)")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("graphs/07_rural_urban_covid.png")
plt.show()


# ============================================================
# 13. MONTHLY / SEASONAL PATTERN
# ============================================================

df["Month"] = df["Date"].dt.month
df["Month Name"] = df["Date"].dt.strftime("%B")

monthly_pattern = (
    df.groupby(
        ["Month", "Month Name"]
    )["Estimated Unemployment Rate (%)"]
    .mean()
    .reset_index()
    .sort_values("Month")
)

print("\n" + "=" * 60)
print("MONTHLY UNEMPLOYMENT PATTERN")
print("=" * 60)

print(monthly_pattern)


plt.figure(figsize=(12, 6))

sns.barplot(
    data=monthly_pattern,
    x="Month Name",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Monthly Unemployment Pattern")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("graphs/08_monthly_pattern.png")
plt.show()


# ============================================================
# 14. MONTHLY DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="Month Name",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Monthly Distribution of Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("graphs/09_monthly_distribution.png")
plt.show()


# ============================================================
# 15. EMPLOYMENT TREND
# ============================================================

monthly_employed = (
    df.groupby("Date")["Estimated Employed"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_employed["Date"],
    monthly_employed["Estimated Employed"],
    marker="o"
)

plt.title("Employment Trend")
plt.xlabel("Date")
plt.ylabel("Average Estimated Employed")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("graphs/10_employment_trend.png")
plt.show()


# ============================================================
# 16. LABOUR PARTICIPATION RATE
# ============================================================

monthly_lpr = (
    df.groupby("Date")
    ["Estimated Labour Participation Rate (%)"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_lpr["Date"],
    monthly_lpr["Estimated Labour Participation Rate (%)"],
    marker="o"
)

plt.title("Labour Participation Rate Trend")
plt.xlabel("Date")
plt.ylabel("Labour Participation Rate (%)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.savefig("graphs/11_labour_participation.png")
plt.show()


# ============================================================
# 17. CORRELATION ANALYSIS
# ============================================================

correlation_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

correlation = df[correlation_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation)


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Employment Indicators")
plt.tight_layout()

plt.savefig("graphs/12_correlation_heatmap.png")
plt.show()


# ============================================================
# 18. HIGHEST AND LOWEST UNEMPLOYMENT
# ============================================================

highest_month = monthly_unemployment.loc[
    monthly_unemployment[
        "Estimated Unemployment Rate (%)"
    ].idxmax()
]

lowest_month = monthly_unemployment.loc[
    monthly_unemployment[
        "Estimated Unemployment Rate (%)"
    ].idxmin()
]

highest_region = region_unemployment.index[0]
highest_region_rate = region_unemployment.iloc[0]

lowest_region = region_unemployment.index[-1]
lowest_region_rate = region_unemployment.iloc[-1]


# ============================================================
# 19. COVID CALCULATION
# ============================================================

before_covid = covid_analysis["Before COVID"]
during_covid = covid_analysis["COVID Period"]

covid_increase = during_covid - before_covid

covid_percentage_increase = (
    covid_increase / before_covid
) * 100


# ============================================================
# 20. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL PROJECT RESULTS")
print("=" * 60)

print(
    "\nHighest unemployment month:",
    highest_month["Date"].strftime("%B %Y")
)

print(
    "Highest unemployment rate:",
    round(
        highest_month[
            "Estimated Unemployment Rate (%)"
        ],
        2
    ),
    "%"
)

print(
    "\nLowest unemployment month:",
    lowest_month["Date"].strftime("%B %Y")
)

print(
    "Lowest unemployment rate:",
    round(
        lowest_month[
            "Estimated Unemployment Rate (%)"
        ],
        2
    ),
    "%"
)

print(
    "\nHighest average unemployment region:",
    highest_region
)

print(
    "Rate:",
    round(highest_region_rate, 2),
    "%"
)

print(
    "\nLowest average unemployment region:",
    lowest_region
)

print(
    "Rate:",
    round(lowest_region_rate, 2),
    "%"
)

print("\nCOVID-19 Analysis:")

print(
    "Before COVID:",
    round(before_covid, 2),
    "%"
)

print(
    "During COVID:",
    round(during_covid, 2),
    "%"
)

print(
    "Increase:",
    round(covid_increase, 2),
    "percentage points"
)

print(
    "Percentage increase:",
    round(covid_percentage_increase, 2),
    "%"
)


# ============================================================
# 21. KEY INSIGHTS
# ============================================================

print("\n")
print("=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print("""
1. Unemployment rates changed significantly during the
   observed period from 2019 to 2020.

2. A major increase in unemployment was observed during
   the COVID-19 period.

3. April and May 2020 show a particularly strong disruption
   in employment conditions.

4. Unemployment levels differ significantly across regions,
   indicating regional economic differences.

5. Rural and urban areas show different unemployment patterns.

6. Employment levels changed considerably during the period,
   especially around the COVID-19 disruption.

7. Labour participation also changed along with unemployment,
   showing changes in labour-market activity.

8. The monthly analysis reveals variations in unemployment,
   although this dataset covers only a little more than one
   year, so long-term seasonal conclusions should be made
   cautiously.

9. These findings can help policymakers identify regions and
   groups that require employment support.

10. Policies such as job creation, skill development,
    financial assistance and targeted regional support can
    help reduce unemployment during economic disruptions.
""")


# ============================================================
# 22. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "cleaned_unemployment_data.csv",
    index=False
)

print("\nCleaned dataset saved as:")
print("cleaned_unemployment_data.csv")

print("\nGraphs saved inside:")
print("graphs/")

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)