# --------------------------------------------

# LONDON: HOUSE PRICES (BOROUGH) + CRIME (BOROUGH, MONTHLY → MONTHLY ALIGNMENT)

# Step-by-step pipeline with prints at each step

# --------------------------------------------

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


# ---------- STEP 0. LOAD DATA ----------
print("\n[STEP 0] Load datasets")
# Make sure to adjust these paths if you've moved the files
df_price = pd.read_csv("C:/Users/ASUS/Downloads/land-registry-house-prices-borough (1).csv")
df_crime = pd.read_csv("C:/Users/ASUS/Downloads/MPS Borough Level Crime (Historical).csv")

print(f"Price shape: {df_price.shape}; Crime shape: {df_crime.shape}")
print("\nPrice (head):")
print(df_price.head(5))
print("\nCrime (head):")
print(df_crime.head(5))


# ---------- STEP 1. BASIC INSPECTION ----------
print("\n[STEP 1] Inspect price 'Measure' values and columns")
print("Unique Measures:", df_price["Measure"].unique())
print("Price columns:", list(df_price.columns))


# ---------- STEP 2. ENSURE 'Value' IS NUMERIC (removes commas like '146,043') ----------
print("\n[STEP 2] Clean 'Value' to numeric")
before_non_numeric = df_price["Value"].isna().sum()
df_price["Value"] = (
    df_price["Value"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("£", "", regex=False)
    .str.strip()
)
df_price["Value"] = pd.to_numeric(df_price["Value"], errors="coerce")
after_non_numeric = df_price["Value"].isna().sum()
print(f"Non-numeric 'Value' before: {before_non_numeric}, after: {after_non_numeric}")


# ---------- STEP 3. PREP A CLEAN HOUSE-PRICE MONTHLY TABLE FOR MERGING ----------
# We need to parse 'Year ending Dec YYYY' into a proper monthly date
print("\n[STEP 3] Prepare a clean monthly house-price table for merging")

df_house_monthly_temp = df_price.copy()

# Extract Year and Month. Assumes format 'Year ending Mon YYYY'.
# Based on head, it seems to always be 'Dec' for price data, implying End-of-Year values.
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# Regex to capture the month abbreviation and year
date_pattern = r"Year ending (\w{3})\s(\d{4})"
date_matches = df_house_monthly_temp['Year'].str.extract(date_pattern)

# If date_matches is empty, it means the pattern didn't match. Add error handling if necessary.
if date_matches.empty:
    raise ValueError("Could not extract month and year from 'Year' column in price data. Check format.")

df_house_monthly_temp['Month_Abbr'] = date_matches[0]
df_house_monthly_temp['Year_Numeric'] = pd.to_numeric(date_matches[1])

# Convert month abbreviation to month number
df_house_monthly_temp['Month_Numeric'] = df_house_monthly_temp['Month_Abbr'].map(month_mapping)

# Create a proper datetime object (YYYY-MM-01 as requested for day)
# Ensure Year_Numeric and Month_Numeric are not NaN before creating date
df_house_monthly_temp.dropna(subset=['Year_Numeric', 'Month_Numeric'], inplace=True)
df_house_monthly_temp['Year_Month_Day'] = pd.to_datetime(
    df_house_monthly_temp['Year_Numeric'].astype(str) + '-' +
    df_house_monthly_temp['Month_Numeric'].astype(int).astype(str) + '-01'
)

# Tidy names
df_house_monthly_temp["Area"] = df_house_monthly_temp["Area"].str.strip()

# Now pivot to get Mean, Median, Sales as columns for each Area and Year_Month_Day
df_house_eoy = df_house_monthly_temp.pivot_table(
    index=["Code", "Area", "Year_Month_Day"],
    columns="Measure",
    values="Value",
    aggfunc="first" # Assumes unique Measure for each Code-Area-Year_Month_Day combination
).reset_index()
df_house_eoy.columns.name = None # Remove the 'Measure' column name for clarity

# Rename the date column to 'Date' as per user request
df_house_eoy.rename(columns={'Year_Month_Day': 'Date'}, inplace=True)


print("House EOY (End-of-Year/Monthly) (head):")
print(df_house_eoy[["Area", "Date", "Mean", "Median", "Sales"]].head(6))
print("House EOY shape:", df_house_eoy.shape)

print(f"Months present in Price data (from 'Year ending Mon YYYY' format): {df_house_eoy['Date'].dt.month.unique()}")
print("As observed, house price data is typically recorded 'Year ending Dec YYYY'.\nThis means it's available only for December of each year. Merging will only match December crime data.")


# ---------- STEP 4. CRIME: KEEP ONLY NUMERIC YYYYMM COLUMNS, MELT, EXTRACT YEAR_MONTH_DAY, CLEAN ----------
print("\n[STEP 4] Crime: select monthly columns → long format → extract YYYY-MM-01 date")
date_cols = [c for c in df_crime.columns if str(c).isdigit()]  # YYYYMM columns

if "BoroughName" not in df_crime.columns:
    raise ValueError("Expected a 'BoroughName' column in the crime dataset.")

df_crime_monthly = df_crime.melt(
    id_vars=["BoroughName", "MajorText", "MinorText"], # Keep Major/Minor text for now, can drop later if not needed
    value_vars=date_cols,
    var_name="YearMonth_Raw",
    value_name="CrimeCount"
)

# Ensure numeric crime counts
df_crime_monthly["CrimeCount"] = pd.to_numeric(df_crime_monthly["CrimeCount"], errors="coerce").fillna(0).astype(int)

# Convert YYYYMM to YYYY-MM-01 datetime object
df_crime_monthly["Date"] = pd.to_datetime(df_crime_monthly["YearMonth_Raw"], format='%Y%m', errors='coerce')

# Drop rows where Date conversion failed
df_crime_monthly.dropna(subset=['Date'], inplace=True)

# Aggregate (sum) crime counts per borough and date (01-MM-YYYY)
# This sums crime counts from 'MajorText' and 'MinorText' categories within the same borough and month
df_crime_monthly = (
    df_crime_monthly.groupby(["BoroughName", "Date"], as_index=False)["CrimeCount"]
    .sum()
)

df_crime_monthly["BoroughName"] = df_crime_monthly["BoroughName"].str.strip()


print("Crime monthly (head):")
print(df_crime_monthly.head(6))
print("Crime monthly shape:", df_crime_monthly.shape)


# ---------- STEP 5. DIAGNOSTICS: WHICH BOROUGHS/YEAR_MONTH_DAY PAIRS MATCH / MISMATCH ----------
print("\n[STEP 5] Diagnostics: check name mismatches and date ranges before merge")

price_boroughs = set(df_house_eoy["Area"].unique())
crime_boroughs = set(df_crime_monthly["BoroughName"].unique())

only_in_price = sorted(price_boroughs - crime_boroughs)
only_in_crime = sorted(crime_boroughs - price_boroughs)

print(f"Boroughs only in PRICE (will be dropped by inner join): {only_in_price}")
print(f"Boroughs only in CRIME (will be dropped by inner join): {only_in_crime}")

# Show date coverage overlap
print("\nDate coverage:")
print("House prices range:", df_house_eoy["Date"].min().strftime('%Y-%m-%d'), "→", df_house_eoy["Date"].max().strftime('%Y-%m-%d'))
print("Crime data range:", df_crime_monthly["Date"].min().strftime('%Y-%m-%d'), "→", df_crime_monthly["Date"].max().strftime('%Y-%m-%d'))
print("Note: As house price data is End-of-Year, only crime data for December will match.")


# ---------- STEP 6. MERGE (INNER) TO KEEP ONLY EXACT MATCHES ON BOROUGH AND MONTH-YEAR ----------
print("\n[STEP 6] Merge house (EOY) × crime (monthly) on (Area == BoroughName, Date); drop mismatches (inner join)")

# Using 'Date' column directly now that both are datetime objects representing YYYY-MM-01
merged = pd.merge(
    df_house_eoy,                  # columns: Code, Area, Date, Mean, Median, Sales
    df_crime_monthly,              # columns: BoroughName, Date, CrimeCount
    left_on=["Area", "Date"],
    right_on=["BoroughName", "Date"],
    how="inner"
).drop(columns=["BoroughName"]) # Drop duplicate BoroughName after merge

print("Merged (head):")
print(merged.head(10))
print("Merged shape:", merged.shape)


# ---------- STEP 7. (OPTIONAL) SORT & CLEAN FINAL OUTPUT ----------
# Sorting
merged = merged.sort_values(["Area", "Date"]).reset_index(drop=True)
print("\n[STEP 7] Sorted merged (head):")
print(merged.head(10))
print(merged.shape)

# Drop 'Code' column as requested at the end, if not needed in the final model.
# Keeping 'Area' as it's the name you usually refer to.
df_final = merged.drop(columns=["Code"])

print("\nFinal Data---------------------------------------------------------------")
print(df_final.head(10))
print(df_final.shape)

# If you want to save:
df_final.to_csv("./dataset.csv", index=False)
print("Saved: london_borough_price_crime_merged_monthly_aligned.csv")
