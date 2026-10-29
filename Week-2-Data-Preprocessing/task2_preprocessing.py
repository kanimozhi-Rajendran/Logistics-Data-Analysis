
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "logistics_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "week2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Collect and load the sample logistics dataset
df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Step 2: Standardize column names and text values
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

# Step 3: Identify missing values and duplicate records
missing_before = df.isnull().sum()
duplicates_before = int(df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Step 4: Convert columns to suitable data types
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"], errors="coerce"
    )

numeric_cols = [
    "Distance_km",
    "Shipping_Cost",
    "Delivery_Days",
    "Expected_Days",
    "Order_Quantity",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Step 5: Handle missing values
for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        median_value = df[col].median()
        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

categorical_cols = df.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_cols:
    if df[col].isnull().any():
        mode_values = df[col].mode(dropna=True)
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])

# Date values that could not be parsed are left missing and reported.
# Avoid inventing dates when no reliable replacement exists.

# Step 6: Detect outliers using the IQR method
outlier_rows = []

for col in numeric_cols:
    if col not in df.columns:
        continue

    values = df[col].dropna()
    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (df[col] < lower_bound) | (df[col] > upper_bound)

    outlier_rows.append({
        "Column": col,
        "Q1": q1,
        "Q3": q3,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": int(mask.sum()),
    })

outlier_report = pd.DataFrame(outlier_rows)
outlier_report.to_csv(
    OUTPUT_DIR / "outlier_report.csv", index=False
)

# Step 7: Save the cleaned dataset
cleaned_file = OUTPUT_DIR / "cleaned_logistics_data.csv"
df.to_csv(cleaned_file, index=False)

# Step 8: Normalize selected numeric columns using Min-Max scaling
# Formula: (value - minimum) / (maximum - minimum)
normalized_df = df.copy()

for col in numeric_cols:
    if col not in normalized_df.columns:
        continue

    minimum = normalized_df[col].min()
    maximum = normalized_df[col].max()

    if pd.notna(minimum) and pd.notna(maximum):
        if maximum != minimum:
            normalized_df[col] = (
                normalized_df[col] - minimum
            ) / (maximum - minimum)
        else:
            normalized_df[col] = 0.0

normalized_df.to_csv(
    OUTPUT_DIR / "normalized_logistics_data.csv",
    index=False
)

# Step 9: Create a data quality summary
summary = [
    "WEEK 2: DATA CLEANING AND PREPROCESSING",
    f"Original rows: {len(pd.read_csv(DATA_FILE))}",
    f"Rows after removing duplicates: {len(df)}",
    f"Duplicate rows detected before removal: {duplicates_before}",
    "",
    "Missing values before cleaning:",
]

for col, count in missing_before.items():
    summary.append(f"{col}: {count}")

summary.extend([
    "",
    "Missing values after cleaning:",
])

for col, count in df.isnull().sum().items():
    summary.append(f"{col}: {count}")

summary.extend([
    "",
    "Outlier detection: IQR method",
    "Outliers are flagged in outlier_report.csv.",
    "They are not automatically deleted or capped.",
    "Normalization: Min-Max scaling to the 0-1 range.",
    f"Cleaned data: {cleaned_file.name}",
    "Normalized data: normalized_logistics_data.csv",
])

(OUTPUT_DIR / "data_quality_summary.txt").write_text(
    "\n".join(summary), encoding="utf-8"
)

print("\n--- PREPROCESSING COMPLETED ---")
print("Rows after cleaning:", len(df))
print("Duplicates removed:", duplicates_before)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nOutlier report:")
print(outlier_report)
print("\nOutput folder:", OUTPUT_DIR)
print("Created: cleaned_logistics_data.csv")
print("Created: normalized_logistics_data.csv")
print("Created: outlier_report.csv")
print("Created: data_quality_summary.txt")
