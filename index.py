import pandas as pd
import numpy as np

# 1. Dataset Selection & Loading
print("--- 1. Loading Dataset ---")
file_path = "sales_data.csv"

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin1')

print(f"Dataset '{file_path}' loaded successfully.")

# 2. Data Inspection
print("\n--- 2. Data Inspection ---")
print("Dataset Shape (Rows, Columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 3. Identification of Columns and Data Types
print("\n--- 3. Columns and Data Types ---")
df.info()

# 4. Identification of Missing Values
print("\n--- 4. Missing Values Audit ---")
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

# 5. Handling Missing Values (Fallback Logic)
print("\n--- 5. Handling Missing Values ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# 6. Removal of Duplicate or Incorrect Records
print("\n--- 6. Removing Duplicates & Auditing Records ---")
duplicates = df.duplicated().sum()
print(f"Duplicate records found: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate records removed.")

# 7. Preparation for Analysis & Feature Engineering
print("\n--- 7. Preparing Dataset for Analysis ---")
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed')

# Feature Engineering
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Ship Duration (Days)'] = (df['Ship Date'] - df['Order Date']).dt.days

# Export Cleaned Dataset
cleaned_file_path = "cleaned_sales_data.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved as '{cleaned_file_path}'. Shape: {df.shape}")
