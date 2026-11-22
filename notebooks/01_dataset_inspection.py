
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load dataset
DATA_PATH = Path("data/raw/creditcard.csv")

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

# 2. Dataset dimensions
print("\nDataset shape:", df.shape)
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# 3. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 4. First five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 5. Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 6. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())

# 7. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

# 8. Class distribution
print("\n========== FRAUD DISTRIBUTION ==========")
print(df["Class"].value_counts())

print("\nClass percentages:")
print(df["Class"].value_counts(normalize=True) * 100)

# 9. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe().T)

# 10. Infinite values
print("\n========== INFINITE VALUES ==========")
print("Infinite values:", np.isinf(df.select_dtypes(include=np.number)).sum().sum())

# 11. Amount analysis
print("\n========== TRANSACTION AMOUNT ==========")
print(df["Amount"].describe())

# 12. Fraud vs legitimate transaction amount
print("\n========== AMOUNT BY CLASS ==========")
print(df.groupby("Class")["Amount"].agg(
    ["count", "mean", "median", "min", "max"]
))

# 13. Save inspection results
Path("outputs").mkdir(exist_ok=True)

with open("outputs/dataset_inspection.txt", "w", encoding="utf-8") as file:
    file.write("DATASET OVERVIEW\n")
    file.write(str(df.shape) + "\n\n")
    file.write("DATA TYPES\n")
    file.write(str(df.dtypes) + "\n\n")
    file.write("MISSING VALUES\n")
    file.write(str(df.isnull().sum()) + "\n\n")
    file.write("DUPLICATES\n")
    file.write(str(df.duplicated().sum()) + "\n\n")
    file.write("CLASS DISTRIBUTION\n")
    file.write(str(df["Class"].value_counts()) + "\n\n")
    file.write("STATISTICAL SUMMARY\n")
    file.write(str(df.describe().T) + "\n")

print("\nInspection completed successfully!")
