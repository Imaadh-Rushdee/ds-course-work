
import pandas as pd
from pathlib import Path

# Load original dataset
df = pd.read_csv("data/raw/creditcard.csv")

print("========== BEFORE CLEANING ==========")
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["Class"].value_counts())

# Remove exact duplicate rows
df_cleaned = df.drop_duplicates(keep="first").copy()

print("\n========== AFTER CLEANING ==========")
print("Dataset shape:", df_cleaned.shape)
print("\nClass distribution:")
print(df_cleaned["Class"].value_counts())

# Check remaining duplicates
print("\nRemaining duplicate rows:",
      df_cleaned.duplicated().sum())

# Check missing values
print("Total missing values:",
      df_cleaned.isnull().sum().sum())

# Calculate removed records
removed = len(df) - len(df_cleaned)

print("\n========== CLEANING SUMMARY ==========")
print("Original records:", len(df))
print("Removed duplicate records:", removed)
print("Remaining records:", len(df_cleaned))

# Save cleaned dataset
output_path = Path("data/processed/creditcard_cleaned.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df_cleaned.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)
print("Data cleaning completed successfully!")
