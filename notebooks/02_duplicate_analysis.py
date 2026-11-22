
import pandas as pd
from pathlib import Path

# Load original dataset
df = pd.read_csv("data/raw/creditcard.csv")

print("========== DUPLICATE ANALYSIS ==========")

# Identify all rows belonging to duplicate groups
duplicate_mask = df.duplicated(keep=False)
duplicate_rows = df[duplicate_mask].copy()

print("\nDuplicate occurrences (excluding first copies):")
print(df.duplicated().sum())

print("\nTotal rows involved in duplicate groups:")
print(len(duplicate_rows))

# Check fraud distribution among duplicate rows
print("\nFraud distribution in duplicate groups:")
print(duplicate_rows["Class"].value_counts())

print("\nFraud percentages in duplicate groups:")
print(duplicate_rows["Class"].value_counts(normalize=True) * 100)

# Check for identical input features with different class labels
feature_columns = df.columns.drop("Class")

feature_duplicate_mask = df.duplicated(
    subset=feature_columns,
    keep=False
)

feature_duplicates = df[feature_duplicate_mask]

print("\n========== FEATURE DUPLICATE ANALYSIS ==========")
print("Rows with repeated input features:",
      len(feature_duplicates))

print("\nClass distribution among repeated input features:")
print(feature_duplicates["Class"].value_counts())

# Display sample duplicate records
print("\n========== SAMPLE DUPLICATE RECORDS ==========")
print(
    duplicate_rows.sort_values(
        by=["Time", "Amount", "Class"]
    ).head(20)
)

# Save results
Path("outputs").mkdir(exist_ok=True)

with open(
    "outputs/duplicate_analysis.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("DUPLICATE OCCURRENCES\n")
    file.write(str(df.duplicated().sum()))

    file.write("\n\nROWS IN DUPLICATE GROUPS\n")
    file.write(str(len(duplicate_rows)))

    file.write("\n\nDUPLICATE CLASS DISTRIBUTION\n")
    file.write(str(duplicate_rows["Class"].value_counts()))

    file.write("\n\nREPEATED INPUT FEATURES\n")
    file.write(str(len(feature_duplicates)))

    file.write("\n\nCLASS DISTRIBUTION OF REPEATED FEATURES\n")
    file.write(str(feature_duplicates["Class"].value_counts()))

print("\nDuplicate analysis completed!")
