
import pandas as pd

df = pd.read_csv("data/processed/creditcard_cleaned.csv")

print("========== FEATURE ANALYSIS ==========")

# Check dataset information
print("\nDataset shape:", df.shape)

# Check feature statistics
print("\n========== AMOUNT STATISTICS ==========")
print(df["Amount"].describe())

print("\n========== TIME STATISTICS ==========")
print(df["Time"].describe())

# Compare Amount between legitimate and fraudulent transactions
print("\n========== AMOUNT BY CLASS ==========")
print(
    df.groupby("Class")["Amount"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
)

# Compare Time between legitimate and fraudulent transactions
print("\n========== TIME BY CLASS ==========")
print(
    df.groupby("Class")["Time"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
)

# Check feature ranges
print("\n========== FEATURE STANDARD DEVIATION ==========")
print(df.drop(columns=["Class"]).std().sort_values())

print("\nFeature analysis completed!")
