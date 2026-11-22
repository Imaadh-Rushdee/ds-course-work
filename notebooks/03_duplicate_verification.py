
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

feature_columns = df.columns.drop("Class")

# Find identical input features with different Class labels
label_counts = (
    df.groupby(list(feature_columns))["Class"]
    .nunique()
)

conflicting_groups = label_counts[label_counts > 1]

print("========== LABEL CONFLICT CHECK ==========")
print("Groups with conflicting labels:", len(conflicting_groups))

if len(conflicting_groups) == 0:
    print("No conflicting labels found.")
else:
    print("WARNING: Some identical feature combinations have different labels.")

# Check additional duplicate copies by class
print("\n========== DUPLICATES BY CLASS ==========")
print(df[df.duplicated()]["Class"].value_counts())

print("\nVerification completed!")
