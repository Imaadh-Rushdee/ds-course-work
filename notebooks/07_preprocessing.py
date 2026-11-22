
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

print("========== PREPROCESSING ==========")
print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Scale Amount and Time only
scaler = StandardScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train[["Amount", "Time"]] = scaler.fit_transform(
    X_train[["Amount", "Time"]]
)

X_test[["Amount", "Time"]] = scaler.transform(
    X_test[["Amount", "Time"]]
)

# Save prepared datasets
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\n========== FINAL VERIFICATION ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training fraud cases:", int(y_train.sum()))
print("Testing fraud cases:", int(y_test.sum()))

print("\nPreprocessing completed successfully!")
