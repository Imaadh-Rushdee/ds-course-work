
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

# Create output folder
output_dir = Path("outputs/eda")
output_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# VISUALISATION 1: CLASS DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Class")

plt.title("Distribution of Legitimate and Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.yscale("log")

plt.tight_layout()
plt.savefig(output_dir / "01_class_distribution.png", dpi=300)
plt.close()


# ==========================================
# VISUALISATION 2: TRANSACTION AMOUNT BOXPLOT
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Class", y="Amount")

plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount")
plt.yscale("symlog", linthresh=1)

plt.tight_layout()
plt.savefig(output_dir / "02_amount_boxplot.png", dpi=300)
plt.close()


# ==========================================
# VISUALISATION 3: AMOUNT DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

for class_value, label in [(0, "Legitimate"), (1, "Fraudulent")]:
    sns.histplot(
        data=df[df["Class"] == class_value],
        x="Amount",
        bins=50,
        stat="density",
        element="step",
        fill=False,
        label=label
    )

plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")
plt.xlim(0, 2000)
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / "03_amount_distribution.png", dpi=300)
plt.close()


# ==========================================
# VISUALISATION 4: TRANSACTION TIME
# ==========================================

plt.figure(figsize=(10, 5))

for class_value, label in [(0, "Legitimate"), (1, "Fraudulent")]:
    sns.histplot(
        data=df[df["Class"] == class_value],
        x="Time",
        bins=50,
        stat="density",
        element="step",
        fill=False,
        label=label
    )

plt.title("Transaction Time Distribution by Class")
plt.xlabel("Elapsed Time (seconds)")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / "04_time_distribution.png", dpi=300)
plt.close()


# ==========================================
# VISUALISATION 5: CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(16, 12))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title("Correlation Heatmap of Dataset Features")

plt.tight_layout()
plt.savefig(output_dir / "05_correlation_heatmap.png", dpi=300)
plt.close()


print("EDA completed successfully!")
print("Total visualisations generated: 5")
print("Saved in:", output_dir)
