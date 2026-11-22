
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

# Load preprocessed datasets
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Train and evaluate
for name, model in models.items():

    print("\n" + "=" * 50)
    print("MODEL:", name)
    print("=" * 50)

    # Training
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix figure
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraudulent"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        f"outputs/models/{filename}_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # Save trained model
    joblib.dump(model, f"models/{filename}.joblib")

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "PR-AUC": pr_auc
    })

# Save comparison table
results_df = pd.DataFrame(results)
results_df.to_csv(
    "outputs/models/model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nModel training and evaluation completed!")
