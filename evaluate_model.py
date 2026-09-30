import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
# Input paths
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"
# Model path
MODEL_PATH = "models/random_forest_model.pkl"
# Load testing data
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()
# Load trained model
model = joblib.load(MODEL_PATH)
print("Testing data loaded successfully.")
print("Testing feature shape:", X_test.shape)
print("\nGenerating predictions...")
# Predict
y_pred = model.predict(X_test)
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    pos_label=-1
)
recall = recall_score(
    y_test,
    y_pred,
    pos_label=-1
)
f1 = f1_score(
    y_test,
    y_pred,
    pos_label=-1
)
print("\nMODEL PERFORMANCE")
print("----------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[-1, 1],
        target_names=[
            "Phishing",
            "Legitimate"
        ]
    )
)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save evaluation results
os.makedirs("reports", exist_ok=True)
results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})
results.to_csv(
    "reports/model_evaluation.csv",
    index=False
)
print("\nEvaluation results saved to reports/model_evaluation.csv")