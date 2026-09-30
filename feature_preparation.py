import pandas as pd
import os
from sklearn.model_selection import train_test_split
# Dataset path
DATA_PATH = "data/processed/phishing_processed.csv"
# Output paths
X_TRAIN_PATH = "data/processed/X_train.csv"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"
Y_TEST_PATH = "data/processed/y_test.csv"
# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]
print("\nNumber of input features:", X.shape[1])
print("Target column: class")
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# Create processed directory
os.makedirs("data/processed", exist_ok=True)
# Save training and testing data
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)
print("\nFeature preparation completed.")
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)
print("\nTraining and testing datasets saved successfully.")
