import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
# Input paths
X_TRAIN_PATH = "data/processed/X_train.csv"
Y_TRAIN_PATH = "data/processed/y_train.csv"
# Model output path
MODEL_PATH = "models/random_forest_model.pkl"
# Load training data
X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
print("Training data loaded successfully.")
print("Training feature shape:", X_train.shape)
print("Training target shape:", y_train.shape)
# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
print("\nTraining Random Forest model...")
# Train model
model.fit(X_train, y_train)
print("Model training completed successfully.")
# Create model directory
os.makedirs("models", exist_ok=True)
# Save model
joblib.dump(model, MODEL_PATH)
print("Model saved successfully.")
print("Model path:", MODEL_PATH)
