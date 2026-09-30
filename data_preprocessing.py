import pandas as pd
# Load the dataset
df = pd.read_csv("data/raw/phishing.csv")
# Display first five rows
print(df.head())
# Dataset shape
print("Dataset Shape:", df.shape)
# Column names
print("Columns:")
print(df.columns)
# Data types
print("Data Types:")
print(df.dtypes)
# Missing values
print("Missing Values:")
print(df.isnull().sum())
# Duplicate rows
print("Duplicate Rows:", df.duplicated().sum())
# Remove Index column
df = df.drop("Index", axis=1)
# Save cleaned dataset
df.to_csv("data/processed/phishing_cleaned.csv", index=False)
print("Data preprocessing completed successfully!")
