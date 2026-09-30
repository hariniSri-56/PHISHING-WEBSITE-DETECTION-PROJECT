import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load cleaned dataset
df = pd.read_csv("data/processed/phishing_cleaned.csv")
# Display class distribution
print(df["class"].value_counts())
# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x="class", data=df)
plt.title("Phishing vs Legitimate Websites")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")

plt.show()
