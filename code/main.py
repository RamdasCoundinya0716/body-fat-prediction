import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Configure visual styles
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)

# Load dataset
df = pd.read_csv("E:/Projects/mlops/body-fat-prediction/bodyfat.csv")

# Basic pandas EDA
print("Dataset shape:", df.shape)
print("\nSummary statistics:\n", df.describe().T)

# Check feature skewness (to identify asymmetry in data distribution)
print("\nFeature Skewness:\n", df.skew().sort_values(ascending=False))

# Plot histograms for key numerical features
key_features = ["bodyfat", "weight_kg", "height_in", "abdomen", "age"]
df[key_features].hist(bins=25, figsize=(12, 8), color="skyblue", edgecolor="black")
plt.suptitle("Distribution of Key Numerical Features", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# Plot boxplots to check for outliers in the same key features
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[key_features])
plt.title("Boxplots for Key Numerical Features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()

X = df.drop(columns=["bodyfat", "density"])
y = df["bodyfat"]

# Split the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scaler
scaler = StandardScaler()

# Fit on training data, transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability (optional)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Quick check
print("Training data shape:", X_train_scaled.shape)
print("Test data shape:", X_test_scaled.shape)
print("\nFeature means after scaling (should be ~0):\n", X_train_scaled.mean().round(2))
print("\nFeature std devs after scaling (should be ~1):\n", X_train_scaled.std().round(2))

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_lr = lin_reg.predict(X_test_scaled)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2 = r2_score(y_test, y_pred_lr)

print("📊 Linear Regression Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# Display predicted vs actual comparison
comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred_lr[:10].round(2)
})
print("\nSample Predictions:\n", comparison)

joblib.dump(lin_reg, "linear_model_bodyfat.pkl")
joblib.dump(scaler, "scaler_bodyfat.pkl")

