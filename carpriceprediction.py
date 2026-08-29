import pandas as pd

# CSV file load karna
df = pd.read_csv("car data.csv")

# First 5 rows
print(df.head())

# Dataset ka size
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Dataset ki basic information
print("\nDataset Information:")
df.info()

# Missing values check
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows check
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Duplicate rows remove karna
df = df.drop_duplicates()

print("\nDataset Shape after removing duplicates:")
print(df.shape)

# Features (input)
X = df.drop("Selling_Price", axis=1)

# Target (output)
y = df["Selling_Price"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data size:")
print(X_train.shape)

print("Testing data size:")
print(X_test.shape)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Categorical columns
categorical_features = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

# Numerical columns
numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

print("\nPreprocessing setup completed!")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Linear Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# Model ko training data se train karna
model.fit(X_train, y_train)

print("\nModel training completed!")
# Test data par prediction
y_pred = model.predict(X_test)

print("\nPredicted Selling Prices:")
print(y_pred)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n----- Model Performance -----")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)
import matplotlib.pyplot as plt

# Actual vs Predicted prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()
# Example: ek car ki selling price predict karna

sample_car = X_test.iloc[[0]]

predicted_price = model.predict(sample_car)

print("\n----- Example Prediction -----")
print("Car Details:")
print(sample_car)

print("\nPredicted Selling Price:", predicted_price[0])
print("Actual Selling Price:", y_test.iloc[0])

