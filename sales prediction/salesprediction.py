import pandas as pd

# Load the dataset
data = pd.read_csv("../data/Advertising.csv")

# Display first 5 rows
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display basic statistics
print("\nBasic Statistics:")
print(data.describe())
# Remove unnecessary index column
data = data.drop("Unnamed: 0", axis=1)

print("\nDataset after cleaning:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())
# Correlation analysis
print("\nCorrelation Matrix:")
print(data.corr())
import matplotlib.pyplot as plt

# TV vs Sales
plt.scatter(data["TV"], data["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()

# Radio vs Sales
plt.scatter(data["Radio"], data["Sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.show()

# Newspaper vs Sales
plt.scatter(data["Newspaper"], data["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.show()
# ============================================================
# SALES PREDICTION USING PYTHON
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("../data/Advertising.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(data.head())


# ============================================================
# 2. DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(data.info())


print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print(data.describe())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Remove unnecessary index column
data = data.drop("Unnamed: 0", axis=1)

# Remove duplicate rows
data = data.drop_duplicates()

print("\n" + "=" * 60)
print("DATA AFTER CLEANING")
print("=" * 60)

print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

correlation = data.corr()
print(correlation)


# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            round(correlation.iloc[i, j], 2),
            ha="center",
            va="center"
        )

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("../outputs/correlation_matrix.png")

plt.show()


# ============================================================
# 6. TV VS SALES
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["TV"],
    data["Sales"],
    alpha=0.7
)

plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")

plt.tight_layout()

plt.savefig("../outputs/tv_vs_sales.png")

plt.show()


# ============================================================
# 7. RADIO VS SALES
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Radio"],
    data["Sales"],
    alpha=0.7
)

plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")

plt.tight_layout()

plt.savefig("../outputs/radio_vs_sales.png")

plt.show()


# ============================================================
# 8. NEWSPAPER VS SALES
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Newspaper"],
    data["Sales"],
    alpha=0.7
)

plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")

plt.tight_layout()

plt.savefig("../outputs/newspaper_vs_sales.png")

plt.show()


# ============================================================
# 9. DEFINE FEATURES AND TARGET
# ============================================================

X = data[
    [
        "TV",
        "Radio",
        "Newspaper"
    ]
]

y = data["Sales"]


# ============================================================
# 10. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 11. FEATURE SELECTION USING RFE
# ============================================================

rfe_model = LinearRegression()

rfe = RFE(
    estimator=rfe_model,
    n_features_to_select=2
)

rfe.fit(X_train, y_train)

feature_selection = pd.DataFrame({
    "Feature": X.columns,
    "Selected": rfe.support_,
    "Ranking": rfe.ranking_
})

print("\n" + "=" * 60)
print("FEATURE SELECTION USING RFE")
print("=" * 60)

print(feature_selection)

selected_features = X.columns[rfe.support_]

print("\nSelected Features:")
print(list(selected_features))


# ============================================================
# 12. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(X_test)


# ============================================================
# 13. RANDOM FOREST
# ============================================================

random_forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=8
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_predictions = random_forest_model.predict(X_test)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)


random_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

random_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        random_forest_predictions
    )
)

random_r2 = r2_score(
    y_test,
    random_forest_predictions
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        random_mae
    ],

    "RMSE": [
        linear_rmse,
        random_rmse
    ],

    "R2 Score": [
        linear_r2,
        random_r2
    ]

})

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(results.round(4))

results.to_csv(
    "../outputs/model_results.csv",
    index=False
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

if random_r2 > linear_r2:

    best_model = random_forest_model
    best_predictions = random_forest_predictions
    best_model_name = "Random Forest"

else:

    best_model = linear_model
    best_predictions = linear_predictions
    best_model_name = "Linear Regression"


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(best_model_name)

print("R2 Score:", round(
    r2_score(y_test, best_predictions),
    4
))


# ============================================================
# 17. ACTUAL VS PREDICTED SALES
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.8
)

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.title(
    "Actual vs Predicted Sales - " + best_model_name
)

plt.tight_layout()

plt.savefig(
    "../outputs/actual_vs_predicted.png"
)

plt.show()


# ============================================================
# 18. RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - best_predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    best_predictions,
    residuals,
    alpha=0.8
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
    "../outputs/residual_plot.png"
)

plt.show()


# ============================================================
# 19. LINEAR REGRESSION COEFFICIENTS
# ============================================================

coefficients = pd.DataFrame({

    "Feature": X.columns,

    "Coefficient": linear_model.coef_

})

coefficients = coefficients.sort_values(
    "Coefficient",
    ascending=False
)

print("\n" + "=" * 60)
print("LINEAR REGRESSION COEFFICIENTS")
print("=" * 60)

print(coefficients.round(4))

coefficients.to_csv(
    "../outputs/linear_coefficients.csv",
    index=False
)


# ============================================================
# 20. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": random_forest_model.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(importance.round(4))


# ============================================================
# 21. MARKETING BUDGET SCENARIO PREDICTION
# ============================================================

scenarios = pd.DataFrame({

    "TV": [
        200,
        250,
        150
    ],

    "Radio": [
        30,
        40,
        25
    ],

    "Newspaper": [
        20,
        20,
        10
    ]

})

scenarios["Predicted Sales"] = best_model.predict(
    scenarios[
        [
            "TV",
            "Radio",
            "Newspaper"
        ]
    ]
)

print("\n" + "=" * 60)
print("MARKETING BUDGET SCENARIOS")
print("=" * 60)

print(
    scenarios.round(2)
)

scenarios.to_csv(
    "../outputs/scenario_predictions.csv",
    index=False
)


# ============================================================
# 22. FINAL BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

print("""
1. TV advertising has the strongest correlation with Sales.

2. Radio advertising also has a positive relationship with Sales.

3. Newspaper advertising has a comparatively weaker relationship
   with Sales.

4. Random Forest provides a stronger predictive performance than
   Linear Regression on the test data.

5. Marketing teams can use the model to compare different
   advertising-budget scenarios.

6. The model is a decision-support tool. Correlation and prediction
   do not prove that advertising spend alone causes sales changes.
""")


print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
