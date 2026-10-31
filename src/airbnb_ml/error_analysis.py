import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("models/tuned_log_random_forest.joblib")

X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test = pd.read_pickle('data/processed/y_test.pkl')

predictions_log = model.predict(X_test)

# Convert log prices back to dollars
actual = y_test
predicted = np.exp(predictions_log)

# Build an error analysis DataFrame
errors = X_test.copy()

errors["actual_price"] = np.asarray(actual)
errors["predicted_price"] = predicted
errors["residual"] = errors["actual_price"] - errors["predicted_price"]
errors["absolute_error"] = errors["residual"].abs()
errors["percentage_error"] = (
    errors["absolute_error"] / errors["actual_price"] * 100
)

errors.head()

plt.figure(figsize=(8, 6))

plt.scatter(
    errors["actual_price"],
    errors["predicted_price"],
    alpha=0.5
)

lims = [
    min(errors["actual_price"].min(), errors["predicted_price"].min()),
    max(errors["actual_price"].max(), errors["predicted_price"].max())
]

plt.plot(lims, lims, "--", color="red")

plt.xlabel("Actual nightly price ($)")
plt.ylabel("Predicted nightly price ($)")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(
    errors["actual_price"],
    errors["residual"],
    alpha=0.5
)

plt.axhline(0, linestyle="--", color="red")

plt.xlabel("Actual nightly price ($)")
plt.ylabel("Residual ($)")
plt.title("Residuals vs Actual Prices")
plt.tight_layout()
plt.show()

errors["price_group"] = pd.qcut(
    errors["actual_price"],
    q=5
)

price_analysis = errors.groupby(
    "price_group",
    observed=True
).agg(
    listings=("actual_price", "count"),
    mean_actual=("actual_price", "mean"),
    mean_predicted=("predicted_price", "mean"),
    MAE=("absolute_error", "mean"),
    mean_residual=("residual", "mean")
)

print(price_analysis)

price_analysis["MAE"].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.xlabel("Actual price group")
plt.ylabel("MAE ($)")
plt.title("Prediction Error by Price Range")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

largest_errors = errors.nlargest(
    10,
    "absolute_error"
)

columns = [
    "actual_price",
    "predicted_price",
    "residual",
    "absolute_error",
    "accommodates",
    "bedrooms",
    "neighbourhood_cleansed"
]

print(largest_errors[columns].to_string(index=False))