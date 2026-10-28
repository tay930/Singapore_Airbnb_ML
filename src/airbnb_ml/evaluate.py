import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = joblib.load('models/linear_regression.joblib')

X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test = pd.read_pickle('data/processed/y_test.pkl')

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²:    {r2:.4f}")

residuals = y_test.to_numpy() - predictions

plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.5)
lims = [
    min(y_test.min(), predictions.min()),
    max(y_test.max(), predictions.max())
]
plt.plot(lims, lims, "--")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted Airbnb Prices")
plt.tight_layout()
plt.savefig("data/processed/linear_regression_actual_vs_predicted.png",
            dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted price")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals vs Predicted Prices")
plt.tight_layout()
plt.savefig("data/processed/linear_regression_residuals_vs_predicted.png",
            dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 6))
plt.hist(residuals, bins=40)
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig("data/processed/linear_regression_residual_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()

results = pd.DataFrame([{
    "model": "Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "actual_vs_predicted": "linear_regression_actual_vs_predicted.png",
    "residuals_vs_predicted": "linear_regression_residuals_vs_predicted.png",
    "residual_distribution": "linear_regression_residual_distribution.png"
}])

results.to_csv(
    "data/processed/linear_regression_model_results.csv",
    mode="a",
    header=not __import__("os").path.exists(
        "data/processed/model_results.csv"
    ),
    index=False
)