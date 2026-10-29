import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_name = "tuned_random_forest"
figures_dir = Path("reports/figures") / model_name
figures_dir.mkdir(parents=True, exist_ok=True)

model = joblib.load('models/tuned_random_forest.joblib')

X_test = pd.read_pickle('data/processed/X_test.pkl')
y_test_actual = pd.read_pickle('data/processed/y_test.pkl')

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
r2 = r2_score(y_test_actual, predictions)

print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²:    {r2:.4f}")

residuals = y_test_actual.to_numpy() - predictions

plt.figure(figsize=(7, 6))
plt.scatter(y_test_actual, predictions, alpha=0.5)
lims = [
    min(y_test_actual.min(), predictions.min()),
    max(y_test_actual.max(), predictions.max())
]
plt.plot(lims, lims, "--")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual vs Predicted Airbnb Prices")
plt.tight_layout()
plt.savefig(figures_dir / "actual_vs_predicted.png",
            dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(predictions, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted price")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals vs Predicted Prices")
plt.tight_layout()
plt.savefig(figures_dir / "residuals_vs_predicted.png",
            dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 6))
plt.hist(residuals, bins=40)
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(figures_dir / "residual_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()

results = pd.DataFrame([{
    "model": "Tuned RF",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "actual_vs_predicted": str(figures_dir / "actual_vs_predicted.png"),
    "residuals_vs_predicted": str(figures_dir / "residuals_vs_predicted.png"),
    "residual_distribution": str(figures_dir / "residual_distribution.png")
}])

results_path = Path("data/processed/model_comparison.csv")

if results_path.exists():
    existing = pd.read_csv(results_path)
    existing = existing[existing["model"] != results.iloc[0]["model"]]
    results = pd.concat([existing, results], ignore_index=True)

results.to_csv(results_path, index=False)
