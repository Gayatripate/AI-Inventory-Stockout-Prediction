
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Set project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "inventory_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load processed data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Dataset loaded:", df.shape)

# 3. Define input features and target
features = [
    "store",
    "item",
    "year",
    "month",
    "day_of_week",
    "day_of_month",
    "sales_lag_7",
    "sales_lag_14",
    "rolling_mean_7",
    "rolling_mean_30"
]

target = "sales"

# 4. Sort by date
df = df.sort_values("date").reset_index(drop=True)

# 5. Chronological train-test split
split_date = df["date"].quantile(0.8)

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Split date:", split_date)

# 6. Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 7. Predict sales on test data
predictions = model.predict(X_test)
predictions = np.maximum(predictions, 0)

# 8. Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n========== MODEL RESULTS ==========")
print(f"MAE: {mae:.2f} units")
print(f"RMSE: {rmse:.2f} units")

# 9. Save model and feature names
model_path = MODEL_DIR / "demand_model.pkl"
joblib.dump(
    {"model": model, "features": features},
    model_path
)

print("\nModel saved successfully!")
print("Location:", model_path)

# 10. Show actual vs predicted sales
results = test_df[["date", "store", "item", "sales"]].copy()
results["predicted_sales"] = predictions
results["prediction_error"] = (
    results["sales"] - results["predicted_sales"]
)

results_path = PROJECT_ROOT / "data" / "processed" / "predictions.csv"
results.to_csv(results_path, index=False)

print("Predictions saved to:", results_path)
print("\nSample predictions:")
print(results.head(10).to_string(index=False))
