
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "inventory_features.csv"
)

MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.pkl"

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "inventory_recommendations.csv"
)


# =========================================================
# 2. LOAD HISTORICAL DATA
# =========================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Feature dataset not found: {DATA_PATH}")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run python src/train_model.py first."
    )

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

required_columns = [
    "date",
    "store",
    "item",
    "sales",
    "sales_lag_7",
    "sales_lag_14",
    "rolling_mean_7",
    "rolling_mean_30",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in feature dataset: {missing_columns}"
    )

df = df.dropna(subset=["date"]).copy()

df = df.sort_values("date")


# =========================================================
# 3. GET LATEST RECORD FOR EACH STORE AND ITEM
# =========================================================

latest = (
    df.groupby(["store", "item"], as_index=False)
    .tail(1)
    .copy()
    .reset_index(drop=True)
)

if latest.empty:
    raise ValueError("No usable store/item records were found.")


# =========================================================
# 4. LOAD TRAINED RANDOM FOREST MODEL
# =========================================================

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
features = saved_model["features"]

missing_features = [
    col for col in features if col not in latest.columns
]

if missing_features:
    raise ValueError(
        f"Model features are missing from the dataset: {missing_features}"
    )

X = latest[features].apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError(
        "Some model features contain missing or invalid values. "
        "Clean the feature dataset before predicting."
    )


# =========================================================
# 5. PREDICT DEMAND
# =========================================================

latest["predicted_demand"] = np.maximum(
    model.predict(X),
    0
)

# Combine model prediction with recent average demand.
# This smooths the estimate instead of relying on one value.
latest["avg_daily_demand"] = (
    latest["predicted_demand"] * 0.5
    + latest["rolling_mean_7"] * 0.5
).clip(lower=0)


# =========================================================
# 6. INVENTORY ASSUMPTIONS FOR DEMONSTRATION
# =========================================================

# IMPORTANT:
# These stock quantities are simulated, not actual stock measurements.
latest["current_stock"] = (
    latest["avg_daily_demand"]
    * latest["store"].map(
        lambda store: (
            1 if store % 4 == 0
            else 2 if store % 4 == 1
            else 5 if store % 4 == 2
            else 8
        )
    )
).round().clip(lower=0)

# Assumed supplier lead time.
latest["lead_time_days"] = 3

# Two days of estimated demand held as safety stock.
latest["safety_stock"] = (
    latest["avg_daily_demand"] * 2
).round()


# =========================================================
# 7. CALCULATE REORDER POINT
# =========================================================

latest["lead_time_demand"] = (
    latest["avg_daily_demand"]
    * latest["lead_time_days"]
).round()

latest["reorder_point"] = (
    latest["lead_time_demand"]
    + latest["safety_stock"]
).round()


# =========================================================
# 8. CALCULATE STOCKOUT RISK
# =========================================================

latest["stockout_risk"] = np.select(
    [
        latest["current_stock"] < latest["lead_time_demand"],
        latest["current_stock"] <= latest["reorder_point"],
    ],
    [
        "High",
        "Medium",
    ],
    default="Low",
)


# =========================================================
# 9. CALCULATE REORDER QUANTITY
# =========================================================

latest["reorder_quantity"] = (
    latest["reorder_point"] - latest["current_stock"]
).clip(lower=0).round().astype(int)


# =========================================================
# 10. SAVE INVENTORY RECOMMENDATIONS
# =========================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

latest.to_csv(OUTPUT_PATH, index=False)


# =========================================================
# 11. DISPLAY RESULTS
# =========================================================

print("\nINVENTORY ANALYSIS COMPLETED")
print("-" * 40)

print("Store/item combinations analyzed:", len(latest))
print("Demand model: Random Forest")
print("Output file:", OUTPUT_PATH)

print("\nStockout risk summary:")
print(
    latest["stockout_risk"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

print("\nSample recommendations:")

display_columns = [
    "store",
    "item",
    "predicted_demand",
    "avg_daily_demand",
    "current_stock",
    "lead_time_days",
    "safety_stock",
    "reorder_point",
    "stockout_risk",
    "reorder_quantity",
]

print(
    latest[display_columns]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\nNOTE: Current stock and lead time use demonstration assumptions.")
