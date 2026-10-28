
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "inventory_features.csv"
)

MODEL_PATH = BASE_DIR / "models" / "demand_model.pkl"

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "future_demand_forecast.csv"
)


# =========================================================
# 2. LOAD DATA AND MODEL
# =========================================================

def main():

    print("Loading inventory data...")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    df = df.sort_values(["store", "item", "date"]).reset_index(
        drop=True
    )

    saved_model = joblib.load(MODEL_PATH)

    model = saved_model["model"]
    features = saved_model["features"]

    print("Data loaded successfully.")
    print("Latest historical date:", df["date"].max())


    # =====================================================
    # 3. PREPARE HISTORICAL SALES FOR EACH STORE AND ITEM
    # =====================================================

    history = {}

    for (store, item), group in df.groupby(["store", "item"]):

        group = group.sort_values("date")

        history[(store, item)] = (
            group["sales"].dropna().astype(float).tolist()
        )

    latest_date = df["date"].max()

    forecast_records = []


    # =====================================================
    # 4. GENERATE 7-DAY RECURSIVE FORECAST
    # =====================================================

    for day in range(1, 8):

        forecast_date = latest_date + pd.Timedelta(days=day)

        daily_records = []

        for (store, item), sales_history in history.items():

            if len(sales_history) < 30:
                continue

            # Generate features using the latest available history.
            row = {
                "store": store,
                "item": item,
                "year": forecast_date.year,
                "month": forecast_date.month,
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "sales_lag_7": sales_history[-7],
                "sales_lag_14": sales_history[-14],
                "rolling_mean_7": float(np.mean(sales_history[-7:])),
                "rolling_mean_30": float(np.mean(sales_history[-30:])),
            }

            # Ensure the model receives features in training order.
            X = pd.DataFrame(
                [[row[feature] for feature in features]],
                columns=features,
            )

            predicted_demand = max(
                0.0,
                float(model.predict(X)[0]),
            )

            record = {
                "date": forecast_date,
                "store": store,
                "item": item,
                "predicted_demand": predicted_demand,
            }

            daily_records.append(record)

        # Save predictions for the current day.
        forecast_records.extend(daily_records)

        # Add the predicted sales to history so later forecast
        # days can use the updated lag and rolling features.
        for record in daily_records:

            key = (record["store"], record["item"])

            history[key].append(record["predicted_demand"])


    # =====================================================
    # 5. SAVE FORECAST
    # =====================================================

    forecast_df = pd.DataFrame(forecast_records)

    if forecast_df.empty:
        print("No forecasts were generated.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(OUTPUT_PATH, index=False)

    print("\n7-day recursive forecast generated successfully!")
    print("Forecast records:", len(forecast_df))
    print("Forecast starts:", forecast_df["date"].min().date())
    print("Forecast ends:", forecast_df["date"].max().date())
    print("Saved to:", OUTPUT_PATH)

    print("\nSample forecasts:")
    print(forecast_df.head(10).to_string(index=False))

    print("\nTotal predicted demand by forecast date:")

    daily_totals = (
        forecast_df.groupby("date")["predicted_demand"]
        .sum()
        .round(2)
    )

    print(daily_totals)


if __name__ == "__main__":
    main()
