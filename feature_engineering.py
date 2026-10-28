
import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the original dataset
input_path = PROJECT_ROOT / "data" / "raw" / "train.csv"

# Create processed data folder
output_folder = PROJECT_ROOT / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path, parse_dates=["date"])

# Sort records before creating time-based features
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day

# Previous sales for the same store and item
grouped_sales = df.groupby(["store", "item"])["sales"]

df["sales_lag_7"] = grouped_sales.shift(7)
df["sales_lag_14"] = grouped_sales.shift(14)

# Rolling averages use only earlier sales, avoiding current-day leakage
df["rolling_mean_7"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda s: s.shift(1).rolling(7).mean())
)

df["rolling_mean_30"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda s: s.shift(1).rolling(30).mean())
)

# Remove rows without sufficient historical data
df = df.dropna().reset_index(drop=True)

# Save processed data
output_path = output_folder / "inventory_features.csv"
df.to_csv(output_path, index=False)

print("Feature engineering completed successfully!")
print("Saved to:", output_path)
print("Processed dataset shape:", df.shape)
print("\nNew columns:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())
