
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load prediction results
file_path = PROJECT_ROOT / "data" / "processed" / "predictions.csv"
df = pd.read_csv(file_path, parse_dates=["date"])

# Calculate evaluation metrics
actual = df["sales"]
predicted = df["predicted_sales"]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

print("MODEL EVALUATION RESULTS")
print("------------------------")
print(f"Total predictions: {len(df)}")
print(f"MAE: {mae:.2f} units")
print(f"RMSE: {rmse:.2f} units")

# Show a sample of actual vs predicted sales
print("\nSample predictions:")
print(
    df[["date", "store", "item", "sales", "predicted_sales"]]
    .head(10)
    .to_string(index=False)
)

# Compare actual and predicted sales for a sample
sample = df.head(100).copy()

plt.figure(figsize=(12, 5))
plt.plot(sample["sales"].to_numpy(), label="Actual Sales")
plt.plot(sample["predicted_sales"].to_numpy(), label="Predicted Sales")
plt.title("Actual vs Predicted Sales (100 test records)")
plt.xlabel("Test record")
plt.ylabel("Units sold")
plt.legend()
plt.tight_layout()
plt.show()
