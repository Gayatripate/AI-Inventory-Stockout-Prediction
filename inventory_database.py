
import sqlite3
import pandas as pd
from pathlib import Path


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inventory_recommendations.csv"
)

DB_PATH =  PROJECT_ROOT / "data" / "inventory.db"

TABLE_NAME = "inventory_recommendations"


# =====================================================
# 2. CHECK INPUT FILE
# =====================================================

if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Recommendations CSV not found: {CSV_PATH}\n"
        "Run: python src/inventory_logic.py"
    )


# =====================================================
# 3. LOAD AND VALIDATE DATA
# =====================================================

df = pd.read_csv(CSV_PATH)

if df.empty:
    raise ValueError(
        "The recommendations CSV is empty. "
        "Check your inventory logic and input data."
    )

required_columns = [
    "store",
    "item",
    "avg_daily_demand",
    "current_stock",
    "lead_time_days",
    "safety_stock",
    "reorder_point",
    "stockout_risk",
    "reorder_quantity",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =====================================================
# 4. CREATE DATABASE FOLDER
# =====================================================

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# =====================================================
# 5. SAVE DATA TO SQLITE SAFELY
# =====================================================

with sqlite3.connect(DB_PATH, timeout=30) as connection:

    # Replace the table only after validation succeeds.
    df.to_sql(
        TABLE_NAME,
        connection,
        if_exists="replace",
        index=False
    )

    # Confirm that the data was inserted.
    row_count = connection.execute(
        f'SELECT COUNT(*) FROM "{TABLE_NAME}"'
    ).fetchone()[0]

    # Query risk summary.
    query1 = """
        SELECT stockout_risk, COUNT(*) AS product_count
        FROM inventory_recommendations
        GROUP BY stockout_risk
        ORDER BY product_count DESC;
    """

    risk_summary = pd.read_sql_query(
        query1,
        connection
    )

    # Query high-risk products.
    query2 = """
        SELECT store, item, current_stock,
               reorder_point, reorder_quantity,
               predicted_demand, stockout_risk
        FROM inventory_recommendations
        WHERE stockout_risk = 'High'
        ORDER BY reorder_quantity DESC
        LIMIT 10;
    """

    high_risk_products = pd.read_sql_query(
        query2,
        connection
    )


# =====================================================
# 6. DISPLAY RESULTS
# =====================================================

print("\nDATABASE UPDATED SUCCESSFULLY")
print("-" * 40)
print("Database:", DB_PATH)
print("Table:", TABLE_NAME)
print("Records saved:", row_count)

print("\nProducts by stockout risk:")
print(risk_summary.to_string(index=False))

print("\nTop 10 high-risk products:")
if high_risk_products.empty:
    print("No high-risk products found.")
else:
    print(high_risk_products.to_string(index=False))

print("\nSQLite connection closed successfully.")
