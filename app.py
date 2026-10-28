
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PROJECT PATHS AND PAGE CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "inventory.db"
MODEL_PATH = BASE_DIR / "models" / "demand_model.pkl"
SALES_PATH = BASE_DIR / "data" / "raw" / "train.csv"
FORECAST_PATH = (
    BASE_DIR / "data" / "processed" / "future_demand_forecast.csv"
)

TABLE_NAME = "inventory_recommendations"

st.set_page_config(
    page_title="Inventory Intelligence",
    page_icon="📦",
    layout="wide",
)


# =========================================================
# 2. DATABASE FUNCTION
# =========================================================

def load_inventory_data():
    """Load inventory data from SQLite."""

    if not DB_PATH.is_file():
        return None, f"Database file not found: {DB_PATH}"

    try:
        connection_string = f"{DB_PATH.as_uri()}?mode=ro"

        with sqlite3.connect(connection_string, uri=True) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn,
            )["name"].tolist()

            if TABLE_NAME not in tables:
                return None, f"Table '{TABLE_NAME}' was not found."

            df = pd.read_sql_query(
                f'SELECT * FROM "{TABLE_NAME}"',
                conn,
            )

        if df.empty:
            return None, "The inventory table contains no records."

        return df, None

    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        return None, f"Database error: {error}"


# =========================================================
# 3. MODEL FUNCTIONS
# =========================================================

@st.cache_resource
def load_demand_model():
    """Load the trained Random Forest model."""

    if not MODEL_PATH.is_file():
        return None, None

    saved = joblib.load(MODEL_PATH)

    return saved["model"], saved["features"]


def predict_demand(df):
    """Generate predictions using existing feature values."""

    try:
        model, features = load_demand_model()

        if model is None:
            return None, (
                "Model not found. Run python src/train_model.py first."
            )

        missing = [
            column for column in features
            if column not in df.columns
        ]

        if missing:
            return None, (
                "Missing model features: " + ", ".join(missing)
            )

        X = df[features].apply(pd.to_numeric, errors="coerce")

        if X.isna().any().any():
            return None, "Some model features contain invalid values."

        result = df.copy()
        result["predicted_demand"] = np.maximum(
            model.predict(X), 0
        )

        return result, None

    except (ValueError, TypeError, KeyError, OSError) as error:
        return None, f"Prediction error: {error}"


def find_column(df, possible_names):
    """Find a column without depending on capitalization."""

    columns_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in columns_lookup:
            return columns_lookup[name.lower()]

    return None


# =========================================================
# 4. LOAD SALES DATA
# =========================================================

if SALES_PATH.is_file():
    try:
        sales_df = pd.read_csv(SALES_PATH)

        required_columns = {"date", "store", "item", "sales"}

        if not required_columns.issubset(sales_df.columns):
            st.error(
                "The sales CSV must contain date, store, item, sales."
            )
            st.stop()

        sales_df["date"] = pd.to_datetime(
            sales_df["date"], errors="coerce"
        )

        sales_df["sales"] = pd.to_numeric(
            sales_df["sales"], errors="coerce"
        )

        sales_df = sales_df.dropna(
            subset=["date", "store", "item", "sales"]
        )

    except (OSError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not load sales data: {error}")
        st.stop()

else:
    sales_df = pd.DataFrame()


# =========================================================
# 5. LOAD INVENTORY DATABASE
# =========================================================

inventory_df, database_error = load_inventory_data()


# =========================================================
# 6. SIDEBAR FILTERS
# =========================================================

st.sidebar.title("📊 Dashboard Filters")

if not sales_df.empty:
    stores = sorted(sales_df["store"].unique().tolist())
    items = sorted(sales_df["item"].unique().tolist())

    selected_stores = st.sidebar.multiselect(
        "Select stores",
        options=stores,
        default=stores,
    )

    selected_items = st.sidebar.multiselect(
        "Select items",
        options=items,
        default=items,
    )

    filtered_sales = sales_df[
        sales_df["store"].isin(selected_stores)
        & sales_df["item"].isin(selected_items)
    ].copy()

else:
    filtered_sales = pd.DataFrame()
    st.sidebar.warning("Sales CSV is unavailable.")


# =========================================================
# 7. DASHBOARD HEADER
# =========================================================

st.title("📦 Inventory Intelligence Dashboard")

st.write(
    "Analyze sales trends, monitor inventory, and review "
    "stockout risks and reorder recommendations."
)

st.divider()


# =========================================================
# 8. SALES OVERVIEW
# =========================================================

st.header("Sales Overview")

if not filtered_sales.empty:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Units Sold",
        f"{filtered_sales['sales'].sum():,.0f}",
    )

    col2.metric(
        "Stores",
        filtered_sales["store"].nunique(),
    )

    col3.metric(
        "Products",
        filtered_sales["item"].nunique(),
    )

    st.subheader("Sales Trend")

    daily_sales = (
        filtered_sales.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    fig_trend = px.line(
        daily_sales,
        x="date",
        y="sales",
        title="Daily Sales",
        labels={
            "date": "Date",
            "sales": "Units Sold",
        },
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True,
    )

    st.subheader("Sales by Store")

    store_sales = (
        filtered_sales.groupby("store", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )

    fig_store = px.bar(
        store_sales,
        x="store",
        y="sales",
        title="Total Sales by Store",
        labels={
            "store": "Store",
            "sales": "Units Sold",
        },
    )

    st.plotly_chart(
        fig_store,
        use_container_width=True,
    )

else:
    st.info("No sales records match the selected filters.")


# =========================================================
# 9. INVENTORY AND STOCKOUT ANALYSIS
# =========================================================

st.divider()
st.header("Inventory & Stockout Analysis")

# These variables are defined outside the database condition
# so that later sections can safely check database availability.

display_df = pd.DataFrame()
risk_col = None
store_col = None
item_col = None

if database_error:
    st.error(database_error)

elif inventory_df is not None:

    st.success("Connected to SQLite successfully!")

    st.caption(
        f"Database: {DB_PATH.name} | Table: {TABLE_NAME}"
    )

    stock_col = find_column(
        inventory_df,
        ["current_stock", "stock", "stock_quantity"],
    )

    store_col = find_column(
        inventory_df,
        ["store", "store_id"],
    )

    item_col = find_column(
        inventory_df,
        ["item", "item_id", "product_id"],
    )

    reorder_col = find_column(
        inventory_df,
        ["reorder_point"],
    )

    risk_col = find_column(
        inventory_df,
        ["stockout_risk", "risk_level", "risk_category", "risk"],
    )

    reorder_qty_col = find_column(
        inventory_df,
        [
            "reorder_quantity",
            "recommended_reorder_quantity",
            "recommended_order_quantity",
        ],
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Inventory Records",
        f"{len(inventory_df):,}",
    )

    col2.metric(
        "Stores in Inventory",
        inventory_df[store_col].nunique()
        if store_col else "N/A",
    )

    col3.metric(
        "Products in Inventory",
        inventory_df[item_col].nunique()
        if item_col else "N/A",
    )

    # -----------------------------------------------------
    # Inventory filters
    # -----------------------------------------------------

    st.subheader("Inventory Recommendations")

    display_df = inventory_df.copy()

    if risk_col:

        risk_values = sorted(
            display_df[risk_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_risks = st.multiselect(
            "Filter by stockout risk",
            options=risk_values,
            default=risk_values,
        )

        display_df = display_df[
            display_df[risk_col].astype(str).isin(selected_risks)
        ]

        risk_counts = (
            display_df[risk_col]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Risk")
            .reset_index(name="Records")
        )

        st.subheader("Stockout Risk Distribution")

        if not risk_counts.empty:
            fig_risk = px.bar(
                risk_counts,
                x="Risk",
                y="Records",
                title="Records by Stockout Risk",
            )

            st.plotly_chart(
                fig_risk,
                use_container_width=True,
            )

    elif stock_col and reorder_col:

        st.info(
            "No stockout-risk column found. "
            "Showing a stock-versus-reorder-point alert."
        )

        display_df["_stock_status"] = np.where(
            pd.to_numeric(
                display_df[stock_col], errors="coerce"
            )
            <= pd.to_numeric(
                display_df[reorder_col], errors="coerce"
            ),
            "Reorder Needed",
            "Stock Above Reorder Point",
        )

        statuses = sorted(
            display_df["_stock_status"].dropna().unique().tolist()
        )

        selected_statuses = st.multiselect(
            "Filter by stock status",
            options=statuses,
            default=statuses,
        )

        display_df = display_df[
            display_df["_stock_status"].isin(selected_statuses)
        ]

        status_counts = (
            display_df["_stock_status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Records")
        )

        st.bar_chart(status_counts.set_index("Status"))

    else:
        st.warning(
            "Required inventory risk columns were not found."
        )

    # -----------------------------------------------------
    # HIGH-RISK INVENTORY ALERTS
    # -----------------------------------------------------

    st.subheader("🚨 High-Risk Inventory Alerts")

    if risk_col:

        high_risk_df = inventory_df[
            inventory_df[risk_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("high")
        ].copy()

        if not high_risk_df.empty:

            st.warning(
                f"{len(high_risk_df)} inventory records "
                "are marked High Risk."
            )

            alert_columns = [
                column
                for column in [
                    store_col,
                    item_col,
                    "predicted_demand",
                    "avg_daily_demand",
                    stock_col,
                    reorder_col,
                    "reorder_quantity",
                    risk_col,
                ]
                if column and column in high_risk_df.columns
            ]

            st.dataframe(
                high_risk_df[alert_columns],
                use_container_width=True,
            )

            st.download_button(
                label="Download High-Risk Inventory CSV",
                data=high_risk_df[alert_columns]
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="high_risk_inventory.csv",
                mime="text/csv",
                key="download_high_risk_inventory",
            )

        else:
            st.success(
                "No high-risk inventory records were found."
            )

    else:
        st.info(
            "The inventory database does not contain a "
            "stockout-risk column."
        )

    # -----------------------------------------------------
    # REORDER RECOMMENDATIONS
    # -----------------------------------------------------

    st.subheader("Reorder Recommendations")

    if reorder_qty_col and item_col:

        reorder_chart_df = inventory_df.copy()

        reorder_chart_df[reorder_qty_col] = pd.to_numeric(
            reorder_chart_df[reorder_qty_col],
            errors="coerce",
        )

        reorder_chart_df = reorder_chart_df.dropna(
            subset=[reorder_qty_col]
        )

        if not reorder_chart_df.empty:

            fig_reorder = px.bar(
                reorder_chart_df,
                x=item_col,
                y=reorder_qty_col,
                title="Recommended Reorder Quantity",
                labels={
                    item_col: "Item",
                    reorder_qty_col: "Reorder Quantity",
                },
            )

            st.plotly_chart(
                fig_reorder,
                use_container_width=True,
            )

    else:
        st.info(
            "A reorder-quantity column was not found."
        )

    # -----------------------------------------------------
    # AI DEMAND PREDICTION
    # -----------------------------------------------------

    st.divider()
    st.header("🤖 AI Demand Forecast")

    predicted_df, prediction_error = predict_demand(
        inventory_df
    )

    if prediction_error:

        st.warning(
            "Demand prediction is currently unavailable: "
            + prediction_error
        )

    elif predicted_df is not None:

        st.success(
            "Demand predictions generated successfully!"
        )

        st.caption(
            "These predictions use existing feature values in "
            "the inventory database. They are not automatically "
            "forecasts for future dates."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Average Predicted Demand",
            f"{predicted_df['predicted_demand'].mean():.2f} units",
        )

        col2.metric(
            "Total Predicted Demand",
            f"{predicted_df['predicted_demand'].sum():,.0f} units",
        )

        prediction_columns = [
            column
            for column in [
                "date",
                "store",
                "item",
                "sales",
                "predicted_demand",
                "current_stock",
                "lead_time_days",
                "safety_stock",
                "reorder_point",
            ]
            if column in predicted_df.columns
        ]

        st.dataframe(
            predicted_df[prediction_columns].head(100),
            use_container_width=True,
        )

        st.download_button(
            label="Download Demand Predictions CSV",
            data=predicted_df.to_csv(index=False)
            .encode("utf-8"),
            file_name="inventory_demand_predictions.csv",
            mime="text/csv",
            key="download_demand_predictions",
        )

    # -----------------------------------------------------
    # FULL INVENTORY RECORDS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Inventory Records")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
    )

    st.download_button(
        label="Download Inventory Records CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="inventory_records.csv",
        mime="text/csv",
        key="download_inventory_records",
    )

    st.caption(
        "Note: current stock values in this project are simulated "
        "assumptions. Use real stock data before making actual "
        "restocking decisions."
    )


# =========================================================
# 10. SEVEN-DAY DEMAND FORECAST
# =========================================================

st.divider()
st.header("📅 7-Day Demand Forecast")

if FORECAST_PATH.is_file():

    try:
        forecast_df = pd.read_csv(
            FORECAST_PATH,
            parse_dates=["date"],
        )

        required_forecast_columns = {
            "date",
            "store",
            "item",
            "predicted_demand",
        }

        if not required_forecast_columns.issubset(
            forecast_df.columns
        ):

            st.error(
                "Forecast CSV must contain date, store, item, "
                "and predicted_demand."
            )

        elif forecast_df.empty:

            st.warning(
                "The forecast file is empty. Generate it again."
            )

        else:

            forecast_df["store"] = pd.to_numeric(
                forecast_df["store"],
                errors="coerce",
            )

            forecast_df["item"] = pd.to_numeric(
                forecast_df["item"],
                errors="coerce",
            )

            forecast_df["predicted_demand"] = pd.to_numeric(
                forecast_df["predicted_demand"],
                errors="coerce",
            )

            forecast_df = forecast_df.dropna(
                subset=[
                    "date",
                    "store",
                    "item",
                    "predicted_demand",
                ]
            )

            if forecast_df.empty:

                st.warning(
                    "No valid forecast records were found."
                )

            else:

                store_options = sorted(
                    forecast_df["store"].unique().tolist()
                )

                selected_store = st.selectbox(
                    "Select Store",
                    options=store_options,
                    key="forecast_store",
                )

                store_df = forecast_df[
                    forecast_df["store"] == selected_store
                ].copy()

                item_options = sorted(
                    store_df["item"].unique().tolist()
                )

                selected_item = st.selectbox(
                    "Select Item",
                    options=item_options,
                    key="forecast_item",
                )

                selected_forecast = store_df[
                    store_df["item"] == selected_item
                ].sort_values("date")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Total Predicted Demand (7 Days)",
                    f"{selected_forecast['predicted_demand'].sum():.2f} units",
                )

                col2.metric(
                    "Average Daily Predicted Demand",
                    f"{selected_forecast['predicted_demand'].mean():.2f} units",
                )

                fig_forecast = px.line(
                    selected_forecast,
                    x="date",
                    y="predicted_demand",
                    markers=True,
                    title=(
                        f"7-Day Demand Forecast — "
                        f"Store {int(selected_store)}, "
                        f"Item {int(selected_item)}"
                    ),
                    labels={
                        "date": "Forecast Date",
                        "predicted_demand": "Predicted Demand",
                    },
                )

                st.plotly_chart(
                    fig_forecast,
                    use_container_width=True,
                )

                st.subheader("Daily Forecast Records")

                display_forecast = selected_forecast.copy()

                display_forecast["date"] = (
                    display_forecast["date"]
                    .dt.strftime("%Y-%m-%d")
                )

                st.dataframe(
                    display_forecast,
                    use_container_width=True,
                )

                st.download_button(
                    label="Download 7-Day Forecast CSV",
                    data=selected_forecast.to_csv(index=False)
                    .encode("utf-8"),
                    file_name="seven_day_demand_forecast.csv",
                    mime="text/csv",
                    key="download_seven_day_forecast",
                )

                st.caption(
                    "The dataset ends on 2017-12-31, so this forecast "
                    "covers January 1–7, 2018. These are historical "
                    "model estimates, not current 2026 forecasts."
                )

    except (OSError, pd.errors.ParserError, ValueError) as error:

        st.error(
            f"Could not read the forecast file: {error}"
        )

else:

    st.warning(
        "Forecast file not found. Run "
        "python src/forecast_demand.py from your project folder."
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Inventory & Stockout Prediction System | "
    "Streamlit • SQLite • Random Forest"
)
