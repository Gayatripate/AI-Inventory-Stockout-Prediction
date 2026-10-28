# AI-Based Inventory & Stockout Prediction System

## Project Overview
The AI-Based Inventory & Stockout Prediction System uses machine learning to analyze historical sales, predict product demand, and support inventory planning. It includes a Streamlit dashboard for exploring demand forecasts, stockout risk, and reorder recommendations.

## Objectives
- Predict product demand using historical sales data.
- Analyze inventory and identify potentially high-risk items.
- Calculate reorder recommendations using estimated demand and safety stock.
- Visualize inventory insights through an interactive dashboard.
- Generate a 7-day demand forecast for stores and items.

## Technologies Used
- **Programming:** Python
- **Machine Learning:** Scikit-learn, Random Forest Regressor
- **Data Analysis:** Pandas, NumPy
- **Dashboard:** Streamlit
- **Visualizations:** Plotly
- **Database:** SQLite
- **Model Storage:** Joblib

## Key Features
- Historical sales analysis
- Machine learning-based demand prediction
- Inventory risk classification: High, Medium, and Low
- Reorder quantity recommendations
- SQLite storage for inventory recommendations
- Interactive charts and inventory summaries
- Seven-day demand forecasting
- CSV data export

## Model Evaluation
The model was evaluated on 180,000 test predictions.

- **Mean Absolute Error (MAE):** 6.25 units
- **Root Mean Squared Error (RMSE):** 8.15 units

These metrics describe prediction errors on the evaluation dataset. Further baseline comparison and forecast backtesting are needed to establish how well the model generalizes.

## Project Structure

```text
AI_Inventory_Stockout_Prediction/
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── inventory.db
├── models/
│   └── demand_model.pkl
└── src/
    ├── train_model.py
    ├── evaluate_model.py
    ├── inventory_logic.py
    ├── inventory_database.py
    └── forecast_demand.py
```

## How to Run

1. Clone or download the repository.
2. Install the required Python packages.
3. Open a terminal in the project folder.
4. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

To retrain the model or regenerate inventory recommendations and forecasts, run the relevant scripts from the `src` folder as configured in the project.

## Important Notes
Current stock levels and lead times are simulated assumptions, not live inventory data. Forecast results are model-generated estimates and should be validated before operational use.

## Future Improvements
- Improve forecasting through time-based backtesting.
- Compare the model against a simple baseline.
- Refine inventory risk and reorder recommendations.
- Add real inventory and supplier lead-time data.

## Author
Created as a machine learning and data analytics portfolio project.
