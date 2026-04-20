import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import os

FARM_STATE_PATH = os.path.join("data", "farm_state.json")


def _load_farm_state() -> dict:
    """Load the live farm state written by data_generator after simulation."""
    if os.path.exists(FARM_STATE_PATH):
        with open(FARM_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_ml_pipeline(df: pd.DataFrame) -> dict:
    """
    Trains Dual Linear Regression models on daily-aggregated data with lagged features.
    Model A: Daily Income vs 30-Day Moving Average Weather
    Model B: Daily Expense vs EUR/PLN
    """
    print("Starting Advanced ML Engine (Dual Regression & Time-Series Aggregation)...")
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # --- 1. Daily Aggregation ---
    # Weather and FX are identical for a given day, so taking the first valid is fine.
    weather_cols = ["Rain_mm", "Temp_Max", "EUR_PLN"]
    if "Corn_Price_USD" in df.columns: weather_cols.append("Corn_Price_USD")
    if "Soybean_Price_USD" in df.columns: weather_cols.append("Soybean_Price_USD")

    # Group external signals daily
    daily_signals = df.groupby("Date")[weather_cols].first()

    # Aggregate financial streams
    income_series = df[df["Transaction_Type"] == "INCOME"].groupby("Date")["Revenue"].sum()
    # Expenses have negative profit, so we take absolute value to represent cost magnitude
    expense_series = df[df["Transaction_Type"] == "EXPENSE"].groupby("Date")["Profit"].sum().abs()

    # Combine into a single daily DataFrame
    daily_df = pd.DataFrame(index=daily_signals.index)
    daily_df = daily_df.join(daily_signals)
    daily_df["Daily_Income"] = income_series
    daily_df["Daily_Expense"] = expense_series
    daily_df.fillna(0, inplace=True)
    
    # --- 2. Feature Engineering (Moving Averages) ---
    daily_df["Rain_30d_Avg"] = daily_df["Rain_mm"].rolling(window=30, min_periods=1).mean()
    daily_df["Temp_30d_Avg"] = daily_df["Temp_Max"].rolling(window=30, min_periods=1).mean()
    daily_df["EUR_PLN"] = daily_df["EUR_PLN"] # We use current day EUR for expenses (spot buying)

    # Clean missing for ML
    ml_df = daily_df.dropna()

    # --- 3. Dual Regression Models ---
    
    # MODEL A: Income vs Lags (Weather)
    features_A = ["Rain_30d_Avg", "Temp_30d_Avg"]
    X_a = ml_df[features_A]
    y_a = ml_df["Daily_Income"]
    model_A = LinearRegression()
    model_A.fit(X_a, y_a)
    coefs_A = dict(zip(features_A, model_A.coef_))

    # MODEL B: Expenses vs FX (Euro)
    features_B = ["EUR_PLN"]
    X_b = ml_df[features_B]
    y_b = ml_df["Daily_Expense"]
    model_B = LinearRegression()
    model_B.fit(X_b, y_b)
    coefs_B = dict(zip(features_B, model_B.coef_))

    print("\n--- ML Engine Results (True Causal Impact) ---")
    print(f"[Model A: Income] +1mm Rain (30d Avg) impacts Daily Income by: {coefs_A['Rain_30d_Avg']:,.2f} PLN")
    print(f"[Model A: Income] +1°C Temp (30d Avg) impacts Daily Income by: {coefs_A['Temp_30d_Avg']:,.2f} PLN")
    print(f"[Model B: Expense] +1 PLN in EUR/PLN impacts Daily Expense by: {coefs_B['EUR_PLN']:,.2f} PLN")
    print("------------------------------------------------\n")

    # --- Analytics & Payload Generation ---
    rhd_revenue = df.loc[df["Sales_Channel"] == "RHD", "Revenue"].sum() if "Sales_Channel" in df.columns else 0.0
    skup_revenue = df.loc[df["Sales_Channel"] == "Skup", "Revenue"].sum() if "Sales_Channel" in df.columns else 0.0
    product_revenue = df[df["Transaction_Type"]=="INCOME"].groupby("Product")["Revenue"].sum().to_dict()
    
    cattle_products = df[df["Product"].str.contains("wołowy|Wołowy", na=False)]
    total_cattle_sold = cattle_products["Quantity"].sum()

    # Revenue trend: last 30 days vs prior 30 days
    max_date = daily_df.index.max()
    last_30 = daily_df[daily_df.index >= max_date - pd.Timedelta(days=30)]["Daily_Income"].sum()
    prior_30 = daily_df[(daily_df.index >= max_date - pd.Timedelta(days=60)) & (daily_df.index < max_date - pd.Timedelta(days=30))]["Daily_Income"].sum()
    trend_pct = ((last_30 - prior_30) / abs(prior_30) * 100) if prior_30 != 0 else 0

    farm_config = {}
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'farm_inventory_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            farm_config = json.load(f)

    farm_state = _load_farm_state()

    payload = {
        "model_income_coefs": coefs_A,
        "model_expense_coefs": coefs_B,
        "total_profit": df["Profit"].sum(),
        "total_revenue": df["Revenue"].sum(),
        "date_min": df["Date"].min().strftime("%Y-%m-%d"),
        "date_max": df["Date"].max().strftime("%Y-%m-%d"),
        "rhd_revenue": rhd_revenue,
        "skup_revenue": skup_revenue,
        "rhd_limit_pln": 100000,
        "product_revenue_breakdown": product_revenue,
        "total_cattle_sold_units": int(total_cattle_sold),
        "trend_last30_vs_prior30_pct": round(trend_pct, 2),
        "farm_config": farm_config,
        "farm_state": farm_state,
        "latest_eur_pln": round(daily_df["EUR_PLN"].iloc[-1], 4) if not daily_df.empty else 0.0
    }
    
    if "Corn_Price_USD" in daily_df.columns:
        payload["correlation_corn_income"] = round(daily_df["Corn_Price_USD"].corr(daily_df["Daily_Income"]), 4)
    if "Soybean_Price_USD" in daily_df.columns:
        payload["correlation_soybean_income"] = round(daily_df["Soybean_Price_USD"].corr(daily_df["Daily_Income"]), 4)

    # Legacy key for visualizer compatibility (it visualizes both now in the same dictionary for simplicity)
    payload["coefficients"] = {**coefs_A, **coefs_B}

    return payload
