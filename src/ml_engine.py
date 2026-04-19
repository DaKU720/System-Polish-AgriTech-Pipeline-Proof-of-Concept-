import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def run_ml_pipeline(df: pd.DataFrame) -> dict:
    """
    Trains a LinearRegression model to understand the impact of external signals on Profit.
    Returns the influence coefficients.
    """
    print("Starting Machine Learning Engine (Linear Regression)...")
    
    # Selecting the features as defined in requirements
    features = ["Rain_mm", "Temp_Max", "EUR_PLN"]
    target = "Profit"
    
    # We must ensure there's no missing data in features and target
    # The API enricher and Cleaner should have handled it, but safety first.
    clean_df = df.dropna(subset=features + [target])
    
    X = clean_df[features]
    y = clean_df[target]
    
    model = LinearRegression()
    model.fit(X, y)
    
    coefficients = dict(zip(features, model.coef_))
    intercept = model.intercept_
    
    print("\n--- ML Engine Results (Impact on Profit) ---")
    for feature, coef in coefficients.items():
        if feature == "EUR_PLN":
            print(f"[{feature}] 1 PLN increase in EUR_PLN changes Profit by: {coef:.2f} PLN")
        elif feature == "Temp_Max":
            print(f"[{feature}] 1°C increase changes Profit by: {coef:.2f} PLN")
        elif feature == "Rain_mm":
            print(f"[{feature}] 1mm of Rain changes Profit by: {coef:.2f} PLN")
    print("--------------------------------------------\n")
            
    # Include an overview payload for the AI Reporter to summarize
    payload = {
        "coefficients": coefficients,
        "intercept": intercept,
        "total_profit": y.sum(),
        "total_revenue": clean_df["Revenue"].sum(),
        "avg_profit_per_transaction": y.mean(),
        "date_min": clean_df["Date"].min(),
        "date_max": clean_df["Date"].max()
    }
    
    # Extra feature insight for the LLM if yf data is available
    if "Corn_Price_USD" in clean_df.columns:
        corr_corn = clean_df["Corn_Price_USD"].corr(y)
        payload["correlation_corn_profit"] = corr_corn
    
    return payload
