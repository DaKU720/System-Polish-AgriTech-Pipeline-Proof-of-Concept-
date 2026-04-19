import pandas as pd
import numpy as np
from . import config

def clean_data(session_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw sales DataFrame.
    Fixes age anomalies, fills missing values, and recalculates mathematics.
    """
    print("Starting data validation and cleaning process...")
    
    # 1. Fix Age
    median_age = df.loc[(df["Buyer_Age"] >= 18) & (df["Buyer_Age"] <= 95), "Buyer_Age"].median()
    invalid_age_mask = (df["Buyer_Age"] < 18) | (df["Buyer_Age"] > 95) | (df["Buyer_Age"].isna())
    invalid_age_count = invalid_age_mask.sum()
    df.loc[invalid_age_mask, "Buyer_Age"] = median_age
    print(f"-> Fixed {invalid_age_count} invalid Buyer_Age records with median ({median_age}).")

    # 2. Fix Missing Data (NaN in Quantity)
    missing_qty_count = df["Quantity"].isna().sum()
    # Replace NaN with median per product
    df["Quantity"] = df.groupby("Product")["Quantity"].transform(lambda x: x.fillna(x.median()))
    print(f"-> Filled {missing_qty_count} missing Quantity records with product medians.")

    # 3. Math Logic Enforcement
    # Revenue = Quantity * Unit_Price
    # Profit = Revenue - (Quantity * Unit_Cost)
    
    # Check how many are objectively wrong before fixing
    expected_revenue = df["Quantity"] * df["Unit_Price"]
    wrong_math_count = (abs(df["Revenue"] - expected_revenue) > 0.01).sum()
    
    df["Revenue"] = df["Quantity"] * df["Unit_Price"]
    df["Profit"] = df["Revenue"] - (df["Quantity"] * df["Unit_Cost"])
    print(f"-> Forced mathematical logic on Revenue and Profit. Fixed {wrong_math_count} erroneous entries.")

    # Save cleaned data
    output_path = f"{config.DATA_DIR}/{config.CLEANED_DATA_TEMPLATE.format(session_id)}"
    df.to_csv(output_path, index=False)
    print(f"Cleaned data successfully saved to: {output_path}")

    return df
