import pandas as pd
import numpy as np
from . import config

VALID_BUYER_TYPES = {
    "Osoba prywatna", "Restauracja lokalna", "Sklep ekologiczny", "Rynek targowy",
    "Skup rolny", "Przetwórnia", "Hurtownia spożywcza", "Zakład mięsny",
    "Właściciel konia", "Klub jeździecki", "Stadnina", "Koszt wewnętrzny",
}

VALID_TX_TYPES = {"INCOME", "EXPENSE"}


def clean_data(session_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw sales DataFrame:
      1. Validate & fix Buyer_Type (replaced Buyer_Age — farm context, not retail)
      2. Fill missing Quantity with product medians
      3. Fill & validate Sales_Channel; enforce RHD quantity limits
      4. Validate Transaction_Type (INCOME / EXPENSE)
      5. Enforce Revenue = Qty × Unit_Price math
      6. Add Cashflow_Balance running total column
    """
    print("Starting data validation and cleaning process...")

    # ── 1. Buyer_Type validation (replaces the old Buyer_Age column) ──────────
    if "Buyer_Type" in df.columns:
        missing_buyer = df["Buyer_Type"].isna().sum()
        # Fill missing based on channel
        channel_defaults = {
            "RHD":  "Osoba prywatna",
            "Skup": "Skup rolny",
            "B2B":  "Właściciel konia",
            "N/A":  "Koszt wewnętrzny",
        }
        mask_nan = df["Buyer_Type"].isna()
        for ch, default in channel_defaults.items():
            df.loc[mask_nan & (df["Sales_Channel"] == ch), "Buyer_Type"] = default
        df["Buyer_Type"] = df["Buyer_Type"].fillna("Koszt wewnętrzny")

        # Flag unknown values
        invalid_buyer = ~df["Buyer_Type"].isin(VALID_BUYER_TYPES)
        if invalid_buyer.sum() > 0:
            df.loc[invalid_buyer, "Buyer_Type"] = "Nieznany"
        print(f"-> Fixed {missing_buyer} missing Buyer_Type records (filled by channel context).")
    elif "Buyer_Age" in df.columns:
        # Legacy support: rename column if old data is loaded
        df.rename(columns={"Buyer_Age": "Buyer_Type"}, inplace=True)
        df["Buyer_Type"] = "Nieznany (migracja)"
        print("-> [MIGRATION] Renamed legacy Buyer_Age → Buyer_Type.")

    # ── 2. Fix Missing Quantity (NaN) ─────────────────────────────────────────
    missing_qty = df["Quantity"].isna().sum()
    df["Quantity"] = df.groupby("Product")["Quantity"].transform(
        lambda x: x.fillna(x.median())
    )
    print(f"-> Filled {missing_qty} missing Quantity records with product medians.")

    # ── 3. Sales_Channel: fill NaN + enforce RHD legal limits ─────────────────
    if "Sales_Channel" in df.columns:
        missing_ch = df["Sales_Channel"].isna().sum()
        df["Sales_Channel"] = df["Sales_Channel"].fillna("Skup")
        print(f"-> Filled {missing_ch} missing Sales_Channel tags with default 'Skup'.")

        # RHD law: cattle sold under RHD label must be ≤ 2 head per transaction
        rhd_cattle_mask = (
            (df["Sales_Channel"] == "RHD")
            & (df["Product"].str.contains("Żywiec", na=False))
            & (df["Quantity"] > 2)
        )
        n_reverted = rhd_cattle_mask.sum()
        df.loc[rhd_cattle_mask, "Sales_Channel"] = "Skup"
        if n_reverted:
            print(f"-> [TAX AUDIT] Reverted {n_reverted} RHD cattle transactions to 'Skup' "
                  f"(quantity > 2 head violates RHD Rozporządzenie 2017).")

    # ── 4. Transaction_Type: fill missing ─────────────────────────────────────
    if "Transaction_Type" in df.columns:
        missing_tx = df["Transaction_Type"].isna().sum()
        # Products that are always costs
        expense_keywords = [
            "Paliwo", "Pasza", "KRUS", "Wizyta", "Koszty stałe",
            "Nawóz", "Inwestycja startowa"
        ]
        for kw in expense_keywords:
            mask = df["Product"].str.contains(kw, na=False) & df["Transaction_Type"].isna()
            df.loc[mask, "Transaction_Type"] = "EXPENSE"
        df["Transaction_Type"] = df["Transaction_Type"].fillna("INCOME")
        invalid_tx = ~df["Transaction_Type"].isin(VALID_TX_TYPES)
        df.loc[invalid_tx, "Transaction_Type"] = "INCOME"
        print(f"-> Validated Transaction_Type. Fixed {missing_tx} missing/invalid entries.")
    else:
        # Add column if missing (legacy data)
        df["Transaction_Type"] = df["Revenue"].apply(lambda r: "EXPENSE" if r == 0 else "INCOME")
        print("-> [MIGRATION] Added Transaction_Type column based on Revenue values.")

    # ── 5. Math Logic Enforcement ─────────────────────────────────────────────
    expected_revenue = (df["Quantity"] * df["Unit_Price"]).round(2)
    wrong_math = (abs(df["Revenue"] - expected_revenue) > 0.01).sum()
    df["Revenue"] = expected_revenue
    df["Profit"]  = (df["Revenue"] - df["Quantity"] * df["Unit_Cost"]).round(2)
    print(f"-> Enforced Revenue & Profit math. Fixed {wrong_math} erroneous entries.")

    # ── 6. Cashflow running balance  ──────────────────────────────────────────
    # Sort by date for running total, then restore original order
    df_sorted = df.sort_values("Date").copy()
    df_sorted["Cashflow_Balance"] = df_sorted["Profit"].cumsum().round(2)
    df["Cashflow_Balance"] = df_sorted["Cashflow_Balance"]

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = f"{config.DATA_DIR}/{config.CLEANED_DATA_TEMPLATE.format(session_id)}"
    df.to_csv(output_path, index=False)
    print(f"Cleaned data successfully saved to: {output_path}")
    return df
