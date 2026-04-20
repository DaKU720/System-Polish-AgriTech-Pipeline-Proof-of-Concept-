import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from datetime import datetime

REPORTS_DIR = "reports"
CHARTS_DIR  = os.path.join(REPORTS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def _fmt_pln(x, _):
    """Y-axis formatter: show in thousands PLN."""
    return f"{x/1000:.0f}k"


def generate_all_charts(df: pd.DataFrame, ml_results: dict) -> list:
    """
    Generate all analytical charts and return list of saved file paths.
    Called after ML engine, before AI reporter.
    """
    paths = []
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    today = datetime.now().strftime("%Y-%m-%d")

    paths.append(_chart_cashflow(df, today))
    paths.append(_chart_revenue_by_product(df, today))
    paths.append(_chart_channel_split(df, today))
    paths.append(_chart_monthly_income_vs_expense(df, today))
    paths.append(_chart_ml_feature_impact(ml_results, today))

    paths = [p for p in paths if p]  # filter None
    print(f"[Charts] Generated {len(paths)} charts in {CHARTS_DIR}/")
    return paths


# ─────────────────────────────────────────────────────────────────────────────

def _chart_cashflow(df: pd.DataFrame, today: str) -> str:
    """Running cash balance over the entire simulation period."""
    try:
        df_sorted = df.sort_values("Date").copy()
        df_sorted["daily_cash"] = df_sorted["Profit"]
        df_daily = df_sorted.groupby("Date")["daily_cash"].sum().cumsum().reset_index()
        df_daily.columns = ["Date", "Balance"]

        fig, ax = plt.subplots(figsize=(12, 5))
        color = "#2ecc71" if df_daily["Balance"].iloc[-1] > 0 else "#e74c3c"

        ax.fill_between(df_daily["Date"], df_daily["Balance"], alpha=0.15, color=color)
        ax.plot(df_daily["Date"], df_daily["Balance"], color=color, linewidth=2)
        ax.axhline(0, color="#7f8c8d", linewidth=0.8, linestyle="--")

        ax.set_title("Cumulative Cash Flow — Rancho Zachodniopomorskie", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance (PLN thousands)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pln))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = os.path.join(CHARTS_DIR, f"{today}_cashflow.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[Charts] cashflow error: {e}")
        return None


def _chart_revenue_by_product(df: pd.DataFrame, today: str) -> str:
    """Horizontal bar chart: gross revenue per product (INCOME only)."""
    try:
        income = df[df["Transaction_Type"] == "INCOME"] if "Transaction_Type" in df.columns else df
        by_prod = income.groupby("Product")["Revenue"].sum().sort_values()
        by_prod = by_prod[by_prod > 0]

        fig, ax = plt.subplots(figsize=(10, max(4, len(by_prod) * 0.55)))
        bars = ax.barh(by_prod.index, by_prod.values, color="#3498db", alpha=0.82, edgecolor="white")

        for bar, val in zip(bars, by_prod.values):
            ax.text(val + by_prod.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val/1000:.0f}k PLN", va="center", fontsize=8.5)

        ax.set_title("Gross Revenue by Product", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Revenue (PLN)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pln))
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, by_prod.max() * 1.18)
        fig.tight_layout()

        path = os.path.join(CHARTS_DIR, f"{today}_revenue_by_product.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[Charts] revenue_by_product error: {e}")
        return None


def _chart_channel_split(df: pd.DataFrame, today: str) -> str:
    """Pie chart: revenue split RHD / Skup / B2B."""
    try:
        income = df[df["Transaction_Type"] == "INCOME"] if "Transaction_Type" in df.columns else df
        by_ch = income.groupby("Sales_Channel")["Revenue"].sum()
        by_ch = by_ch[by_ch > 0].sort_values(ascending=False)

        COLORS = {"RHD": "#27ae60", "Skup": "#2980b9", "B2B": "#8e44ad", "N/A": "#95a5a6"}
        colors = [COLORS.get(c, "#bdc3c7") for c in by_ch.index]

        fig, ax = plt.subplots(figsize=(7, 5))
        wedges, texts, autotexts = ax.pie(
            by_ch.values, labels=by_ch.index, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("Revenue Split by Sales Channel", fontsize=13, fontweight="bold", pad=10)
        fig.tight_layout()

        path = os.path.join(CHARTS_DIR, f"{today}_channel_split.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[Charts] channel_split error: {e}")
        return None


def _chart_monthly_income_vs_expense(df: pd.DataFrame, today: str) -> str:
    """Grouped bar: monthly total income vs total expenses."""
    try:
        df2 = df.copy()
        df2["YearMonth"] = df2["Date"].dt.to_period("M").astype(str)

        if "Transaction_Type" in df2.columns:
            income  = df2[df2["Transaction_Type"] == "INCOME"].groupby("YearMonth")["Revenue"].sum()
            expense = df2[df2["Transaction_Type"] == "EXPENSE"].groupby("YearMonth").apply(
                lambda x: (x["Quantity"].fillna(1) * x["Unit_Cost"].fillna(0)).sum(),
                include_groups=False
            )
        else:
            income  = df2[df2["Revenue"] > 0].groupby("YearMonth")["Revenue"].sum()
            expense = df2[df2["Profit"] < 0].groupby("YearMonth")["Profit"].abs().sum()

        months = sorted(set(income.index) | set(expense.index))
        inc_vals = [income.get(m, 0) for m in months]
        exp_vals = [expense.get(m, 0) for m in months]

        x = np.arange(len(months))
        w = 0.4

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - w/2, inc_vals, width=w, label="Income", color="#27ae60", alpha=0.85)
        ax.bar(x + w/2, exp_vals, width=w, label="Expenses", color="#e74c3c", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7.5)
        ax.set_title("Monthly Income vs Expenses", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("PLN")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pln))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = os.path.join(CHARTS_DIR, f"{today}_monthly_income_vs_expense.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[Charts] monthly_income_vs_expense error: {e}")
        return None


def _chart_ml_feature_impact(ml_results: dict, today: str) -> str:
    """Horizontal bar showing ML-derived impact of external factors on Income/Expense."""
    try:
        coefs = ml_results.get("coefficients", {})
        if not coefs:
            return None

        labels = {
            "Rain_30d_Avg": "Rain (30d MA) +1mm -> Income",
            "Temp_30d_Avg": "Temp (30d MA) +1°C -> Income",
            "EUR_PLN":      "EUR/PLN +1.0 -> Expense",
        }
        names  = [labels.get(k, k) for k in coefs]
        values = list(coefs.values())
        colors = ["#27ae60" if v > 0 else "#e74c3c" for v in values]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="#7f8c8d", linewidth=0.8)

        for bar, val in zip(bars, values):
            sign = "+" if val > 0 else ""
            ax.text(val + (max(values)-min(values)) * 0.02 * (1 if val >= 0 else -1),
                    bar.get_y() + bar.get_height() / 2,
                    f"{sign}{val:,.0f} PLN", va="center", fontsize=9,
                    ha="left" if val >= 0 else "right")

        ax.set_title("ML Causation Models: Feature Impact on Daily PLN", fontsize=12, fontweight="bold")
        ax.set_xlabel("PLN change per daily aggregate")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        path = os.path.join(CHARTS_DIR, f"{today}_ml_feature_impact.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as e:
        print(f"[Charts] ml_feature_impact error: {e}")
        return None
