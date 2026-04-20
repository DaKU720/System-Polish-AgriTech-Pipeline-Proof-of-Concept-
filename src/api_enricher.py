import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
import os
from . import config

CACHE_DIR = "data/api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cached_or_fetch(cache_name: str, fetch_func, *args):
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        print(f"[Cache] Loading {cache_name} from local cache...")
        return pd.read_pickle(cache_path)
    
    # Not cached, fetch
    df = fetch_func(*args)
    df.to_pickle(cache_path)
    return df


def fetch_open_meteo_data(start_date: str, end_date: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetches historical max temp and precipitation from Open-Meteo."""
    print("Fetching Open-Meteo historical weather data...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,precipitation_sum"
        f"&timezone=Europe/Warsaw"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame({
        "Date": data["daily"]["time"],
        "Temp_Max": data["daily"]["temperature_2m_max"],
        "Rain_mm": data["daily"]["precipitation_sum"]
    })
    return df

def fetch_nbp_currency(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical EUR/PLN exchange rate from NBP handling >365 days limits."""
    print("Fetching NBP historical EUR/PLN exchange rate...")
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    all_rates = []
    
    # NBP limits to 367 days per request. Chunk it by 360 days.
    current_start = start_dt
    while current_start <= end_dt:
        current_end = min(current_start + timedelta(days=360), end_dt)
        url = f"http://api.nbp.pl/api/exchangerates/rates/a/eur/{current_start.strftime('%Y-%m-%d')}/{current_end.strftime('%Y-%m-%d')}/?format=json"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            all_rates.extend(data.get("rates", []))
        else:
            print(f"Warning: Failed to fetch NBP data for {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}.")
        
        current_start = current_end + timedelta(days=1)
        
    if not all_rates:
         return pd.DataFrame(columns=["Date", "EUR_PLN"])
         
    df = pd.DataFrame([{
        "Date": rate["effectiveDate"],
        "EUR_PLN": rate["mid"]
    } for rate in all_rates])
    
    return df

def fetch_yfinance_futures(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches Corn and Soybean futures from Yahoo Finance."""
    print("Fetching global futures (Corn & Soybean) from yfinance...")
    # Add 1 day to end_date for yfinance to include the last day
    end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_date_yf = end_date_parsed.strftime("%Y-%m-%d")
    
    # ZC=F is Corn, ZS=F is Soybean
    # Using progress=False to keep logs clean
    data = yf.download(["ZC=F", "ZS=F"], start=start_date, end=end_date_yf, progress=False)
    
    if data.empty:
        return pd.DataFrame(columns=["Date", "Corn_Price_USD", "Soybean_Price_USD"])
    
    # yfinance multi-tick download returns MultiIndex columns. We want the 'Close' prices.
    if 'Close' in data.columns.levels[0]:
        close_prices = data['Close'].reset_index()
    else:
        # Fallback if structure is different
        close_prices = data.reset_index()
    
    # Rename columns safely
    close_prices = close_prices.rename(columns={"index": "Date", "Date": "Date", "ZC=F": "Corn_Price_USD", "ZS=F": "Soybean_Price_USD"})
    close_prices["Date"] = close_prices["Date"].dt.strftime("%Y-%m-%d")
    
    # Keep only needed columns
    cols_to_keep = ["Date"]
    if "Corn_Price_USD" in close_prices.columns: cols_to_keep.append("Corn_Price_USD")
    if "Soybean_Price_USD" in close_prices.columns: cols_to_keep.append("Soybean_Price_USD")
        
    return close_prices[cols_to_keep]

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the sales DataFrame with Weather, Currency, and Commodity data.
    """
    print("Starting API Enrichment process...")
    # Find the range of dates in our mock data
    start_date = df["Date"].min()
    end_date = df["Date"].max()
    
    # 1. Weather
    weather_df = _get_cached_or_fetch(f"weather_{start_date}_{end_date}", fetch_open_meteo_data, start_date, end_date, config.LATITUDE, config.LONGITUDE)
    
    # 2. Currency
    nbp_df = _get_cached_or_fetch(f"nbp_{start_date}_{end_date}", fetch_nbp_currency, start_date, end_date)
    
    # 3. Commodities
    yf_df = _get_cached_or_fetch(f"yfinance_{start_date}_{end_date}", fetch_yfinance_futures, start_date, end_date)
    
    # Create a base calendar to handle missing weekend dates from stock/bank
    calendar_df = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date).strftime("%Y-%m-%d")})
    
    # Merge on Calendar
    merged_calendar = pd.merge(calendar_df, weather_df, on="Date", how="left")
    merged_calendar = pd.merge(merged_calendar, nbp_df, on="Date", how="left")
    merged_calendar = pd.merge(merged_calendar, yf_df, on="Date", how="left")
    
    # Forward fill to cover weekends for NBP and Stock market
    merged_calendar.ffill(inplace=True)
    # Backward fill to cover the start if it begins on a weekend
    merged_calendar.bfill(inplace=True)
    
    print("Merging enriched features with sales data...")
    # Finally merge with the main Sales DataFrame
    enriched_df = pd.merge(df, merged_calendar, on="Date", how="left")
    
    return enriched_df
