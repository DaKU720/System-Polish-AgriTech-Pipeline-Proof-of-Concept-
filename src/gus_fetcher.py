import requests
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Optional

CACHE_DIR = "data/api_cache"
GUS_CACHE_FILE = os.path.join(CACHE_DIR, "gus_prices.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)

# ─── GUS BDL 2024 Fallback prices (from GUS Szczecin official report) ─────────
# These are real published values for Zachodniopomorskie / Polska 2024.
# Used when no GUS_BDL_API_KEY is set or API is unreachable.
GUS_2024_FALLBACK = {
    "Rzepak (Tona)": {
        "skup_pln_per_t": 1991.0,     # GUS: 199.08 zł/dt × 10
        "retail_premium": 1.20,        # +20% for direct retail
        "cost_ratio": 0.52,            # ~52% of sale price = production cost
        "seasonal_months": [8, 9, 10, 11, 12, 1, 2, 3],  # harvest Jul, sell Aug–Mar
    },
    "Gryka (Tona)": {
        "skup_pln_per_t": 1050.0,     # niche crop, 2024 regional estimate
        "retail_premium": 1.25,
        "cost_ratio": 0.48,
        "seasonal_months": [10, 11, 12, 1, 2, 3, 4],
    },
    "Ziemniaki (Decytona)": {
        "skup_pln_per_t": 93.26,      # GUS: 93.26 zł/dt
        "retail_premium": 2.56,        # targowisko: 238.54 zł/dt → ×2.56
        "cost_ratio": 0.45,
        "seasonal_months": [10, 11, 12, 1, 2, 3, 4, 5],
    },
    "Żywiec wołowy (Highlander)": {
        "skup_pln_per_kg": 11.16,     # GUS: 11.16 zł/kg żywiec wołowy
        "avg_weight_kg": 550,          # Highlander avg live weight
        "retail_premium_per_kg": 1.60, # premium for direct retail
        "cost_ratio": 0.55,
    },
    "Mleko krowie (HL)": {
        "skup_pln_per_hl": 203.54,    # GUS: 203.54 zł/hl
        "retail_premium": 1.25,
        "cost_ratio": 0.42,
    },
    "Jaja kurze (100 szt.)": {
        "skup_pln_per_100": 62.0,     # estimate from regional data (44.4M eggs/year)
        "retail_premium": 1.35,
        "cost_ratio": 0.55,
    },
    "Drób rzeźny (KG)": {
        "skup_pln_per_kg": 5.15,      # GUS: 5.15 zł/kg
        "retail_premium": 1.35,
        "cost_ratio": 0.52,
    },
    "Wynajem boksów (konie)": {
        "monthly_pln": 1300.0,        # market rate Zachodniopomorskie
        "cost_ratio": 0.22,
    },
    # Operating cost inputs
    "Paliwo rolnicze (litr)": {
        "price_pln": 5.80,            # diesel agro 2024 avg PLN/litr
        "refund_pln": 1.035,          # akcyza zwrot: limit × 1.035 PLN/litr
    },
    "Nawóz (kg)": {
        "price_pln": 2.10,            # mocznik/sól potasowa avg 2024
    },
    "Pasza (kg)": {
        "price_pln": 1.45,            # mieszanka treściwa 2024
    },
}


def _load_cache():
    if os.path.exists(GUS_CACHE_FILE):
        with open(GUS_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(data: dict):
    with open(GUS_CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


def _try_gus_bdl_api(api_key: str) -> Optional[dict]:
    """
    Attempt to fetch current agricultural prices from GUS BDL REST API.
    
    GUS BDL API docs: https://bdl.stat.gov.pl/api/v1/swagger-ui/
    Key variable IDs for Zachodniopomorskie (voivodeship code: 32):
      - We query 'data/by-variable/{variableId}' endpoint.
      
    NOTE: GUS BDL variable IDs are stable but their availability depends on
    the data publication calendar (quarterly / annual). If unavailable, 
    we gracefully fall back to GUS_2024_FALLBACK.
    """
    BASE = "https://bdl.stat.gov.pl/api/v1"
    HEADERS = {"X-ClientId": api_key, "Accept": "application/json"}
    UNIT_ID = "321000000000"  # Zachodniopomorskie TERYT code

    # Known BDL variable IDs for agricultural prices
    GUS_VARIABLES = {
        "rzepak_pln_dt":   453413,   # Ceny skupu rzepaku (zł/dt)
        "zyto_pln_dt":     453412,   # Ceny skupu żyta
        "burak_pln_dt":    453415,   # Buraki cukrowe
        "mleko_pln_hl":    453420,   # Mleko krowie (zł/hl)
        "zywiec_wol_pln_kg": 453424, # Żywiec wołowy (zł/kg)
        "zywiec_wie_pln_kg": 453423, # Żywiec wieprzowy (zł/kg)
    }

    live = {}
    for name, var_id in GUS_VARIABLES.items():
        try:
            url = f"{BASE}/data/by-variable/{var_id}?unit-level=2&unit-parent-id={UNIT_ID}&year=2024&format=json"
            r = requests.get(url, headers=HEADERS, timeout=8)
            if r.status_code == 200:
                payload = r.json()
                results = payload.get("results", [])
                if results:
                    val = results[0].get("values", [{}])[-1].get("val")
                    if val is not None:
                        live[name] = float(val)
        except Exception:
            continue

    return live if live else None


def get_prices(use_cache_hours: int = 168) -> dict:
    """
    Returns price dictionary. Priority:
      1. Fresh GUS BDL API data (if GUS_BDL_API_KEY set and API reachable)
      2. Cached GUS BDL data (if fetched within `use_cache_hours`)
      3. GUS 2024 fallback constants (always works, offline-safe)
    """
    api_key = os.environ.get("GUS_BDL_API_KEY", "")

    # Try cache first
    cached = _load_cache()
    if cached:
        age_hours = (datetime.now() - cached["fetched_at"]).total_seconds() / 3600
        if age_hours < use_cache_hours:
            print(f"[GUS] Loaded prices from local cache (age: {age_hours:.0f}h)")
            return cached["prices"]

    # Try live API
    if api_key:
        print("[GUS] Fetching live prices from GUS BDL API...")
        live = _try_gus_bdl_api(api_key)
        if live:
            # Merge live data into fallback structure (update known fields)
            merged = dict(GUS_2024_FALLBACK)
            if "rzepak_pln_dt" in live:
                merged["Rzepak (Tona)"]["skup_pln_per_t"] = live["rzepak_pln_dt"] * 10
            if "mleko_pln_hl" in live:
                merged["Mleko krowie (HL)"]["skup_pln_per_hl"] = live["mleko_pln_hl"]
            if "zywiec_wol_pln_kg" in live:
                merged["Żywiec wołowy (Highlander)"]["skup_pln_per_kg"] = live["zywiec_wol_pln_kg"]
            _save_cache({"prices": merged, "fetched_at": datetime.now(), "source": "GUS_BDL_API"})
            print(f"[GUS] Live API prices saved to cache.")
            return merged
        else:
            print("[GUS] API unavailable or no data returned. Using GUS 2024 fallback.")
    else:
        print("[GUS] No GUS_BDL_API_KEY set. Using GUS 2024 official baseline prices.")

    return GUS_2024_FALLBACK
