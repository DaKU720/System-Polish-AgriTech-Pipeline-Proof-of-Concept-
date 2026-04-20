import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
from . import config
from .gus_fetcher import get_prices
import os


def load_farm_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'farm_inventory_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _price_band(base: float, spread: float = 0.08) -> float:
    """Return base price ± spread % random variance (market noise)."""
    return round(base * random.uniform(1 - spread, 1 + spread), 4)


def generate_mock_data(session_id: str):
    """
    Generates a dense, stateful, chronological transaction log (~10 000 rows)
    for a Zachodniopomorskie farm using:
      1. GUS BDL official prices (live or 2024 baseline)
      2. farm_inventory_config.json for herd / land parameters
      3. Realistic seasonal patterns from GUS Szczecin 2024 report
    """
    farm_config = load_farm_config()
    gus = get_prices()

    print(f"Loaded Farm Config: {farm_config['farm_name']}")

    sim_days  = farm_config["simulation_days"]            # 730
    region    = farm_config["location"]
    cattle_n  = farm_config["livestock"]["cattle_count"]  # 300
    breed     = farm_config["livestock"]["cattle_breed"]  # Highlander
    silo_cap  = farm_config["land"].get("silo_capacity_tons", 250)
    arable_ha = farm_config["land"]["arable_hectares"]    # 50 ha

    # ── Land split (GUS-derived allocation) ─────────────────────────────────
    RZEPAK_HA    = int(arable_ha * 0.40)   # 20 ha
    GRYKA_HA     = int(arable_ha * 0.30)   # 15 ha
    ZIEMNIAKI_HA = arable_ha - RZEPAK_HA - GRYKA_HA   # 15 ha
    PASTURE_HA   = farm_config["land"]["pasture_hectares"]  # 150 ha
    STABLE_N     = farm_config["services"]["horse_stables_count"]

    # ── Dynamic state ───────────────────────────────────────────────────────
    current_cattle = cattle_n
    silos = {"Rzepak (Tona)": 0.0, "Gryka (Tona)": 0.0, "Ziemniaki (Decytona)": 0.0}
    stables_occupied = STABLE_N  # updated each month

    # ── Quick price helpers from GUS module ─────────────────────────────────
    P_rzepak_skup    = gus["Rzepak (Tona)"]["skup_pln_per_t"]
    P_rzepak_rhd     = P_rzepak_skup * gus["Rzepak (Tona)"]["retail_premium"]
    C_rzepak         = lambda: _price_band(P_rzepak_skup * gus["Rzepak (Tona)"]["cost_ratio"])

    P_gryka_skup     = gus["Gryka (Tona)"]["skup_pln_per_t"]
    P_gryka_rhd      = P_gryka_skup * gus["Gryka (Tona)"]["retail_premium"]
    C_gryka          = lambda: _price_band(P_gryka_skup * gus["Gryka (Tona)"]["cost_ratio"])

    P_ziem_skup      = gus["Ziemniaki (Decytona)"]["skup_pln_per_t"]      # zł/dt
    P_ziem_rhd       = P_ziem_skup * gus["Ziemniaki (Decytona)"]["retail_premium"]
    C_ziem           = lambda: _price_band(P_ziem_skup * gus["Ziemniaki (Decytona)"]["cost_ratio"])

    SKP_WOL          = gus["Żywiec wołowy (Highlander)"]["skup_pln_per_kg"]
    AVG_KG           = gus["Żywiec wołowy (Highlander)"]["avg_weight_kg"]
    P_wol_skup       = SKP_WOL * AVG_KG                                   # per head
    P_wol_rhd        = SKP_WOL * gus["Żywiec wołowy (Highlander)"]["retail_premium_per_kg"] * AVG_KG
    C_wol            = lambda: _price_band(P_wol_skup * gus["Żywiec wołowy (Highlander)"]["cost_ratio"])

    P_mleko_skup     = gus["Mleko krowie (HL)"]["skup_pln_per_hl"]
    P_mleko_rhd      = P_mleko_skup * gus["Mleko krowie (HL)"]["retail_premium"]
    C_mleko          = lambda: _price_band(P_mleko_skup * gus["Mleko krowie (HL)"]["cost_ratio"])

    P_jaja_skup      = gus["Jaja kurze (100 szt.)"]["skup_pln_per_100"]
    P_jaja_rhd       = P_jaja_skup * gus["Jaja kurze (100 szt.)"]["retail_premium"]
    C_jaja           = lambda: _price_band(P_jaja_skup * gus["Jaja kurze (100 szt.)"]["cost_ratio"])

    P_drob_skup      = gus["Drób rzeźny (KG)"]["skup_pln_per_kg"]
    P_drob_rhd       = P_drob_skup * gus["Drób rzeźny (KG)"]["retail_premium"]
    C_drob           = lambda: _price_band(P_drob_skup * gus["Drób rzeźny (KG)"]["cost_ratio"])

    P_boksy          = gus["Wynajem boksów (konie)"]["monthly_pln"]
    C_boksy          = lambda: _price_band(P_boksy * gus["Wynajem boksów (konie)"]["cost_ratio"])

    FUEL_PRICE       = gus["Paliwo rolnicze (litr)"]["price_pln"]
    FEED_PRICE       = gus["Pasza (kg)"]["price_pln"]
    FERT_PRICE       = gus["Nawóz (kg)"]["price_pln"]

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=sim_days)

    rows = []

    # Buyer type pools by channel
    RHD_BUYERS  = ["Osoba prywatna", "Restauracja lokalna", "Sklep ekologiczny", "Rynek targowy"]
    SKUP_BUYERS = ["Skup rolny", "Przetwórnia", "Hurtownia spożywcza", "Zakład mięsny"]
    B2B_BUYERS  = ["Właściciel konia", "Klub jeździecki", "Stadnina"]

    def buyer_type(channel: str) -> str:
        if channel == "RHD":  return random.choice(RHD_BUYERS)
        if channel == "Skup": return random.choice(SKUP_BUYERS)
        if channel == "B2B":  return random.choice(B2B_BUYERS)
        return "Koszt wewnętrzny"

    def add(date_str, product, qty, channel, unit_price, unit_cost, tx_type="INCOME"):
        rev  = round(qty * unit_price, 2)
        prof = round(rev - qty * unit_cost, 2)
        rows.append([date_str, region, product, qty,
                     round(unit_price, 4), round(unit_cost, 4),
                     rev, prof, buyer_type(channel), channel, tx_type])

    # ── DAY 0: Capital injection / startup investment ────────────────────────
    add(start_date.strftime("%Y-%m-%d"),
        "Inwestycja startowa (stado, ziemia, infrastruktura)",
        1, "N/A", 0, abs(farm_config["initial_capital_pln"]), "EXPENSE")

    print(f"Generating {sim_days}-day simulation (targeting ~10 000 rows)…")

    for offset in range(1, sim_days + 1):
        cur = start_date + timedelta(days=offset)
        ds  = cur.strftime("%Y-%m-%d")
        m, d, wd = cur.month, cur.day, cur.weekday()

        # ════════════════════════════════════════════════════════════════════
        # ANNUAL / SEASONAL EVENTS
        # ════════════════════════════════════════════════════════════════════

        # Rzepak harvest (late July)
        if m == 7 and d == 20:
            qty = round(RZEPAK_HA * 3.14 * random.uniform(0.85, 1.15), 1)
            silos["Rzepak (Tona)"] = min(silo_cap, qty)
            print(f"  [{ds}] Rzepak harvest: {silos['Rzepak (Tona)']:.1f} t")

        # Gryka harvest (mid-September)
        if m == 9 and d == 15:
            qty = round(GRYKA_HA * 1.29 * random.uniform(0.85, 1.15), 1)
            silos["Gryka (Tona)"] = min(silo_cap, qty)
            print(f"  [{ds}] Gryka harvest: {silos['Gryka (Tona)']:.1f} t")

        # Ziemniaki harvest (early October)
        if m == 10 and d == 5:
            qty = round(ZIEMNIAKI_HA * 310 * random.uniform(0.85, 1.15), 0)
            silos["Ziemniaki (Decytona)"] = min(silo_cap * 10, qty)
            print(f"  [{ds}] Ziemniaki harvest: {silos['Ziemniaki (Decytona)']:.0f} dt")

        # Spring fertilisation (April) — bulk purchase
        if m == 4 and d == 1:
            fert_kg = arable_ha * 200
            add(ds, "Zakup nawozów (wiosna)", fert_kg, "N/A", 0, FERT_PRICE, "EXPENSE")

        # Autumn fertilisation (September)
        if m == 9 and d == 1:
            fert_kg = arable_ha * 150
            add(ds, "Zakup nawozów (jesień)", fert_kg, "N/A", 0, FERT_PRICE, "EXPENSE")

        # ARiMR direct payments (15 November each year)
        if d == 15 and m == 11:
            total_ha = PASTURE_HA + arable_ha
            subsidy  = round(total_ha * random.uniform(850, 950), 2)
            add(ds, "ARiMR Dopłaty bezpośrednie (UE)", 1, "N/A", subsidy, 0)

        # Fuel excise refund (1 Feb & 1 Aug)
        if d == 1 and m in [2, 8]:
            refund = round(arable_ha * 110 * gus["Paliwo rolnicze (litr)"]["refund_pln"], 2)
            add(ds, "Zwrot podatku akcyzowego – paliwo", 1, "N/A", refund, 0)

        # ════════════════════════════════════════════════════════════════════
        # QUARTERLY EVENTS
        # ════════════════════════════════════════════════════════════════════
        if d == 1 and m in [1, 4, 7, 10]:
            add(ds, "Składka KRUS (ubezpieczenie rolnicze)", 1, "N/A", 0, 1742, "EXPENSE")

        # ════════════════════════════════════════════════════════════════════
        # MONTHLY EVENTS (1st of month)
        # ════════════════════════════════════════════════════════════════════
        if d == 1:
            # Stable rent - track occupancy
            rented = random.randint(int(STABLE_N * 0.75), STABLE_N)
            stables_occupied = rented   # update state
            add(ds, "Wynajem boksów (konie)", rented, "B2B",
                _price_band(P_boksy), C_boksy())

            # Fixed monthly maintenance (energy, repairs, insurance)
            add(ds, "Koszty stałe (energia, serwis, ubezp.)",
                1, "N/A", 0, farm_config["monthly_maintenance_costs_pln"], "EXPENSE")

            # Veterinary visit
            add(ds, "Wizyta weterynaryjna + leki",
                1, "N/A", 0, random.randint(600, 3200), "EXPENSE")

            # Herd reproduction: calving season March–May
            if m in [3, 4, 5]:
                calves = int(current_cattle * random.uniform(0.05, 0.15))
                current_cattle += calves

        # ════════════════════════════════════════════════════════════════════
        # DAILY OPERATING COSTS
        # ════════════════════════════════════════════════════════════════════

        # 1. Daily fuel consumption (tractors, machinery)
        daily_fuel_l = arable_ha * random.uniform(0.8, 2.5)
        add(ds, "Paliwo rolnicze (litr)", round(daily_fuel_l, 1), "N/A",
            0, FUEL_PRICE, "EXPENSE")

        # 2. Daily feed for cattle (supplement to grazing)
        feed_per_cow = random.uniform(1.5, 4.0)   # kg supplement / cow / day
        total_feed   = round(current_cattle * feed_per_cow, 1)
        add(ds, "Pasza dla bydła (kg)", total_feed, "N/A", 0, FEED_PRICE, "EXPENSE")

        # ════════════════════════════════════════════════════════════════════
        # DAILY MILK (every day – batched to dairy pickup / local shop)
        # ════════════════════════════════════════════════════════════════════
        dairy_cows = int(current_cattle * 0.35)
        milk_hl    = round(dairy_cows * random.uniform(0.10, 0.18), 2)  # HL/day
        channel    = "RHD" if milk_hl < 5 else "Skup"
        add(ds, "Mleko krowie (HL)", milk_hl, channel,
            _price_band(P_mleko_skup if channel == "Skup" else P_mleko_rhd),
            C_mleko())

        # ════════════════════════════════════════════════════════════════════
        # DAILY EGG COLLECTION (every day)
        # ════════════════════════════════════════════════════════════════════
        egg_batches = random.randint(1, 6)   # units of 100 eggs
        channel     = "RHD" if egg_batches <= 2 else "Skup"
        add(ds, "Jaja kurze (100 szt.)", egg_batches, channel,
            _price_band(P_jaja_skup if channel == "Skup" else P_jaja_rhd),
            C_jaja())

        # ════════════════════════════════════════════════════════════════════
        # POULTRY SALES (3×/week, April–October)
        # ════════════════════════════════════════════════════════════════════
        if wd in [1, 3, 5] and 4 <= m <= 10:
            drob_kg = round(random.uniform(30, 150), 1)
            channel  = "RHD" if drob_kg < 50 else "Skup"
            add(ds, "Drób rzeźny (KG)", drob_kg, channel,
                _price_band(P_drob_skup if channel == "Skup" else P_drob_rhd),
                C_drob())

        # ════════════════════════════════════════════════════════════════════
        # CROP SALES FROM SILO (probabilistic when stock available)
        # ════════════════════════════════════════════════════════════════════
        for crop, stock in silos.items():
            if stock <= 0:
                continue
            # Higher sell probability closer to next harvest / spring
            sell_prob = 0.45 if m in [1, 2, 3, 4, 5, 6] else 0.25
            if random.random() < sell_prob:
                if "Decytona" in crop:
                    qty = min(stock, round(random.uniform(10, 120), 0))
                    channel = "RHD" if qty <= 25 else "Skup"
                    add(ds, crop, qty, channel,
                        _price_band(P_ziem_skup if channel == "Skup" else P_ziem_rhd),
                        C_ziem())
                elif "Rzepak" in crop:
                    qty = min(stock, round(random.uniform(0.5, 8.0), 1))
                    channel = "RHD" if qty <= 2 else "Skup"
                    add(ds, crop, qty, channel,
                        _price_band(P_rzepak_skup if channel == "Skup" else P_rzepak_rhd),
                        C_rzepak())
                elif "Gryka" in crop:
                    qty = min(stock, round(random.uniform(0.3, 4.0), 1))
                    channel = "RHD" if qty <= 1 else "Skup"
                    add(ds, crop, qty, channel,
                        _price_band(P_gryka_skup if channel == "Skup" else P_gryka_rhd),
                        C_gryka())
                silos[crop] = round(stock - qty, 2)

        # ════════════════════════════════════════════════════════════════════
        # BEEF CATTLE SALES (rare, inventory-protected)
        # ════════════════════════════════════════════════════════════════════
        if random.random() < 0.04:
            min_herd = int(cattle_n * 0.40)
            if current_cattle > min_herd:
                sell_qty = random.randint(1, 5)
                sell_qty = min(sell_qty, current_cattle - min_herd)
                channel  = "RHD" if sell_qty <= 2 else "Skup"
                add(ds, f"Żywiec wołowy ({breed})", sell_qty, channel,
                    _price_band(P_wol_skup if channel == "Skup" else P_wol_rhd),
                    C_wol())
                current_cattle -= sell_qty

    print(f"Final cattle count: {current_cattle} heads")
    print(f"Final silos: { {k: f'{v:.1f}' for k, v in silos.items()} }")
    print(f"Total rows generated (before anomalies): {len(rows)}")

    # Save live farm state so reporter shows ACTUAL inventory, not config defaults
    farm_state = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "simulation_days": sim_days,
        "cattle_current": current_cattle,
        "cattle_start": cattle_n,
        "cattle_breed": breed,
        "silos_final": silos,
        "pasture_ha": PASTURE_HA,
        "arable_ha": arable_ha,
        "stable_count": STABLE_N,
        "stables_occupied_last": stables_occupied,
        "stables_free": STABLE_N - stables_occupied,
    }
    state_path = os.path.join(config.DATA_DIR, "farm_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(farm_state, f, indent=2, ensure_ascii=False)
    print(f"Farm state saved -> {state_path}")

    df = pd.DataFrame(rows, columns=[
        "Date", "Region", "Product", "Quantity",
        "Unit_Price", "Unit_Cost", "Revenue", "Profit",
        "Buyer_Type", "Sales_Channel", "Transaction_Type"
    ])

    # ── INJECT ANOMALIES ─────────────────────────────────────────────────────
    n    = int(len(df) * config.ANOMALY_PERCENTAGE)
    safe = list(range(1, len(df)))

    for i in random.sample(safe, min(n, len(safe))): df.at[i, "Quantity"]      = np.nan
    for i in random.sample(safe, min(n, len(safe))): df.at[i, "Buyer_Type"]    = np.nan
    for i in random.sample(safe, min(n, len(safe))): df.at[i, "Revenue"]      += random.choice([20000, -10000])
    for i in random.sample(safe, min(n // 2, len(safe))): df.at[i, "Sales_Channel"] = np.nan

    out = f"{config.GENERATED_DATA_DIR}/{config.RAW_DATA_TEMPLATE.format(session_id)}"
    df.to_csv(out, index=False)
    print(f"Raw data saved → {out}  ({len(df)} rows)")
    return df
