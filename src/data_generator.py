import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from . import config

def generate_mock_data(session_id: str):
    """
    Generates mock sales data for the AgriTech pipeline and introduces artificial anomalies.
    Returns the generated DataFrame and saves it to a CSV file.
    """
    print(f"Generating {config.NUMBER_OF_RECORDS} mock records...")

    # Define base categories and constraints
    products = [
        "Highlander Cow (Meat)", 
        "Free Range Eggs (Pack 10)", 
        "Stable Rent (Monthly)", 
        "Live Poultry", 
        "Imported Feed (Ton)", 
        "Apples (Box)", 
        "Potatoes (Sack)", 
        "Wheat (Ton)", 
        "Cow Milk (Liter)"
    ]
    
    regions = [
        "Dolnośląskie", "Kujawsko-Pomorskie", "Lubelskie", "Lubuskie",
        "Łódzkie", "Małopolskie", "Mazowieckie", "Opolskie",
        "Podkarpackie", "Podlaskie", "Pomorskie", "Śląskie",
        "Świętokrzyskie", "Warmińsko-Mazurskie", "Wielkopolskie", "Zachodniopomorskie"
    ]

    base_prices = {
        "Highlander Cow (Meat)": (5000, 8000, 3000, 4500), # (min_price, max_price, min_cost, max_cost)
        "Free Range Eggs (Pack 10)": (12, 20, 5, 8),
        "Stable Rent (Monthly)": (800, 1500, 200, 400),
        "Live Poultry": (25, 45, 10, 20),
        "Imported Feed (Ton)": (1200, 1800, 900, 1400),
        "Apples (Box)": (40, 70, 15, 30),
        "Potatoes (Sack)": (20, 35, 8, 15),
        "Wheat (Ton)": (900, 1300, 600, 900),
        "Cow Milk (Liter)": (3, 5, 1, 2)
    }

    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = []
    
    for _ in range(config.NUMBER_OF_RECORDS):
        rand_date = start_date + timedelta(days=random.randint(0, 365))
        region = random.choice(regions)
        product = random.choice(products)
        buyer_age = random.randint(18, 95)
        
        min_p, max_p, min_c, max_c = base_prices[product]
        unit_price = round(random.uniform(min_p, max_p), 2)
        unit_cost = round(random.uniform(min_c, max_c), 2)
        
        # Adjust quantity based on product type
        if "Cow (Meat)" in product or "Rent" in product or "Feed" in product or "Wheat" in product:
            quantity = random.randint(1, 5)
        elif "Eggs" in product or "Poultry" in product or "Apples" in product or "Potatoes" in product:
            quantity = random.randint(5, 100)
        else: # Milk
            quantity = random.randint(50, 1000)
            
        revenue = round(quantity * unit_price, 2)
        profit = round(revenue - (quantity * unit_cost), 2)
        
        data.append([
            rand_date.strftime("%Y-%m-%d"), 
            region, 
            product, 
            quantity, 
            unit_price, 
            unit_cost, 
            revenue, 
            profit, 
            buyer_age
        ])

    df = pd.DataFrame(data, columns=[
        "Date", "Region", "Product", "Quantity", 
        "Unit_Price", "Unit_Cost", "Revenue", "Profit", "Buyer_Age"
    ])

    # Introduce Anomalies (approx 5% per anomaly type)
    num_anomalies = int(config.NUMBER_OF_RECORDS * config.ANOMALY_PERCENTAGE)
    
    print(f"Introducing {num_anomalies} anomalies of each type for robustness testing...")
    
    # 1. Missing data (NaN) in Quantity
    nan_indices = random.sample(range(config.NUMBER_OF_RECORDS), num_anomalies)
    df.loc[nan_indices, "Quantity"] = np.nan
    
    # 2. Impossible Age (e.g. 150)
    age_indices = random.sample(range(config.NUMBER_OF_RECORDS), num_anomalies)
    df.loc[age_indices, "Buyer_Age"] = 150
    
    # 3. Math Errors in Revenue
    math_indices = random.sample(range(config.NUMBER_OF_RECORDS), num_anomalies)
    # Give them clearly incorrect revenues
    df.loc[math_indices, "Revenue"] = df.loc[math_indices, "Revenue"] + random.choice([10000, -5000])

    # Save to generated-data folder
    output_path = f"{config.GENERATED_DATA_DIR}/{config.RAW_DATA_TEMPLATE.format(session_id)}"
    df.to_csv(output_path, index=False)
    print(f"Raw data successfully saved to: {output_path}")

    return df
