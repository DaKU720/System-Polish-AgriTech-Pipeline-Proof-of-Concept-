import os
import datetime
import tiktoken
import time
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai
from . import config

def estimate_cost(prompt: str, model="gpt-4o") -> float:
    """
    Calculates the approximate cost of the API call using tiktoken.
    Cost for gpt-4o (Input): ~$5.00 per 1M tokens -> 0.000005 per token.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(prompt))
        cost = num_tokens * 0.000005
        print(f"[FinOps] Prompt Token Count: {num_tokens}")
        print(f"[FinOps] Estimated API Cost (Input): ${cost:.6f}")
        return cost
    except Exception as e:
        print(f"[FinOps] Warning: Could not estimate token count: {e}")
        return 0.0

def load_knowledge_base() -> str:
    """Loads knowledge bases with a character limit to avoid TPM API Errors."""
    rhd_path = f"{config.KNOWLEDGE_DIR}/{config.KNOWLEDGE_FILE}" # Polish-Farm-Guide.txt
    tax_path = f"{config.KNOWLEDGE_DIR}/Polish-Farm-Taxes.txt"
    regional_path = f"{config.KNOWLEDGE_DIR}/Polish-Regional-Farming.txt"
    inventory_path = "farm_inventory_config.json"
    
    kb = ""
    file_limits = {
        rhd_path: 30000, 
        tax_path: 30000, 
        regional_path: 30000,
        inventory_path: 10000
    }
    
    for path, limit in file_limits.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read(limit)
                kb += f"\n--- {os.path.basename(path)} ---\n"
                kb += content
                if len(content) == limit:
                    kb += "\n[Content truncated due to limits...]\n"
    return kb

def generate_report(session_id: str, ml_results: dict, report_type: str = "all"):
    """
    Calls the OpenAI API to generate report(s) based on CLI arguments.
    """
    print(f"Preparing data for AI Reporter (gpt-4o) - Target: {report_type}...")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("[Error] OPENAI_API_KEY not found in environment. Skipping AI report generation.")
        return
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = load_knowledge_base()
    
    today_str = datetime.datetime.now().strftime('%Y-%m-%d')
    current_month = datetime.datetime.now().strftime('%B')
    
    # --- Farm Inventory Context (live state from simulation + config fallback) ---
    farm_cfg   = ml_results.get('farm_config', {})
    farm_state = ml_results.get('farm_state', {})

    cattle_breed        = farm_state.get('cattle_breed') or farm_cfg.get('livestock', {}).get('cattle_breed', 'Highlander')
    cattle_count_start  = farm_state.get('cattle_start') or farm_cfg.get('livestock', {}).get('cattle_count', 300)
    cattle_count_now    = farm_state.get('cattle_current', cattle_count_start)  # ACTUAL post-sim count
    stable_count        = farm_state.get('stable_count') or farm_cfg.get('services', {}).get('horse_stables_count', 20)
    pasture_ha          = farm_state.get('pasture_ha') or farm_cfg.get('land', {}).get('pasture_hectares', 150)
    arable_ha           = farm_state.get('arable_ha') or farm_cfg.get('land', {}).get('arable_hectares', 50)
    silos_now           = farm_state.get('silos_final', {})
    silos_str           = ", ".join([f"{k.split('(')[0].strip()}: {v:.1f}" for k, v in silos_now.items()]) if silos_now else "N/A"

    # Product revenue as readable string
    prod_rev_str = "\n".join([f"    - {p}: {v:,.0f} PLN" for p, v in ml_results.get('product_revenue_breakdown', {}).items()])


    base_data = f"""
    === FARM BUSINESS INTELLIGENCE REPORT DATA ===
    Report Generated: {today_str}

    [FARM INVENTORY SNAPSHOT - LIVE (post-simulation)]
    Cattle NOW: {cattle_count_now} {cattle_breed} cows (started at {cattle_count_start}, net change: {cattle_count_now - cattle_count_start:+d} heads)
    Cattle are grass-fed on {pasture_ha} ha of permanent pasture
    Horse Stables: {stable_count} boxes available for rent
    Arable Land: {arable_ha} ha (Rapeseed, Buckwheat, Potatoes)
    Silo contents right now: {silos_str}
    Total cattle units sold to date: {ml_results.get('total_cattle_sold_units', 'N/A')}

    [FINANCIAL OVERVIEW - Period: {ml_results['date_min']} to {ml_results['date_max']}]
    Total Gross Revenue: {ml_results['total_revenue']:,.2f} PLN
    Before-Tax Profit (after startup capital deducted): {ml_results['total_profit']:,.2f} PLN
    Revenue via RHD (direct retail, tax-advantaged): {ml_results.get('rhd_revenue', 0):,.2f} PLN
    Revenue via Skup (wholesale, fully taxed): {ml_results.get('skup_revenue', 0):,.2f} PLN
    RHD Legal Limit: 100,000 PLN/year
    Profit Trend (last 30 days vs prior 30 days): {ml_results.get('trend_last30_vs_prior30_pct', 0):+.1f}%

    [REVENUE BY PRODUCT]
{prod_rev_str}

    [MACHINE LEARNING CAUSATION MODELS - What drives our business?]
    We trained two separate ML models on daily aggregated time-series data:
    
    MODEL A: CROP INCOME vs LAGGED WEATHER (30-day moving averages)
    - Rain Lags: every +1mm in the 30-day avg rain impacts Daily Income by {ml_results.get('model_income_coefs', {}).get('Rain_30d_Avg', 0):,.0f} PLN
    - Temp Lags: every +1°C in the 30-day avg temp impacts Daily Income by {ml_results.get('model_income_coefs', {}).get('Temp_30d_Avg', 0):,.0f} PLN
    - Corn Futures global tech correlation to our income: {ml_results.get('correlation_corn_income', 'N/A')}
    - Soybean Futures global tech correlation to our income: {ml_results.get('correlation_soybean_income', 'N/A')}
    
    MODEL B: OPERATING EXPENSES vs CURRENCY FLUCTUATION
    - EUR/PLN Exchange Rate: every +1.0 PLN increase in EUR/PLN changes our Daily Expenses by {ml_results.get('model_expense_coefs', {}).get('EUR_PLN', 0):,.0f} PLN
    - (Current Spot EUR/PLN: {ml_results.get('latest_eur_pln', 'N/A')})

    [KNOWLEDGE BASE]
    {knowledge_base}
    """

    if report_type == "all":
        targets = ["Daily", "Weekly", "Monthly", "Quarterly"]
    else:
        targets = [report_type.capitalize()]
    
    for rtype in targets:
        print(f"Generating {rtype} Report...")
        
        system_instruction = (
            f"You are a specific {rtype} Agricultural Business AI for a Polish Farm in Zachodniopomorskie. "
            f"Current Date: {today_str}. Current Month: {current_month}. "
            f"Write a purely {rtype}-focused, actionable SMS report. "
            f"CRITICAL RULES: "
            f"1. You MUST write ONLY in English. Do NOT append a Polish translation. "
            f"2. Do NOT mention data or predictions meant for other timeframes. "
        )

        plain_language_rule = (
            "CRITICAL: Write for a non-farmer business owner / investor. "
            "Use plain business English. NO agricultural jargon. "
            "Replace terms like 'żywiec' with 'beef cattle'. Replace 'RHD' with 'direct retail sales'. "
            "Always explain WHAT to do and WHY, grounded in the ML correlation data above. "
            "Make HARD, SPECIFIC recommendations — not vague suggestions. For example: "
            "'Based on our data, a 1 PLN rise in EUR/PLN costs us X PLN in profit. "
            "Since EUR/PLN is currently at Y, we recommend [specific action].' "
            "Always mention the current cattle herd size and stable occupancy in your overview. "
        )
        system_instruction += plain_language_rule

        if rtype == "Daily":
            system_instruction += (
                "DAILY FOCUS: Operational snapshot for today only. "
                "Cover: (1) Cattle herd status & today's priority tasks. "
                "(2) Horse stable occupancy update. "
                "(3) Any weather-driven action needed based on ML data (e.g. if heavy rain predicted, profit historically drops — act now). "
                "NO financials. NO multi-week forecasting."
            )
        elif rtype == "Weekly":
            system_instruction += (
                "WEEKLY FOCUS: 7-day market risk and opportunity analysis. "
                "MUST cover: (1) How current EUR/PLN level and its ML-proven impact on our profit should drive THIS WEEK's selling decisions. "
                "(2) How US Corn/Soybean futures shifts may affect our Rapeseed and Buckwheat sale prices — be specific about what to hold vs sell from silo. "
                "(3) One concrete RISK and one concrete OPPORTUNITY for the week. "
                "NO overall profit/loss financials."
            )
        elif rtype == "Monthly":
            system_instruction += (
                "MONTHLY FOCUS: Full business performance review. "
                "MUST include in clear sections: "
                "(1) ASSETS: Current cattle count (~{cattle_count_config} {cattle_breed}), stables, land. "
                "(2) REVENUE: Gross total, split between direct retail vs wholesale, and per-product breakdown. "
                "(3) COSTS & TAXES: Operating costs, KRUS payments, estimated income tax (apply Polish agricultural tax rules from knowledge base). "
                "(4) NET PROFIT: Final number after all deductions. Monthly average. "
                "(5) ACTION: One specific financial action for next month based on the ML trends."
            )
        elif rtype == "Quarterly":
            system_instruction += (
                "QUARTERLY FOCUS: Strategic business review for investors/owners. "
                "MUST include: "
                "(1) PORTFOLIO: Asset value summary — cattle herd, land, stables (estimate PLN value). "
                "(2) P&L: Full profit & loss with taxes applied per Polish agricultural law. "
                "(3) ML-DRIVEN STRATEGY: Based on our proven correlations (EUR/PLN impact, temperature, rain, commodity prices), "
                "give 2-3 specific strategic recommendations for the next quarter — e.g. 'Hedge EUR exposure', 'Hold Buckwheat in silo until October'. "
                "(4) RHD TAX OPTIMIZATION: Are we on track to use the full 100,000 PLN direct retail exemption? If not, what to change. "
                "(5) ARiMR SUBSIDY STATUS: Expected subsidy injection and how to account for it."
            )
            
        system_instruction += f"\nYou are restricted to the {rtype} paradigm!"

        estimate_cost(system_instruction + base_data, model="gpt-4o")
        
        @retry(
            wait=wait_exponential(multiplier=2, min=10, max=60), 
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type(openai.RateLimitError)
        )
        def _call_openai(msgs):
            print(f"[FinOps] Sending request to OpenAI API...")
            return client.chat.completions.create(
                model="gpt-4o",
                messages=msgs,
                temperature=0.3
            )
            
        try:
            response = _call_openai([
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": base_data}
            ])
            
            report_content = response.choices[0].message.content
            
            output_path = f"{config.REPORTS_DIR}/{rtype}/{today_str}_report.md"
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(report_content)
                
            print(f"[{rtype}] Report successfully generated and saved to: {output_path}")
            
        except Exception as e:
            print(f"[Error] AI Reporter ({rtype}) failed: {e}")
