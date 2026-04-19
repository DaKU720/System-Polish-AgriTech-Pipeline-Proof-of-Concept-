import os
import tiktoken
from openai import OpenAI
from . import config

def estimate_cost(prompt: str, model="gpt-4o-mini") -> float:
    """
    Calculates the approximate cost of the API call using tiktoken.
    Cost for gpt-4o-mini (as of mid-2024): ~$0.15 per 1M input tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(prompt))
        cost = (num_tokens / 1_000_000) * 0.150
        print(f"[FinOps] Prompt Token Count: {num_tokens}")
        print(f"[FinOps] Estimated API Cost (Input): ${cost:.6f}")
        return cost
    except Exception as e:
        print(f"[FinOps] Warning: Could not estimate token count: {e}")
        return 0.0

def load_knowledge_base() -> str:
    """Loads the RHD guidelines (Polish-Farm-Guide)."""
    kb_path = f"{config.KNOWLEDGE_DIR}/{config.KNOWLEDGE_FILE}"
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            return f.read()
    print("[Warning] Knowledge base not found. Using empty context.")
    return ""

def generate_report(session_id: str, ml_results: dict):
    """
    Calls the OpenAI API to generate a final Farmer SMS Report in Markdown.
    Includes English and a Polish translation at the bottom.
    """
    print("Preparing data for AI Reporter...")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("[Error] OPENAI_API_KEY not found in environment. Skipping AI report generation.")
        return
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = load_knowledge_base()
    
    system_prompt = (
        "You are a Senior Python Data Architect and Agricultural AI Advisor for a Polish Farm. "
        "Your task is to craft a highly concise, data-driven report specifically formatted as an 'SMS to the Farmer'. "
        "The report MUST be written in English, with a Polish translation appended at the bottom in the same Markdown file. "
        "Focus on hard numbers. Include Total Profit, Weather Impact, EUR/PLN Impact, and 7-day, 1-month, 1-quarter predictions based on the ML trends."
    )
    
    user_prompt = f"""
    Farm ML Analytics Report
    ------------------------
    Total Profit: {ml_results['total_profit']:.2f} PLN
    Total Revenue: {ml_results['total_revenue']:.2f} PLN
    Period: {ml_results['date_min']} to {ml_results['date_max']}
    
    Impact Coefficients (Linear Regression via Scikit-Learn):
    - 1 PLN increase in EUR_PLN: {ml_results['coefficients'].get('EUR_PLN', 0):.2f} PLN impact
    - 1°C increase in Temp_Max: {ml_results['coefficients'].get('Temp_Max', 0):.2f} PLN impact
    - 1mm of Rain: {ml_results['coefficients'].get('Rain_mm', 0):.2f} PLN impact
    
    Commodity Insights:
    - Corn Futures Correlation to Profit: {ml_results.get('correlation_corn_profit', 'N/A')}
    
    Background Knowledge (RHD Tax Rules):
    {knowledge_base}
    
    Generate the SMS Markdown Report in English, followed immediately by the Polish translation section. Do not include boilerplate text, just the concise SMS content.
    """
    
    # FinOps Module
    estimate_cost(system_prompt + user_prompt)
    
    print("Calling OpenAI API (gpt-4o-mini)...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        report_content = response.choices[0].message.content
        
        output_path = f"{config.REPORTS_DIR}/{config.REPORT_TEMPLATE.format(session_id)}"
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(report_content)
            
        print(f"Report successfully generated and saved to: {output_path}")
        
    except Exception as e:
        print(f"[Error] AI Reporter failed: {e}")
