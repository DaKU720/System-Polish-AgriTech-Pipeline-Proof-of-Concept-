import os
import argparse
from dotenv import load_dotenv

from src import data_generator
from src import data_cleaner
from src import api_enricher
from src import ml_engine
from src import reporter
from src import visualizer
from src import config

def ensure_directories():
    """Ensure all nested report directories exist."""
    dirs = [
        config.GENERATED_DATA_DIR,
        config.DATA_DIR,
        f"{config.REPORTS_DIR}/Daily",
        f"{config.REPORTS_DIR}/Weekly",
        f"{config.REPORTS_DIR}/Monthly",
        f"{config.REPORTS_DIR}/Quarterly",
        f"{config.REPORTS_DIR}/charts",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Polish AgriTech Intelligence Pipeline")
    parser.add_argument("--generate-data", action="store_true", help="Generate mock data, clean it, and run ML logic (No AI Reports).")
    parser.add_argument("--report", type=str, choices=["daily", "weekly", "monthly", "quarterly", "all"], help="Generate specific AI reports based on existing data.")
    return parser.parse_args()

def run_data_pipeline(session_id: str):
    print("\n[PHASE 1] Data Mock Engine (Stateful)")
    raw_df = data_generator.generate_mock_data(session_id)
    
    print("\n[PHASE 2] Data Validator & Cleaner")
    clean_df = data_cleaner.clean_data(session_id, raw_df)
    
    print("\n[PHASE 3] API Enricher (External Signals)")
    enriched_df = api_enricher.enrich_data(clean_df)
    
    print("\n[PHASE 4] ML Engine & Analytics")
    ml_results = ml_engine.run_ml_pipeline(enriched_df)

    print("\n[PHASE 4.5] Chart Generator")
    chart_paths = visualizer.generate_all_charts(enriched_df, ml_results)
    ml_results["chart_paths"] = chart_paths

    return ml_results

def main():
    args = parse_arguments()
    load_dotenv()
    ensure_directories()
    
    print("=" * 60)
    print(" POLISH AGRITECH INTELLIGENCE PIPELINE (Rancho-PLN-EUR-ML)")
    print("=" * 60)
    
    session_id = "1" # Hardcoded for simplicity in CLI operations
    
    # If no arguments provided, show help and exit
    if not args.generate_data and not args.report:
        print("Please provide an argument. Example:")
        print("  python main.py --generate-data")
        print("  python main.py --report daily")
        print("  python main.py --generate-data --report all")
        return

    ml_results = None

    if args.generate_data:
        ml_results = run_data_pipeline(session_id)
        
    if args.report:
        if ml_results is None:
            # If we didn't generate data in this run, we need to load existing data and run ML again,
            # or just load existing ML results. As a shortcut, we'll re-run API & ML on the clean data.
            print("\n[Loading Existing Data for Report Generation]")
            clean_path = f"{config.DATA_DIR}/cleaned_sales_data_{session_id}.csv"
            if not os.path.exists(clean_path):
                print(f"[Error] No cleaned data found at {clean_path}. Please run with --generate-data first.")
                return
            
            import pandas as pd
            clean_df = pd.read_csv(clean_path)
            enriched_df = api_enricher.enrich_data(clean_df)
            ml_results = ml_engine.run_ml_pipeline(enriched_df)
            
        print("\n[PHASE 5] AI Agent & Reporting")
        reporter.generate_report(session_id, ml_results, report_type=args.report)
    
    print("\n" + "=" * 60)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
