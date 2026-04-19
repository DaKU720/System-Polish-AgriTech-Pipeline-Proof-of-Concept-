import os
from dotenv import load_dotenv

from src import data_generator
from src import data_cleaner
from src import api_enricher
from src import ml_engine
from src import reporter
from src import config

def main():
    # Load Environment Variables (.env)
    load_dotenv()
    
    print("=" * 60)
    print(" POLISH AGRITECH INTELLIGENCE PIPELINE (Rancho-PLN-EUR-ML)")
    print("=" * 60)
    
    # 1. Ask for Session ID / Report ID
    session_id = input("Please enter the Session / Report ID to begin (e.g. 1, 2, 3): ").strip()
    if not session_id:
        print("Session ID cannot be empty. Defaulting to '1'.")
        session_id = "1"
        
    print("\n[PHASE 1] Data Mock Engine")
    raw_df = data_generator.generate_mock_data(session_id)
    
    print("\n[PHASE 2] Data Validator & Cleaner")
    clean_df = data_cleaner.clean_data(session_id, raw_df)
    
    print("\n[PHASE 3] API Enricher (External Signals)")
    enriched_df = api_enricher.enrich_data(clean_df)
    
    print("\n[PHASE 4] ML Engine & Analytics")
    ml_results = ml_engine.run_ml_pipeline(enriched_df)
    
    print("\n[PHASE 5] AI Agent & Reporting")
    reporter.generate_report(session_id, ml_results)
    
    print("\n" + "=" * 60)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
