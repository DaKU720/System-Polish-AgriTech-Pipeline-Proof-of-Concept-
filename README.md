# AgriTech FinOps Intelligence Pipeline (Proof of Concept)

## Executive Summary

The AgriTech FinOps Intelligence Pipeline is an advanced, stateful simulation and Business Intelligence engine. It was developed to bridge the gap between raw agricultural operational data and financial engineering. 

**The Problem:** Traditional farms generate massive amounts of unstructured, highly seasonal data. This data is heavily impacted by chaotic exogenous variables (weather patterns, global commodity futures, and local currency exchange rates) as well as complex local tax regulations. Standard ERP solutions fail to extract causal relationships between these variables, leaving estate managers and investors unable to accurately project risks or hedge operational costs.

**The Solution:** This Proof of Concept (PoC) introduces a full-stack, data-driven pipeline that not only simulates an enterprise-scale agricultural entity but also integrates a Machine Learning Causation Engine. The system aggregates granular transactions into time-series data, isolates the structural impact of macroeconomic factors, and utilizes a Large Language Model (LLM) to translate these complex mathematical correlations into highly actionable, plain-English strategic advisories.

**Target Audience:** Enterprise agricultural boards, Venture Capital (VC) investors evaluating AgTech portfolios, and technical engineering teams reviewing system architecture for scalability into a SaaS or large-scale data lake infrastructure.

---

## Technical Architecture & Pipeline Execution

The system operates across a strict multi-phase architectural pipeline designed to maintain data integrity and causal validity.

### Phase 1: Stateful Data Mock Engine
Unlike standard random-seed generators, this module behaves as an event-driven farm simulator. It maintains operational states dynamically (e.g., tracking silo capacities, herd reproduction latency, and stable occupancies) over a 730-day timeframe. It fetches real-world baseline pricing via the Polish Central Statistical Office (GUS BDL API) to ensure financial accuracy. 

### Phase 2: Data Validator & ETL Cleaner
The Extraction, Transformation, and Loading (ETL) layer enforces strict mathematical and legal compliance. It autonomously patches missing data pools (using medians) and runs conditional tax audits on the dataset. For instance, it actively monitors and re-tags transactions to comply with specific local regulations (e.g., Polish RHD - retail agricultural trade limitations).

### Phase 3: Macroeconomic API Enricher
The integration layer injects global context into the localized farm data. It merges historical transaction rows with relevant external signals, including localized weather conditions (via Open-Meteo), official FX Spot rates (EUR/PLN), and global commodity futures (US Corn/Soybean prices via yfinance).

### Phase 4: Machine Learning Causation Engine
The analytical core of the system. To avoid false correlations typical of poorly structured data science systems (e.g., linking random rainfall to rigid tax payments), the engine performs Time-Series Daily Aggregation. It splits computations into two independent linear regression layers:
1. **Model A (Agronomic Impact):** Correlates lagged, 30-day moving averages of weather metrics against daily crop income.
2. **Model B (Operational Expense Rate):** Correlates real-time currency devaluation (EUR/PLN flux) against daily recurring expenses (fuel, feed).

### Phase 5: Generative AI Reporter
An integration with OpenAI's GPT-4o model. The LLM does not generate analytical logic on its own; instead, it is fed the strict causal matrices calculated by the ML Engine and the raw limits from the `knowledge` base. It outputs specialized managerial reports across varying temporal scopes (Daily, Weekly, Monthly, Quarterly), advising on advanced financial decisions such as currency hedging or holding silo assets.

---

## Project Structure & Repositories

The project codebase emphasizes modularity and strict separation of concerns.

*   **`src/`**: Contains the core logic scripts mapped exactly to the pipeline phases (`data_generator.py`, `data_cleaner.py`, `api_enricher.py`, `ml_engine.py`, `reporter.py`, `visualizer.py`).
*   **`data/`**: The state management directory. It stores the live `farm_state.json` tracker and the sanitized output of the ETL layer (`cleaned_sales_data.csv`).
*   **`generated-data/`**: The archive for the raw, pre-ETL transaction database (generating up to 10,000 dense rows per simulation).
*   **`knowledge/`**: The static context base. Contains domain-specific text files detailing localized tax legislation and market caps. This folder serves as a RAG (Retrieval-Augmented Generation) foundation, providing the LLM with localized rules without hardcoding them into the Python logic.
*   **`reports/`**: The output directory for the AI Agent. It is subdivided into scope directories (`Daily/`, `Quarterly/`, etc.) and includes a `charts/` sub-directory holding Matplotlib-generated visual intelligence.
*   **`farm_inventory_config.json`**: The central configuration matrix defining the spatial and biological specifications of the simulated environment (herd sizes, land vectors).

---

## Execution & Usage Instructions

To launch the PoC pipeline, follow standard Python execution protocols.

**Prerequisites:** Python 3.9+, API keys securely stored in a `.env` file (e.g., `OPENAI_API_KEY`).

**1. Setup Environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Generate and Clean Data (Phases 1-4.5)**
Execute the simulation, generate the raw dataset, execute the ETL processes, train the ML matrices, and output data visualization graphs. AI reporting is bypassed in this specific command.
```bash
python main.py --generate-data
```

**3. Generate AI Intelligence Reports (Phase 5)**
Run the Generative AI engine on the existing processed data. You can specify a timeframe or command the engine to generate all analytical scopes simultaneously.
```bash
python main.py --report daily
python main.py --report quarterly
python main.py --report all
```

**4. Execute Full Pipeline (End-to-End)**
Run the entire architecture sequentially via a single command line interface entry:
```bash
python main.py --generate-data --report all
```