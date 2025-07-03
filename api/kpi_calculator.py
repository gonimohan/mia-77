import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
# Assuming direct Supabase client usage or through a refined database.py
from supabase import create_client, Client
import os
import json

# Initialize Supabase client (similar to how it's done in main.py or agent_logic.py)
# This should ideally use a shared Supabase client initialization logic
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as e:
        logging.error(f"KPI Calculator: Failed to initialize Supabase client: {e}")
        supabase_client = None


logger = logging.getLogger(__name__)

# Placeholder for agent_logic.load_state if needed, or we pass the state data directly
# from agent_logic import load_state
# For now, assume agent state data (like num_trends) might be passed if not querying SQLite directly here

def get_total_analyses_run() -> Optional[int]:
    """Queries the Supabase 'reports' table for the count of completed analyses."""
    if not supabase_client:
        logger.error("KPI Calculator: Supabase client not initialized.")
        return None
    try:
        # This table name 'reports' is based on 01_create_tables.sql.
        # If agent completion is tracked differently (e.g. in agent's SQLite 'states' table),
        # this logic would need to adapt or call a function that queries that.
        # For now, assuming a 'reports' table with a 'status' column.
        response = supabase_client.table("reports").select("id", count="exact").eq("status", "completed").execute()
        if response.count is not None:
            logger.info(f"KPI: Total analyses run = {response.count}")
            return response.count
        logger.warning("KPI: Could not retrieve count of completed analyses.")
        return 0 # Default to 0 if count is None but no error
    except Exception as e:
        logger.error(f"KPI Error: Failed to get total analyses run: {e}")
        return None

def get_total_documents_processed() -> Optional[int]:
    """Queries the Supabase 'documents' table for the count of analyzed documents."""
    if not supabase_client:
        logger.error("KPI Calculator: Supabase client not initialized.")
        return None
    try:
        # Assumes 'documents' table and 'status' column from previous analysis of main.py
        response = supabase_client.table("documents").select("id", count="exact").eq("status", "analyzed").execute()
        if response.count is not None:
            logger.info(f"KPI: Total documents processed = {response.count}")
            return response.count
        logger.warning("KPI: Could not retrieve count of processed documents.")
        return 0 # Default to 0 if count is None but no error
    except Exception as e:
        logger.error(f"KPI Error: Failed to get total documents processed: {e}")
        return None

def extract_kpis_from_analysis_state(state_data_json: str) -> Dict[str, Optional[int]]:
    """
    Extracts KPI-relevant information (num_trends, num_opportunities)
    from a single MarketIntelligenceState JSON string.

    Args:
        state_data_json: The JSON string representation of MarketIntelligenceState.
                         This would typically be loaded from the agent's SQLite DB.

    Returns:
        A dictionary with extracted KPI values.
    """
    kpis = {
        "num_trends": None,
        "num_opportunities": None,
    }
    try:
        state_dict = json.loads(state_data_json)
        # These keys 'num_trends', 'num_opportunities' were added to MarketIntelligenceState
        kpis["num_trends"] = state_dict.get("num_trends")
        kpis["num_opportunities"] = state_dict.get("num_opportunities")

        # Fallback if num_ fields are not present but lists are
        if kpis["num_trends"] is None and state_dict.get("market_trends") is not None:
             kpis["num_trends"] = len(state_dict.get("market_trends", []))
        if kpis["num_opportunities"] is None and state_dict.get("opportunities") is not None:
            kpis["num_opportunities"] = len(state_dict.get("opportunities", []))

        logger.info(f"KPI Extraction from state: Trends={kpis['num_trends']}, Opportunities={kpis['num_opportunities']}")
    except json.JSONDecodeError:
        logger.error("KPI Extraction: Failed to parse state_data_json.")
    except Exception as e:
        logger.error(f"KPI Extraction: Error processing state data: {e}")
    return kpis

# Example of how this might be used by the scheduled task (Sub-step 1.3)
# def update_average_kpis_from_recent_analyses():
#     # 1. Fetch recent completed analysis states (e.g., from agent's SQLite DB or a Supabase table that stores state_id)
#     #    For each state_id, load the state_data_json.
#     # 2. For each state:
#     #    extracted_values = extract_kpis_from_analysis_state(state_data_json)
#     #    Store these (e.g., num_trends for this specific analysis) in a temporary list or intermediate table.
#     # 3. Calculate averages (e.g., average num_trends over all processed analyses).
#     # 4. Prepare data for kpi_metrics table and call a function to store it (part of Sub-step 1.3).
#     pass

if __name__ == '__main__':
    # Basic test calls
    # Ensure .env is loaded if running this directly for testing
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env') # Assuming .env is in the root of the project

    # Re-initialize supabase_client if it wasn't set due to env vars not loaded initially
    if not supabase_client:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
            try:
                supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
                logger.info("KPI Calculator (main): Supabase client re-initialized for testing.")
            except Exception as e:
                logger.error(f"KPI Calculator (main): Failed to re-initialize Supabase client for testing: {e}")

    if supabase_client:
        print(f"Total Analyses Run: {get_total_analyses_run()}")
        print(f"Total Documents Processed: {get_total_documents_processed()}")
    else:
        print("Supabase client not available. Skipping live KPI queries in main.")

    # Example test for extract_kpis_from_analysis_state
    mock_state_json_valid = """
    {
        "market_domain": "Test Domain",
        "query": "Test Query",
        "market_trends": [{"name": "Trend 1"}, {"name": "Trend 2"}],
        "opportunities": [{"name": "Opp 1"}],
        "num_trends": 2,
        "num_opportunities": 1
    }
    """
    mock_state_json_no_nums = """
    {
        "market_domain": "Test Domain",
        "query": "Test Query",
        "market_trends": [{"name": "Trend 1"}, {"name": "Trend 2"}, {"name": "Trend 3"}],
        "opportunities": []
    }
    """
    print(f"Extracted from valid mock state: {extract_kpis_from_analysis_state(mock_state_json_valid)}")
    print(f"Extracted from mock state without num_ fields: {extract_kpis_from_analysis_state(mock_state_json_no_nums)}")

# Add Path import for dotenv
from pathlib import Path
