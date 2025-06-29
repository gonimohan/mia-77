"""
Test script to verify Supabase connection and table access
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_supabase_client
from dotenv import load_dotenv

load_dotenv()


def test_supabase_tables():
    """Test if we can connect to Supabase and access the tables"""
    try:
        client = get_supabase_client()
        print("✅ Successfully connected to Supabase!")

        # Test accessing each table
        tables_to_test = ["users", "data_sources", "reports", "kpi_metrics", "market_trends", "competitors"]

        for table in tables_to_test:
            try:
                result = client.table(table).select("*").limit(1).execute()
                print(f"✅ Table '{table}' is accessible")
            except Exception as e:
                print(f"❌ Error accessing table '{table}': {e}")

        return True

    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False


if __name__ == "__main__":
    print("Testing Supabase connection...")
    test_supabase_tables()
