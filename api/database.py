
import os
from supabase import create_client, Client
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import logging

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

_supabase_client: Optional[Client] = None

def connect_to_supabase():
    """Initializes the Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise ValueError("Missing Supabase configuration. Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
        
        try:
            print(f"Connecting to Supabase at {SUPABASE_URL}")
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            print("Connected to Supabase successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise

def get_supabase_client() -> Client:
    """Returns the Supabase client. Connects if not already connected."""
    if _supabase_client is None:
        connect_to_supabase()
    return _supabase_client

def insert_document(data: Dict[str, Any], table_name: str = "documents") -> Optional[str]:
    """
    Inserts a new document into the specified table.
    Returns the ID of the inserted document or None if failed.
    """
    try:
        client = get_supabase_client()
        result = client.table(table_name).insert(data).execute()
        if result.data:
            return result.data[0].get('id')
        return None
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        return None

def get_document_by_id(document_id: str, table_name: str = "documents") -> Optional[Dict[str, Any]]:
    """
    Retrieves a document by its ID from the specified table.
    Returns the document dict or None if not found.
    """
    try:
        client = get_supabase_client()
        result = client.table(table_name).select("*").eq("id", document_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        return None

def update_document_by_id(document_id: str, updates: Dict[str, Any], table_name: str = "documents") -> bool:
    """
    Updates a document by its ID in the specified table.
    Returns True if update was successful, False otherwise.
    """
    try:
        client = get_supabase_client()
        result = client.table(table_name).update(updates).eq("id", document_id).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        return False

def delete_document_by_id(document_id: str, table_name: str = "documents") -> bool:
    """
    Deletes a document by its ID from the specified table.
    Returns True if deletion was successful, False otherwise.
    """
    try:
        client = get_supabase_client()
        result = client.table(table_name).delete().eq("id", document_id).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return False

def get_documents_by_user_id(user_id: str, table_name: str = "documents") -> List[Dict[str, Any]]:
    """
    Retrieves all documents for a specific user from the specified table.
    Returns a list of document dicts.
    """
    try:
        client = get_supabase_client()
        result = client.table(table_name).select("*").eq("user_id", user_id).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"Error retrieving documents for user: {e}")
        return []

def get_documents_by_status(status: str, table_name: str = "documents") -> List[Dict[str, Any]]:
    """
    Retrieves all documents with a specific status from the specified table.
    Returns a list of document dicts.
    """
    try:
        client = get_supabase_client()
        result = client.table(table_name).select("*").eq("status", status).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"Error retrieving documents by status: {e}")
        return []

# Market Intelligence specific functions
def insert_report(data: Dict[str, Any]) -> Optional[str]:
    """Insert a new market analysis report."""
    return insert_document(data, "reports")

def get_report_by_id(report_id: str) -> Optional[Dict[str, Any]]:
    """Get a market analysis report by ID."""
    return get_document_by_id(report_id, "reports")

def update_report_by_id(report_id: str, updates: Dict[str, Any]) -> bool:
    """Update a market analysis report by ID."""
    return update_document_by_id(report_id, updates, "reports")

def get_reports_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all reports for a specific user."""
    return get_documents_by_user_id(user_id, "reports")

def insert_data_source(data: Dict[str, Any]) -> Optional[str]:
    """Insert a new data source configuration."""
    return insert_document(data, "data_sources")

def get_data_sources_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all data sources for a specific user."""
    return get_documents_by_user_id(user_id, "data_sources")

def insert_kpi_metric(data: Dict[str, Any]) -> Optional[str]:
    """Insert a new KPI metric."""
    return insert_document(data, "kpi_metrics")

def get_kpi_metrics_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all KPI metrics for a specific user."""
    return get_documents_by_user_id(user_id, "kpi_metrics")

def insert_market_trend(data: Dict[str, Any]) -> Optional[str]:
    """Insert a new market trend."""
    return insert_document(data, "market_trends")

def get_market_trends_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all market trends for a specific user."""
    return get_documents_by_user_id(user_id, "market_trends")

def insert_competitor(data: Dict[str, Any]) -> Optional[str]:
    """Insert a new competitor analysis."""
    return insert_document(data, "competitors")

def get_competitors_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Get all competitors for a specific user."""
    return get_documents_by_user_id(user_id, "competitors")

# Example usage and testing
if __name__ == "__main__":
    try:
        connect_to_supabase()
        print("Supabase connection test successful!")
        
        # Example: Test inserting a document
        # test_doc = {
        #     "filename": "test.txt",
        #     "original_filename": "test.txt",
        #     "status": "uploaded",
        #     "upload_time": "2023-01-01T00:00:00Z"
        # }
        # doc_id = insert_document(test_doc)
        # print(f"Inserted document with ID: {doc_id}")
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
