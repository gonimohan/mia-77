import logging
import requests
import time
from typing import Dict, Tuple, Any, Optional
from urllib.parse import urlparse

# Potentially import API key retrieval from agent_logic if needed,
# but for testing a user-configured source, the key should be in source.config
# from .agent_logic import get_api_key # Or a shared utility

logger = logging.getLogger(__name__)

# Standard headers to mimic a browser for generic API/website checks
DEFAULT_HEADERS = {
    'User-Agent': 'MarketIntelligenceAgent/1.0 (ConnectionTester; +http://example.com/botinfo)'
}

def test_generic_api(config: Dict[str, Any]) -> Tuple[bool, str, Optional[float]]:
    """
    Tests a generic API endpoint.
    Expects 'endpoint' in config.
    Supports 'apiKey' and 'auth_type' ('bearer', 'header') in config.
    'header_name' for apiKey if auth_type is 'header'.
    """
    endpoint = config.get("endpoint")
    api_key = config.get("apiKey") # or config.get("api_key")
    auth_type = config.get("auth_type", "none").lower()
    custom_headers = config.get("headers", {}) # Parsed JSON headers from config

    if not endpoint or not urlparse(endpoint).scheme in ["http", "https"]:
        return False, "Invalid or missing API endpoint URL.", None

    headers = {**DEFAULT_HEADERS, **custom_headers} # Merge default with custom, custom takes precedence

    if api_key:
        if auth_type == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_type == "header":
            header_name = config.get("header_name", "X-API-Key") # Default header name if not specified
            headers[header_name] = api_key
        # Add other auth types here if necessary (e.g., basic auth, query param)

    start_time = time.time()
    try:
        # For many APIs, a simple GET to a base endpoint or a specific health/status check endpoint is enough.
        # Some APIs might require a specific simple query (e.g., limit 1 result).
        # We might need to make this more configurable per API type.
        # For a generic test, a GET request to the provided endpoint is a start.
        # If the endpoint is for POST, this test might fail.
        # A better approach might be a HEAD request if the server supports it and it's sufficient.

        # Let's try a HEAD request first as it's lighter.
        response = requests.head(endpoint, headers=headers, timeout=10, allow_redirects=True)
        response.raise_for_status() # Will raise HTTPError for bad responses (4xx or 5xx)

        # If HEAD is successful, we can consider it connected.
        # For some APIs, a GET might be necessary to confirm data access.
        # Example: response = requests.get(endpoint, headers=headers, timeout=10, params={"limit": 1})

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return True, f"Successfully connected to {endpoint}. Status: {response.status_code}", response_time_ms
    except requests.exceptions.HTTPError as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        error_message = f"HTTP Error: {e.response.status_code} {e.response.reason} for URL {endpoint}."
        if e.response.text:
             error_message += f" Details: {e.response.text[:200]}" # First 200 chars of response
        logger.warning(f"Generic API test failed for {endpoint}: {error_message}")
        return False, error_message, response_time_ms
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        logger.warning(f"Generic API test failed for {endpoint}: {e}")
        return False, f"Request failed: {str(e)}", response_time_ms
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        logger.error(f"Unexpected error testing generic API {endpoint}: {e}")
        return False, f"An unexpected error occurred: {str(e)}", response_time_ms


def test_tavily_connection(config: Dict[str, Any]) -> Tuple[bool, str, Optional[float]]:
    api_key = config.get("apiKey")
    if not api_key:
        return False, "Missing Tavily API Key in configuration.", None

    start_time = time.time()
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": "test connection", "max_results": 1},
            timeout=10,
            headers=DEFAULT_HEADERS
        )
        response.raise_for_status()
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return True, "Successfully connected to Tavily API.", response_time_ms
    except requests.exceptions.HTTPError as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        msg = f"Tavily API HTTP Error: {e.response.status_code}. Response: {e.response.text[:100]}"
        logger.warning(msg)
        return False, msg, response_time_ms
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        logger.error(f"Tavily connection test error: {e}")
        return False, f"Tavily API connection failed: {str(e)}", response_time_ms

def test_newsapi_connection(config: Dict[str, Any]) -> Tuple[bool, str, Optional[float]]:
    api_key = config.get("apiKey")
    if not api_key:
        return False, "Missing NewsAPI Key in configuration.", None

    start_time = time.time()
    try:
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": api_key},
            timeout=10,
            headers=DEFAULT_HEADERS
        )
        response.raise_for_status()
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return True, "Successfully connected to NewsAPI.", response_time_ms
    except requests.exceptions.HTTPError as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        msg = f"NewsAPI HTTP Error: {e.response.status_code}. Response: {e.response.text[:100]}"
        logger.warning(msg)
        return False, msg, response_time_ms
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        logger.error(f"NewsAPI connection test error: {e}")
        return False, f"NewsAPI connection failed: {str(e)}", response_time_ms

def test_google_gemini_connection(config: Dict[str, Any]) -> Tuple[bool, str, Optional[float]]:
    api_key = config.get("apiKey")
    if not api_key:
        return False, "Missing Google Gemini API Key in configuration.", None

    start_time = time.time()
    try:
        # Using v1beta as v1 model listing might be different or more restrictive
        # For Gemini, a more robust test would be to try a very small generation request.
        # Listing models is a good start for key validation & reachability.
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=10,
            headers=DEFAULT_HEADERS
        )
        response.raise_for_status()
        data = response.json()
        if "models" not in data or not isinstance(data["models"], list): # Check if 'models' is a list
            logger.warning(f"Google Gemini API response format unexpected: 'models' field missing or not a list. Response: {data}")
            # Depending on strictness, this could be a failure. For now, accept if status is OK.
            # return False, "Invalid response format from Gemini API, 'models' field missing or malformed.", None

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        return True, "Successfully connected to Google Gemini API (listed models).", response_time_ms
    except requests.exceptions.HTTPError as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        msg = f"Google Gemini API HTTP Error: {e.response.status_code}. Response: {e.response.text[:100]}"
        logger.warning(msg)
        return False, msg, response_time_ms
    except Exception as e:
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        logger.error(f"Google Gemini connection test error: {e}")
        return False, f"Google Gemini API connection failed: {str(e)}", response_time_ms


# Add other specific testers here: e.g., test_mediastack, test_serpapi, test_alpha_vantage, etc.
# based on api/test_api_keys.py structure but using config dict.

# Dispatcher function
def test_data_source_connection(source_type: str, config: Dict[str, Any]) -> Tuple[bool, str, Optional[float]]:
    """
    Tests a data source connection based on its type.

    Args:
        source_type: The type of the data source (e.g., 'api', 'newsapi', 'tavily').
                     This is the 'type' field from the data_sources table.
        config: The configuration dictionary for the data source, containing api_key, endpoint etc.

    Returns:
        A tuple: (success: bool, message: str, response_time_ms: Optional[float])
    """
    source_type_lower = source_type.lower() if source_type else ""
    endpoint_config = config.get("endpoint", "").lower()
    name_config = config.get("name", "").lower() # Though 'name' in config is less standard

    # More specific checks first
    if source_type_lower == "tavily" or "tavily.com" in endpoint_config:
        logger.info(f"Using Tavily tester for source type: {source_type}")
        return test_tavily_connection(config)
    if source_type_lower == "newsapi" or "newsapi.org" in endpoint_config:
        logger.info(f"Using NewsAPI tester for source type: {source_type}")
        return test_newsapi_connection(config)
    if source_type_lower in ["google_gemini", "gemini", "llm"] and \
       ("googleapis.com" in endpoint_config or "gemini" in name_config or "google" in name_config):
        # If type is "llm", we might need another field in config like "provider: google"
        logger.info(f"Using Google Gemini tester for source type: {source_type}")
        return test_google_gemini_connection(config)

    # Fallback to generic API tester
    # Common types that might use the generic tester
    generic_types = ["api", "generic_api", "custom_api", "website_scrape", "rss", "webhook"]
    if source_type_lower in generic_types:
        logger.info(f"Using generic API tester for source type: {source_type}")
        return test_generic_api(config)

    # Default if no specific or generic handler
    logger.warning(f"No specific connection tester for source type: '{source_type}'. Cannot test.")
    return False, f"No connection tester available for type '{source_type}'.", None
