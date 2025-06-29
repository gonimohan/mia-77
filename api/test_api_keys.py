import os
import requests
import logging
from typing import Dict, bool

logger = logging.getLogger(__name__)


def test_news_api() -> bool:
    """Test News API connection"""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return False

    try:
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": api_key},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"News API test failed: {e}")
        return False


def test_mediastack_api() -> bool:
    """Test MediaStack API connection"""
    api_key = os.getenv("MEDIASTACK_API_KEY")
    if not api_key:
        return False

    try:
        response = requests.get(
            "http://api.mediastack.com/v1/news", params={"access_key": api_key, "limit": 1}, timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"MediaStack API test failed: {e}")
        return False


def test_tavily_api() -> bool:
    """Test Tavily API connection"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return False

    try:
        response = requests.post(
            "https://api.tavily.com/search", json={"api_key": api_key, "query": "test", "max_results": 1}, timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Tavily API test failed: {e}")
        return False


def test_serpapi() -> bool:
    """Test SerpAPI connection"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return False

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={"q": "test", "api_key": api_key, "engine": "google", "num": 1},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"SerpAPI test failed: {e}")
        return False


def test_alpha_vantage() -> bool:
    """Test Alpha Vantage API connection"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return False

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": api_key},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Alpha Vantage API test failed: {e}")
        return False


def test_google_gemini() -> bool:
    """Test Google Gemini API connection"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return False

    try:
        # Simple test to check if the API key is valid
        response = requests.get(f"https://generativelanguage.googleapis.com/v1/models?key={api_key}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Google Gemini API test failed: {e}")
        return False


def test_supabase() -> bool:
    """Test Supabase connection"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not key:
        return False

    try:
        response = requests.get(
            f"{url}/rest/v1/", headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=10
        )
        return response.status_code in [200, 404]  # 404 is OK, means connection works
    except Exception as e:
        logger.error(f"Supabase test failed: {e}")
        return False


def test_all_apis() -> Dict[str, bool]:
    """Test all API connections and return results"""
    tests = {
        "News API": test_news_api,
        "MediaStack": test_mediastack_api,
        "Tavily": test_tavily_api,
        "SerpAPI": test_serpapi,
        "Alpha Vantage": test_alpha_vantage,
        "Google Gemini": test_google_gemini,
        "Supabase": test_supabase,
    }

    results = {}
    for name, test_func in tests.items():
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Error testing {name}: {e}")
            results[name] = False

    return results


if __name__ == "__main__":
    results = test_all_apis()
    print("\n=== API Connection Test Results ===")
    for api, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {api}: {'Connected' if status else 'Failed'}")

    working_count = sum(1 for status in results.values() if status)
    total_count = len(results)
    print(f"\nSummary: {working_count}/{total_count} APIs working")
