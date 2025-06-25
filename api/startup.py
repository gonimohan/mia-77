import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('startup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY',
        'GOOGLE_API_KEY',
        'NEWS_API_KEY',
        'TAVILY_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("All required environment variables are set")
    return True

def initialize_database():
    """Initialize the SQLite database"""
    try:
        from agent_logic import init_db
        init_db()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def test_api_connections():
    """Test connections to external APIs"""
    try:
        from test_api_keys import test_all_apis
        results = test_all_apis()
        
        working_apis = sum(1 for result in results.values() if result)
        total_apis = len(results)
        
        logger.info(f"API Connection Test: {working_apis}/{total_apis} APIs working")
        
        for api, status in results.items():
            if status:
                logger.info(f"✅ {api}: Connected")
            else:
                logger.warning(f"❌ {api}: Failed")
        
        return working_apis > 0
    except Exception as e:
        logger.error(f"Failed to test API connections: {e}")
        return False

def startup_checks():
    """Run all startup checks"""
    logger.info("Starting Market Intelligence Agent API...")
    
    # Check environment
    env_ok = check_environment()
    
    # Initialize database
    db_ok = initialize_database()
    
    # Test API connections
    api_ok = test_api_connections()
    
    if env_ok and db_ok:
        logger.info("✅ Startup checks completed successfully")
        if not api_ok:
            logger.warning("⚠️  Some APIs may not be working, but the service will start")
        return True
    else:
        logger.error("❌ Startup checks failed")
        return False

if __name__ == "__main__":
    startup_checks()
