#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the current directory to the path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from database_manager import DatabaseManager
except ImportError as e:
    logger.error(f"Error importing DatabaseManager: {str(e)}")
    sys.exit(1)

def test_database_connection():
    """Test the database connection and basic operations"""
    logger.info("Testing database connection...")
    
    # Default to SQLite for testing
    os.environ["DB_TYPE"] = "sqlite"
    
    try:
        # Create database manager
        db_manager = DatabaseManager()
        
        # Test storing a simple risk assessment
        test_data = {
            "asset_symbol": "TEST",
            "liquidity_rate": 0.01,
            "borrow_rate": 0.02,
            "utilization_rate": 0.5,
            "risk_score": 0.3,
            "risk_level": "medium",
            "data_source": "test",
            "metadata": {"test": True}
        }
        
        result = db_manager.store_risk_assessment(test_data)
        
        if result:
            logger.info("Successfully stored test data in database")
        else:
            logger.error("Failed to store test data in database")
        
        # Test retrieving data
        assessments = db_manager.get_latest_risk_assessments(limit=1)
        
        if assessments and len(assessments) > 0:
            logger.info("Successfully retrieved data from database")
            logger.info(f"Retrieved: {assessments[0]}")
        else:
            logger.error("Failed to retrieve data from database")
        
        # Close connection
        db_manager.close()
        logger.info("Database connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_database_connection()
    sys.exit(0 if success else 1)