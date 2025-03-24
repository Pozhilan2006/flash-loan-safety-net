import os
import logging
from pathlib import Path
from flask import Flask, render_template, send_from_directory
from waitress import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("flash_loan_api.log")
    ]
)
logger = logging.getLogger("flash-loan-api")

# Import modules
try:
    # Import sentiment analysis
    from scripts.sentiment_analysis import SentimentAnalyzer
    
    # Import anomaly detection
    from scripts.anomaly_detection import AnomalyDetector
    
    # Import database manager
    from scripts.database_manager import DatabaseManager
    
    # Import risk analyzer
    from scripts.risk_analyzer import FlashLoanRiskAnalyzer
    
    # Import API app
    from scripts.risk_api import app as api_app
except Exception as e:
    logger.error(f"Import error: {str(e)}")
    logger.error(f"Traceback:", exc_info=True)
    raise

# Register the dashboard route
@api_app.route('/')
def index():
    return render_template('index.html')

# Register static files route
@api_app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Main execution
if __name__ == "__main__":
    try:
        # Set up paths
        BASE_DIR = Path(__file__).parent
        TEMPLATES_DIR = BASE_DIR / "templates"
        STATIC_DIR = BASE_DIR / "static"
        
        # Ensure template directory exists
        if not TEMPLATES_DIR.exists():
            logger.warning(f"Templates directory not found at {TEMPLATES_DIR}. Creating it.")
            TEMPLATES_DIR.mkdir(exist_ok=True)
        
        # Ensure static directory exists
        if not STATIC_DIR.exists():
            logger.warning(f"Static directory not found at {STATIC_DIR}. Creating it.")
            STATIC_DIR.mkdir(exist_ok=True)
        
        # Configure Flask app
        api_app.template_folder = str(TEMPLATES_DIR)
        api_app.static_folder = str(STATIC_DIR)
        
        # Get port from environment or use default
        port = int(os.environ.get("PORT", 5000))
        
        # Determine if we're in debug mode
        debug_mode = os.environ.get("FLASK_ENV") == "development"
        
        if debug_mode:
            logger.info(f"Starting Flash Loan Risk API in development mode...")
            api_app.run(host="0.0.0.0", port=port, debug=True)
        else:
            logger.info(f"Starting Flash Loan Risk API in production mode...")
            logger.info(f"Server running at http://0.0.0.0:{port}")
            logger.info(f"API documentation available at API_DOCUMENTATION.md")
            
            # Print a nice banner
            print("=" * 50)
            print("Flash Loan Risk API Server")
            print("=" * 50)
            print(f"Server running at: http://0.0.0.0:{port}")
            print(f"API documentation: API_DOCUMENTATION.md")
            print(f"Log file: logs\\api_server_{os.environ.get('LOG_DATE', '')}.log")
            print("=" * 50)
            
            # Start production server with Waitress
            serve(api_app, host="0.0.0.0", port=port, threads=int(os.environ.get("THREADS", 4)))
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        raise
