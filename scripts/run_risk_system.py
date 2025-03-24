#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("flash_loan_risk_system.log")
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

# Import our modules
try:
    # Add current directory to path to ensure imports work correctly
    current_dir = str(Path(__file__).parent)
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from risk_analyzer import FlashLoanRiskAnalyzer
    from data_integration import DeFiDataCollector
    from anomaly_detection import FlashLoanAnomalyDetector
    from sentiment_analysis import DeFiSentimentAnalyzer
    from database_manager import DatabaseManager
except ImportError as e:
    logger.error(f"Error importing required modules: {str(e)}")
    logger.error("Make sure all required modules are installed")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Flash Loan Risk Analysis System")
    
    parser.add_argument(
        "--mode", 
        choices=["once", "scheduled", "data-only", "analyze-only", "report-only"],
        default="once",
        help="Run mode: once (single run), scheduled (continuous), data-only (update data only), "
             "analyze-only (analyze existing data), report-only (generate report only)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Interval in minutes for scheduled mode (default: 60)"
    )
    
    parser.add_argument(
        "--db-type",
        choices=["sqlite", "postgres", "mongodb"],
        default="sqlite",
        help="Database type to use (default: sqlite)"
    )
    
    parser.add_argument(
        "--api-keys",
        type=str,
        help="Path to JSON file containing API keys"
    )
    
    return parser.parse_args()

def load_api_keys(api_keys_file):
    """Load API keys from a JSON file and set environment variables"""
    if not api_keys_file or not os.path.exists(api_keys_file):
        logger.warning("API keys file not provided or does not exist")
        return False
    
    try:
        import json
        with open(api_keys_file, "r") as f:
            keys = json.load(f)
        
        # Set environment variables
        for key, value in keys.items():
            os.environ[key] = value
        
        logger.info(f"Loaded API keys from {api_keys_file}")
        return True
    except Exception as e:
        logger.error(f"Error loading API keys: {str(e)}")
        return False

def run_data_collection():
    """Run data collection only"""
    logger.info("Running data collection...")
    collector = DeFiDataCollector()
    results = collector.update_all_data()
    
    success = any(results.values())
    if success:
        logger.info("Data collection completed successfully")
    else:
        logger.error("Data collection failed")
    
    return success

def run_analysis_only():
    """Run analysis on existing data"""
    logger.info("Running analysis on existing data...")
    analyzer = FlashLoanRiskAnalyzer()
    
    # Skip data update
    high_risk_assets = analyzer.analyze_asset_risks()
    suspicious_tx = analyzer.analyze_transactions()
    
    if high_risk_assets:
        logger.warning(f"Found {len(high_risk_assets)} high risk assets")
    else:
        logger.info("No high risk assets found")
    
    if suspicious_tx:
        logger.warning(f"Found {len(suspicious_tx)} suspicious transactions")
    else:
        logger.info("No suspicious transactions found")
    
    return True

def run_report_only():
    """Generate risk report only"""
    logger.info("Generating risk report...")
    analyzer = FlashLoanRiskAnalyzer()
    report = analyzer.generate_risk_report()
    
    if report:
        logger.info(f"Risk report generated with overall risk level: {report['overall_risk_level'].upper()}")
        return True
    else:
        logger.error("Failed to generate risk report")
        return False

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set database type environment variable
    os.environ["DB_TYPE"] = args.db_type
    
    # Load API keys if provided
    if args.api_keys:
        load_api_keys(args.api_keys)
    
    logger.info(f"Starting Flash Loan Risk Analysis System in {args.mode} mode")
    
    if args.mode == "data-only":
        run_data_collection()
    
    elif args.mode == "analyze-only":
        run_analysis_only()
    
    elif args.mode == "report-only":
        run_report_only()
    
    elif args.mode == "once":
        analyzer = FlashLoanRiskAnalyzer()
        analyzer.run_analysis()
    
    elif args.mode == "scheduled":
        logger.info(f"Running in scheduled mode every {args.interval} minutes")
        analyzer = FlashLoanRiskAnalyzer()
        analyzer.start_scheduled_analysis(interval_minutes=args.interval)
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)