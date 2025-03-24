import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import pymongo
import psycopg2
from psycopg2.extras import Json as PsycopgJson
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Database configuration
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()  # Options: sqlite, postgres, mongodb
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")  # PostgreSQL default
DB_NAME = os.getenv("DB_NAME", "flash_loan_risk")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
MONGO_URI = os.getenv("MONGO_URI", f"mongodb://{DB_HOST}:27017/")

# SQLAlchemy setup
Base = declarative_base()

class RiskAssessment(Base):
    """SQLAlchemy model for risk assessment data"""
    __tablename__ = 'risk_assessments'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    asset_symbol = Column(String(10), nullable=False)
    liquidity_rate = Column(Float)
    borrow_rate = Column(Float)
    utilization_rate = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String(10))
    anomaly_score = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, default=False)
    sentiment_score = Column(Float, nullable=True)
    market_sentiment = Column(String(10), nullable=True)
    data_source = Column(String(50))
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata to avoid conflict

class FlashLoanTransaction(Base):
    """SQLAlchemy model for flash loan transactions"""
    __tablename__ = 'flash_loan_transactions'

    id = Column(Integer, primary_key=True)
    transaction_hash = Column(String(66), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    block_number = Column(Integer)
    borrower_address = Column(String(42))
    asset_symbol = Column(String(10))
    amount = Column(Float)
    fee = Column(Float)
    platform = Column(String(50))
    risk_score = Column(Float)
    is_suspicious = Column(Boolean, default=False)
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata to avoid conflict

class MarketSentiment(Base):
    """SQLAlchemy model for market sentiment data"""
    __tablename__ = 'market_sentiment'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    source = Column(String(50))
    sentiment_polarity = Column(Float)
    risk_score = Column(Float)
    overall_sentiment = Column(String(10))
    risk_level = Column(String(10))
    extra_data = Column(JSON, nullable=True)  # Renamed from metadata to avoid conflict

class DatabaseManager:
    """Manages database connections and operations for the flash loan risk system"""
    
    def __init__(self):
        self.db_type = DB_TYPE
        self.connection = None
        self.engine = None
        self.session = None
        self.connect()
    
    def connect(self):
        """Connect to the database based on configuration"""
        try:
            if self.db_type == "sqlite":
                db_path = DATA_DIR / "flash_loan_risk.db"
                connection_string = f"sqlite:///{db_path}"
                self.engine = create_engine(connection_string)
                Base.metadata.create_all(self.engine)
                Session = sessionmaker(bind=self.engine)
                self.session = Session()
                logger.info(f"Connected to SQLite database at {db_path}")
                
            elif self.db_type == "postgres":
                connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                self.engine = create_engine(connection_string)
                Base.metadata.create_all(self.engine)
                Session = sessionmaker(bind=self.engine)
                self.session = Session()
                logger.info(f"Connected to PostgreSQL database at {DB_HOST}:{DB_PORT}/{DB_NAME}")
                
            elif self.db_type == "mongodb":
                self.connection = pymongo.MongoClient(MONGO_URI)
                self.db = self.connection[DB_NAME]
                logger.info(f"Connected to MongoDB at {MONGO_URI}")
                
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection"""
        try:
            if self.db_type in ["sqlite", "postgres"] and self.session:
                self.session.close()
                self.engine.dispose()
                logger.info("Closed SQL database connection")
                
            elif self.db_type == "mongodb" and self.connection:
                self.connection.close()
                logger.info("Closed MongoDB connection")
                
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
    
    def store_risk_assessment(self, assessment_data):
        """Store risk assessment data in the database"""
        try:
            if not assessment_data:
                logger.warning("No risk assessment data to store")
                return False
            
            if isinstance(assessment_data, list):
                # Handle batch insert
                batch_data = assessment_data
            else:
                # Handle single record
                batch_data = [assessment_data]
            
            if self.db_type in ["sqlite", "postgres"]:
                # SQL database storage
                for data in batch_data:
                    assessment = RiskAssessment(
                        timestamp=data.get("timestamp", datetime.now()),
                        asset_symbol=data.get("asset_symbol", "UNKNOWN"),
                        liquidity_rate=data.get("liquidity_rate", 0.0),
                        borrow_rate=data.get("borrow_rate", 0.0),
                        utilization_rate=data.get("utilization_rate", 0.0),
                        risk_score=data.get("risk_score", 0.0),
                        risk_level=data.get("risk_level", "unknown"),
                        anomaly_score=data.get("anomaly_score"),
                        is_anomaly=data.get("is_anomaly", False),
                        sentiment_score=data.get("sentiment_score"),
                        market_sentiment=data.get("market_sentiment"),
                        data_source=data.get("data_source", "unknown"),
                        extra_data=data.get("metadata", {})
                    )
                    self.session.add(assessment)
                
                self.session.commit()
                logger.info(f"Stored {len(batch_data)} risk assessment records in SQL database")
                
            elif self.db_type == "mongodb":
                # MongoDB storage
                collection = self.db["risk_assessments"]
                
                # Add timestamp if not present
                for data in batch_data:
                    if "timestamp" not in data:
                        data["timestamp"] = datetime.now()
                
                result = collection.insert_many(batch_data)
                logger.info(f"Stored {len(result.inserted_ids)} risk assessment records in MongoDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing risk assessment data: {str(e)}")
            
            if self.db_type in ["sqlite", "postgres"] and self.session:
                self.session.rollback()
                
            return False
    
    def store_flash_loan_transaction(self, transaction_data):
        """Store flash loan transaction data in the database"""
        try:
            if not transaction_data:
                logger.warning("No transaction data to store")
                return False
            
            if isinstance(transaction_data, list):
                # Handle batch insert
                batch_data = transaction_data
            else:
                # Handle single record
                batch_data = [transaction_data]
            
            if self.db_type in ["sqlite", "postgres"]:
                # SQL database storage
                for data in batch_data:
                    # Check if transaction already exists
                    tx_hash = data.get("transaction_hash")
                    existing = self.session.query(FlashLoanTransaction).filter_by(transaction_hash=tx_hash).first()
                    
                    if existing:
                        logger.debug(f"Transaction {tx_hash} already exists, skipping")
                        continue
                    
                    transaction = FlashLoanTransaction(
                        transaction_hash=tx_hash,
                        timestamp=data.get("timestamp", datetime.now()),
                        block_number=data.get("block_number", 0),
                        borrower_address=data.get("borrower_address", ""),
                        asset_symbol=data.get("asset_symbol", "UNKNOWN"),
                        amount=data.get("amount", 0.0),
                        fee=data.get("fee", 0.0),
                        platform=data.get("platform", "unknown"),
                        risk_score=data.get("risk_score", 0.0),
                        is_suspicious=data.get("is_suspicious", False),
                        extra_data=data.get("metadata", {})
                    )
                    self.session.add(transaction)
                
                self.session.commit()
                logger.info(f"Stored {len(batch_data)} flash loan transactions in SQL database")
                
            elif self.db_type == "mongodb":
                # MongoDB storage
                collection = self.db["flash_loan_transactions"]
                
                # Add timestamp if not present and check for duplicates
                new_transactions = []
                for data in batch_data:
                    if "timestamp" not in data:
                        data["timestamp"] = datetime.now()
                    
                    # Check if transaction already exists
                    tx_hash = data.get("transaction_hash")
                    existing = collection.find_one({"transaction_hash": tx_hash})
                    
                    if not existing:
                        new_transactions.append(data)
                    else:
                        logger.debug(f"Transaction {tx_hash} already exists, skipping")
                
                if new_transactions:
                    result = collection.insert_many(new_transactions)
                    logger.info(f"Stored {len(result.inserted_ids)} flash loan transactions in MongoDB")
                else:
                    logger.info("No new transactions to store")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing flash loan transaction data: {str(e)}")
            
            if self.db_type in ["sqlite", "postgres"] and self.session:
                self.session.rollback()
                
            return False
    
    def store_market_sentiment(self, sentiment_data):
        """Store market sentiment data in the database"""
        try:
            if not sentiment_data:
                logger.warning("No sentiment data to store")
                return False
            
            if self.db_type in ["sqlite", "postgres"]:
                # SQL database storage
                sentiment = MarketSentiment(
                    timestamp=sentiment_data.get("timestamp", datetime.now()),
                    source=sentiment_data.get("source", "combined"),
                    sentiment_polarity=sentiment_data.get("sentiment_polarity", 0.0),
                    risk_score=sentiment_data.get("risk_score", 0.0),
                    overall_sentiment=sentiment_data.get("overall_sentiment", "neutral"),
                    risk_level=sentiment_data.get("risk_level", "low"),
                    extra_data=sentiment_data.get("metadata", {})
                )
                self.session.add(sentiment)
                self.session.commit()
                logger.info("Stored market sentiment data in SQL database")
                
            elif self.db_type == "mongodb":
                # MongoDB storage
                collection = self.db["market_sentiment"]
                
                # Add timestamp if not present
                if "timestamp" not in sentiment_data:
                    sentiment_data["timestamp"] = datetime.now()
                
                result = collection.insert_one(sentiment_data)
                logger.info(f"Stored market sentiment data in MongoDB with ID: {result.inserted_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing market sentiment data: {str(e)}")
            
            if self.db_type in ["sqlite", "postgres"] and self.session:
                self.session.rollback()
                
            return False
    
    def get_latest_risk_assessments(self, limit=10, asset_symbol=None):
        """Get the latest risk assessments from the database"""
        try:
            if self.db_type in ["sqlite", "postgres"]:
                # SQL database query
                query = self.session.query(RiskAssessment).order_by(RiskAssessment.timestamp.desc())
                
                if asset_symbol:
                    query = query.filter(RiskAssessment.asset_symbol == asset_symbol)
                
                results = query.limit(limit).all()
                
                # Convert to dictionaries
                assessments = []
                for result in results:
                    assessment = {
                        "id": result.id,
                        "timestamp": result.timestamp,
                        "asset_symbol": result.asset_symbol,
                        "liquidity_rate": result.liquidity_rate,
                        "borrow_rate": result.borrow_rate,
                        "utilization_rate": result.utilization_rate,
                        "risk_score": result.risk_score,
                        "risk_level": result.risk_level,
                        "anomaly_score": result.anomaly_score,
                        "is_anomaly": result.is_anomaly,
                        "sentiment_score": result.sentiment_score,
                        "market_sentiment": result.market_sentiment,
                        "data_source": result.data_source,
                        "metadata": result.extra_data
                    }
                    assessments.append(assessment)
                
                return assessments
                
            elif self.db_type == "mongodb":
                # MongoDB query
                collection = self.db["risk_assessments"]
                
                query = {}
                if asset_symbol:
                    query["asset_symbol"] = asset_symbol
                
                results = collection.find(query).sort("timestamp", -1).limit(limit)
                
                # Convert to list and remove MongoDB _id
                assessments = []
                for result in results:
                    result.pop("_id", None)
                    assessments.append(result)
                
                return assessments
                
        except Exception as e:
            logger.error(f"Error getting latest risk assessments: {str(e)}")
            return []
    
    def get_suspicious_transactions(self, days=7):
        """Get suspicious flash loan transactions from the past X days"""
        try:
            # Calculate date threshold
            threshold = datetime.now() - pd.Timedelta(days=days)
            
            if self.db_type in ["sqlite", "postgres"]:
                # SQL database query
                results = self.session.query(FlashLoanTransaction).filter(
                    FlashLoanTransaction.is_suspicious == True,
                    FlashLoanTransaction.timestamp >= threshold
                ).order_by(FlashLoanTransaction.risk_score.desc()).all()
                
                # Convert to dictionaries
                transactions = []
                for result in results:
                    transaction = {
                        "id": result.id,
                        "transaction_hash": result.transaction_hash,
                        "timestamp": result.timestamp,
                        "block_number": result.block_number,
                        "borrower_address": result.borrower_address,
                        "asset_symbol": result.asset_symbol,
                        "amount": result.amount,
                        "fee": result.fee,
                        "platform": result.platform,
                        "risk_score": result.risk_score,
                        "is_suspicious": result.is_suspicious,
                        "metadata": result.extra_data
                    }
                    transactions.append(transaction)
                
                return transactions
                
            elif self.db_type == "mongodb":
                # MongoDB query
                collection = self.db["flash_loan_transactions"]
                
                results = collection.find({
                    "is_suspicious": True,
                    "timestamp": {"$gte": threshold}
                }).sort("risk_score", -1)
                
                # Convert to list and remove MongoDB _id
                transactions = []
                for result in results:
                    result.pop("_id", None)
                    transactions.append(result)
                
                return transactions
                
        except Exception as e:
            logger.error(f"Error getting suspicious transactions: {str(e)}")
            return []
    
    def get_market_sentiment_history(self, days=30):
        """Get market sentiment history for the past X days"""
        try:
            # Calculate date threshold
            threshold = datetime.now() - pd.Timedelta(days=days)
            
            if self.db_type in ["sqlite", "postgres"]:
                # SQL database query
                results = self.session.query(MarketSentiment).filter(
                    MarketSentiment.timestamp >= threshold
                ).order_by(MarketSentiment.timestamp.asc()).all()
                
                # Convert to dictionaries
                sentiment_history = []
                for result in results:
                    sentiment = {
                        "id": result.id,
                        "timestamp": result.timestamp,
                        "source": result.source,
                        "sentiment_polarity": result.sentiment_polarity,
                        "risk_score": result.risk_score,
                        "overall_sentiment": result.overall_sentiment,
                        "risk_level": result.risk_level
                    }
                    sentiment_history.append(sentiment)
                
                return sentiment_history
                
            elif self.db_type == "mongodb":
                # MongoDB query
                collection = self.db["market_sentiment"]
                
                results = collection.find({
                    "timestamp": {"$gte": threshold}
                }).sort("timestamp", 1)
                
                # Convert to list and remove MongoDB _id
                sentiment_history = []
                for result in results:
                    result.pop("_id", None)
                    sentiment_history.append(result)
                
                return sentiment_history
                
        except Exception as e:
            logger.error(f"Error getting market sentiment history: {str(e)}")
            return []
    
    def export_data_to_csv(self, data_type, output_path=None):
        """Export data from the database to CSV files"""
        try:
            if output_path is None:
                output_dir = DATA_DIR / "exports"
                output_dir.mkdir(exist_ok=True)
            else:
                output_dir = Path(output_path)
                output_dir.mkdir(exist_ok=True)
            
            today = datetime.now().strftime("%Y-%m-%d")
            
            if data_type == "risk_assessments":
                # Get risk assessment data
                if self.db_type in ["sqlite", "postgres"]:
                    query = self.session.query(RiskAssessment).order_by(RiskAssessment.timestamp.desc())
                    df = pd.read_sql(query.statement, self.session.bind)
                    
                elif self.db_type == "mongodb":
                    collection = self.db["risk_assessments"]
                    data = list(collection.find())
                    for item in data:
                        item.pop("_id", None)
                    df = pd.DataFrame(data)
                
                # Save to CSV
                output_file = output_dir / f"risk_assessments_{today}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Exported {len(df)} risk assessments to {output_file}")
                
            elif data_type == "transactions":
                # Get transaction data
                if self.db_type in ["sqlite", "postgres"]:
                    query = self.session.query(FlashLoanTransaction).order_by(FlashLoanTransaction.timestamp.desc())
                    df = pd.read_sql(query.statement, self.session.bind)
                    
                elif self.db_type == "mongodb":
                    collection = self.db["flash_loan_transactions"]
                    data = list(collection.find())
                    for item in data:
                        item.pop("_id", None)
                    df = pd.DataFrame(data)
                
                # Save to CSV
                output_file = output_dir / f"flash_loan_transactions_{today}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Exported {len(df)} transactions to {output_file}")
                
            elif data_type == "sentiment":
                # Get sentiment data
                if self.db_type in ["sqlite", "postgres"]:
                    query = self.session.query(MarketSentiment).order_by(MarketSentiment.timestamp.desc())
                    df = pd.read_sql(query.statement, self.session.bind)
                    
                elif self.db_type == "mongodb":
                    collection = self.db["market_sentiment"]
                    data = list(collection.find())
                    for item in data:
                        item.pop("_id", None)
                    df = pd.DataFrame(data)
                
                # Save to CSV
                output_file = output_dir / f"market_sentiment_{today}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Exported {len(df)} sentiment records to {output_file}")
                
            else:
                logger.error(f"Unknown data type: {data_type}")
                return None
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {str(e)}")
            return None

# Main execution
if __name__ == "__main__":
    # Example usage
    db_manager = DatabaseManager()
    
    try:
        # Example: Store a risk assessment
        sample_assessment = {
            "timestamp": datetime.now(),
            "asset_symbol": "DAI",
            "liquidity_rate": 0.03,
            "borrow_rate": 0.05,
            "utilization_rate": 0.65,
            "risk_score": 0.42,
            "risk_level": "medium",
            "anomaly_score": 0.1,
            "is_anomaly": False,
            "sentiment_score": -0.2,
            "market_sentiment": "negative",
            "data_source": "aave",
            "metadata": {
                "total_liquidity": 1000000,
                "total_borrows": 650000
            }
        }
        
        db_manager.store_risk_assessment(sample_assessment)
        
        # Example: Get latest risk assessments
        latest = db_manager.get_latest_risk_assessments(limit=5)
        logger.info(f"Retrieved {len(latest)} latest risk assessments")
        
        # Example: Export data
        db_manager.export_data_to_csv("risk_assessments")
        
    finally:
        # Close the database connection
        db_manager.close()