import logging
import numpy as np
import os
import joblib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

class AnomalyDetector:
    """Detects anomalies in flash loan transactions"""
    
    def __init__(self):
        """Initialize the anomaly detector"""
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the anomaly detection model"""
        model_path = MODELS_DIR / "anomaly_detection_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Loaded anomaly detection model from {model_path}")
            return model
        else:
            logger.info(f"No existing anomaly detection model found")
            return None
    
    def detect_anomalies(self, transactions):
        """Detect anomalies in a list of transactions"""
        if not self.model:
            # Simple threshold-based detection as fallback
            return self._threshold_based_detection(transactions)
        
        try:
            # Extract features from transactions
            features = self._extract_features(transactions)
            
            # Predict anomalies
            anomaly_scores = self.model.decision_function(features)
            anomalies = self.model.predict(features)
            
            # Map results back to transactions
            results = []
            for i, transaction in enumerate(transactions):
                results.append({
                    "transaction": transaction,
                    "anomaly_score": float(anomaly_scores[i]),
                    "is_anomaly": bool(anomalies[i] == -1)
                })
            
            return results
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return self._threshold_based_detection(transactions)
    
    def _extract_features(self, transactions):
        """Extract features from transactions for anomaly detection"""
        features = []
        for tx in transactions:
            # Extract relevant features
            amount = tx.get("amount", 0)
            gas_price = tx.get("gas_price", 0)
            timestamp = tx.get("timestamp", 0)
            
            # Add more features as needed
            features.append([amount, gas_price, timestamp])
        
        return np.array(features)
    
    def _threshold_based_detection(self, transactions):
        """Simple threshold-based anomaly detection"""
        results = []
        
        # Calculate statistics
        amounts = [tx.get("amount", 0) for tx in transactions]
        if not amounts:
            return []
        
        mean_amount = sum(amounts) / len(amounts)
        std_amount = np.std(amounts) if len(amounts) > 1 else 0
        
        # Detect anomalies
        for tx in transactions:
            amount = tx.get("amount", 0)
            
            # Z-score
            z_score = (amount - mean_amount) / (std_amount + 1e-10)
            is_anomaly = abs(z_score) > 3  # More than 3 standard deviations
            
            results.append({
                "transaction": tx,
                "anomaly_score": float(abs(z_score)),
                "is_anomaly": is_anomaly
            })
        
        return results

# For testing
if __name__ == "__main__":
    detector = AnomalyDetector()
    
    # Test with sample transactions
    sample_transactions = [
        {"amount": 1000, "gas_price": 50, "timestamp": 1625097600},
        {"amount": 1200, "gas_price": 55, "timestamp": 1625097700},
        {"amount": 900, "gas_price": 45, "timestamp": 1625097800},
        {"amount": 10000, "gas_price": 100, "timestamp": 1625097900}  # Anomaly
    ]
    
    results = detector.detect_anomalies(sample_transactions)
    print("Anomaly detection results:")
    for result in results:
        print(f"  Amount: {result['transaction']['amount']}, Score: {result['anomaly_score']}, Anomaly: {result['is_anomaly']}")
