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

class FlashLoanRiskAnalyzer:
    """Analyzes flash loan risk using various methods"""
    
    def __init__(self):
        """Initialize the risk analyzer"""
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the risk model"""
        model_path = MODELS_DIR / "flash_loan_risk_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Loaded risk model from {model_path}")
            return model
        else:
            logger.warning(f"Risk model file not found: {model_path}")
            return None
    
    def predict_risk(self, features):
        """Predict risk using the ML model"""
        if not self.model:
            logger.error("Model not loaded. Cannot predict risk.")
            return None
        
        try:
            # Convert features to numpy array
            features_array = np.array([features])
            
            # Predict risk
            risk_probability = self.model.predict_proba(features_array)[0][1]
            risk_class = self.model.predict(features_array)[0]
            
            # Determine risk level
            if risk_probability >= 0.7:
                risk_level = "high"
            elif risk_probability >= 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "risk_score": float(risk_probability),
                "risk_level": risk_level,
                "risk_class": int(risk_class)
            }
        except Exception as e:
            logger.error(f"Error predicting risk: {str(e)}")
            return None
    
    def rule_based_risk_assessment(self, data):
        """Perform rule-based risk assessment"""
        try:
            symbol = data.get("symbol", "").upper()
            amount = data.get("amount", 0)
            
            # Default risk factors
            volatility = 0.1
            liquidity = 1000000
            market_cap = 1000000000
            
            # Adjust risk factors based on token
            if symbol in ["USDC", "USDT", "DAI", "BUSD", "TUSD"]:
                # Stablecoins have low volatility
                volatility = 0.01
                liquidity = 10000000
            elif symbol in ["WETH", "WBTC"]:
                # Major assets have medium volatility but high liquidity
                volatility = 0.05
                liquidity = 5000000
                market_cap = 100000000000
            elif symbol in ["AAVE", "UNI", "LINK", "COMP", "MKR"]:
                # DeFi tokens have higher volatility
                volatility = 0.15
                liquidity = 2000000
                market_cap = 5000000000
            
            # Calculate risk score based on amount and token characteristics
            amount_factor = min(1.0, amount / liquidity)
            volatility_factor = volatility * 5  # Scale up volatility impact
            
            # Combined risk score (0-1)
            risk_score = (amount_factor * 0.7) + (volatility_factor * 0.3)
            risk_score = min(1.0, max(0.0, risk_score))  # Ensure between 0 and 1
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "high"
            elif risk_score >= 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "symbol": symbol,
                "amount": amount,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "factors": {
                    "volatility": volatility,
                    "liquidity": liquidity,
                    "market_cap": market_cap,
                    "amount_factor": amount_factor,
                    "volatility_factor": volatility_factor
                }
            }
        except Exception as e:
            logger.error(f"Error in rule-based risk assessment: {str(e)}")
            return {
                "error": str(e),
                "symbol": data.get("symbol", ""),
                "amount": data.get("amount", 0),
                "risk_score": 0.75,  # Default to high risk on error
                "risk_level": "high"
            }

# For testing
if __name__ == "__main__":
    analyzer = FlashLoanRiskAnalyzer()
    
    # Test rule-based assessment
    test_data = {"symbol": "USDC", "amount": 10000}
    result = analyzer.rule_based_risk_assessment(test_data)
    print("Rule-based assessment result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
