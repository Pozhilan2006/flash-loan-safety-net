from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import json
import time
from pathlib import Path
import logging
from web3 import Web3

# Import custom modules
from scripts.risk_analyzer import FlashLoanRiskAnalyzer
from scripts.web3_integration import Web3Integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("flash_loan_api.log")
    ]
)
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load model with error handling
model_path = MODELS_DIR / "flash_loan_risk_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    logger.info(f"Loaded risk model from {model_path}")
else:
    model = None
    logger.warning(f"Risk model file not found: {model_path}")

# Initialize Flask app
app = Flask(__name__)

# Initialize risk analyzer
risk_analyzer = FlashLoanRiskAnalyzer()

# Initialize Web3 integration
web3_client = Web3Integration()

# Load token addresses from JSON file
token_addresses = {}
token_file = DATA_DIR / "token_addresses.json"
if token_file.exists():
    with open(token_file, "r") as f:
        token_addresses = json.load(f)
else:
    # Default token addresses (for testing)
    token_addresses = {
        "DAI": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",  # Polygon Mumbai DAI (example)
        "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",  # Polygon Mumbai USDC (example)
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",  # Polygon Mumbai USDT (example)
        "WETH": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619"   # Polygon Mumbai WETH (example)
    }
    # Save default addresses
    with open(token_file, "w") as f:
        json.dump(token_addresses, f, indent=4)

@app.route("/api/health", methods=["GET"])
def health_check():
    """API health check endpoint"""
    network_info = web3_client.get_network_info()

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "blockchain_connected": network_info["connected"],
        "network": network_info["network"],
        "block_number": network_info.get("block_number", "N/A")
    })

@app.route("/api/predict", methods=["POST"])
def predict_risk():
    """Predict risk using the ML model"""
    if not model:
        return jsonify({"error": "Model not loaded. Ensure flash_loan_risk_model.pkl exists."}), 500

    try:
        data = request.json
        required_fields = ["liquidityRate", "variableBorrowRate", "stableBorrowRate"]

        # Check if required fields are present
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields", "required_fields": required_fields}), 400

        # Extract base features
        liquidityRate = float(data["liquidityRate"])
        variableBorrowRate = float(data["variableBorrowRate"])
        stableBorrowRate = float(data["stableBorrowRate"])

        # Calculate derived features
        rate_spread = variableBorrowRate - liquidityRate
        rate_ratio = variableBorrowRate / (liquidityRate + 1e-10)  # Adding small value to prevent division by zero
        stable_variable_diff = stableBorrowRate - variableBorrowRate

        # Create feature array with all 6 features
        features = np.array([[
            liquidityRate,
            variableBorrowRate,
            stableBorrowRate,
            rate_spread,
            rate_ratio,
            stable_variable_diff
        ]])

        risk_probability = model.predict_proba(features)[0][1]
        risk_class = model.predict(features)[0]

        # Determine risk level
        if risk_probability >= 0.7:
            risk_level = "high"
        elif risk_probability >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return jsonify({
            "risk_score": float(risk_probability),
            "risk_level": risk_level,
            "risk_class": int(risk_class),
            "factors": {
                "liquidity_rate": liquidityRate,
                "borrow_rate": variableBorrowRate,
                "stable_borrow_rate": stableBorrowRate,
                "rate_spread": rate_spread,
                "rate_ratio": rate_ratio,
                "stable_variable_diff": stable_variable_diff
            }
        })

    except Exception as e:
        logger.error(f"Error in risk prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze_asset():
    """Analyze risk for a specific asset"""
    try:
        data = request.json

        if "symbol" not in data:
            return jsonify({"error": "Missing required field: symbol"}), 400

        # Get asset data from the risk analyzer
        asset_data = risk_analyzer.rule_based_risk_assessment(data)

        return jsonify(asset_data)

    except Exception as e:
        logger.error(f"Error in asset analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/blockchain/check-loan", methods=["POST"])
def check_flash_loan():
    """Check flash loan risk using the blockchain contract"""
    try:
        data = request.json
        required_fields = ["borrower", "tokens", "amounts"]

        # Check if required fields are present
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields", "required_fields": required_fields}), 400

        # Check if blockchain is connected
        network_info = web3_client.get_network_info()
        if not network_info.get("connected", False):
            logger.warning("Blockchain not connected. Returning simulated risk assessment.")

            # Generate a simulated risk assessment based on the input data
            total_amount = sum(data["amounts"])

            # Simple risk heuristic: higher amounts = higher risk
            if total_amount > 10000:
                risk_score = 0.85
                risk_level = "high"
            elif total_amount > 5000:
                risk_score = 0.5
                risk_level = "medium"
            else:
                risk_score = 0.2
                risk_level = "low"

            return jsonify({
                "safe": risk_score < 0.7,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "source": "simulation",
                "note": "This is a simulated assessment as blockchain connection is unavailable"
            })

        # Convert token symbols to addresses if needed
        tokens = []
        for token in data["tokens"]:
            if token.startswith("0x"):
                tokens.append(token)
            else:
                if token in token_addresses:
                    tokens.append(token_addresses[token])
                else:
                    return jsonify({"error": f"Unknown token symbol: {token}"}), 400

        # Check loan using blockchain contract
        result = web3_client.check_flash_loan(
            data["borrower"],
            tokens,
            data["amounts"]
        )

        if result is None:
            logger.warning("Failed to check flash loan on blockchain. Using simulated assessment.")

            # Generate a simulated risk assessment as fallback
            total_amount = sum(data["amounts"])

            # Simple risk heuristic
            if total_amount > 10000:
                risk_score = 0.85
                risk_level = "high"
            elif total_amount > 5000:
                risk_score = 0.5
                risk_level = "medium"
            else:
                risk_score = 0.2
                risk_level = "low"

            return jsonify({
                "safe": risk_score < 0.7,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "source": "simulation",
                "note": "This is a simulated assessment as blockchain contract call failed"
            })

        # Add source information to the result
        if isinstance(result, dict):
            result["source"] = "blockchain"

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error checking flash loan: {str(e)}")

        # Return a simulated assessment in case of error
        return jsonify({
            "safe": False,
            "risk_score": 0.75,
            "risk_level": "high",
            "source": "simulation",
            "error": str(e),
            "note": "This is a simulated assessment due to an error"
        })

@app.route("/api/blockchain/simulate-loan", methods=["POST"])
def simulate_flash_loan():
    """Simulate a flash loan using the blockchain contract"""
    try:
        data = request.json
        required_fields = ["receiver", "tokens", "amounts"]

        # Check if required fields are present
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields", "required_fields": required_fields}), 400

        # Check if blockchain is connected
        network_info = web3_client.get_network_info()
        if not network_info.get("connected", False):
            logger.warning("Blockchain not connected. Returning simulated loan result.")

            # Generate a simulated loan result
            total_amount = sum(data["amounts"])
            token_symbols = [t if not t.startswith("0x") else f"Token-{i}" for i, t in enumerate(data["tokens"])]

            # Simple simulation logic
            success = total_amount < 50000  # Simulate failure for very large loans

            return jsonify({
                "success": success,
                "result": {
                    "success": success,
                    "borrower": data["receiver"],
                    "tokens": token_symbols,
                    "amounts": data["amounts"],
                    "fees": [amount * 0.001 for amount in data["amounts"]],
                    "timestamp": int(time.time())
                },
                "source": "simulation",
                "note": "This is a simulated result as blockchain connection is unavailable"
            })

        # Convert token symbols to addresses if needed
        tokens = []
        for token in data["tokens"]:
            if token.startswith("0x"):
                tokens.append(token)
            else:
                if token in token_addresses:
                    tokens.append(token_addresses[token])
                else:
                    return jsonify({"error": f"Unknown token symbol: {token}"}), 400

        # Optional params
        params = data.get("params", b'')

        # Simulate loan using blockchain contract
        result = web3_client.simulate_flash_loan(
            data["receiver"],
            tokens,
            data["amounts"],
            params
        )

        if not result:
            logger.warning("Failed to simulate flash loan on blockchain. Using simulated result.")

            # Generate a simulated loan result as fallback
            total_amount = sum(data["amounts"])
            token_symbols = [t if not t.startswith("0x") else f"Token-{i}" for i, t in enumerate(data["tokens"])]

            # Simple simulation logic
            success = total_amount < 50000  # Simulate failure for very large loans

            return jsonify({
                "success": success,
                "result": {
                    "success": success,
                    "borrower": data["receiver"],
                    "tokens": token_symbols,
                    "amounts": data["amounts"],
                    "fees": [amount * 0.001 for amount in data["amounts"]],
                    "timestamp": int(time.time())
                },
                "source": "simulation",
                "note": "This is a simulated result as blockchain contract call failed"
            })

        # Add source information to the result
        return jsonify({
            "success": True,
            "result": result,
            "source": "blockchain"
        })

    except Exception as e:
        logger.error(f"Error simulating flash loan: {str(e)}")

        # Return a simulated result in case of error
        return jsonify({
            "success": False,
            "error": str(e),
            "result": {
                "success": False,
                "reason": "API error occurred",
                "error": str(e)
            },
            "source": "simulation",
            "note": "This is a simulated result due to an error"
        })

@app.route("/api/blockchain/token-price", methods=["GET"])
def get_token_price():
    """Get token price from the blockchain oracle"""
    try:
        token = request.args.get("token")

        if not token:
            return jsonify({"error": "Missing required parameter: token"}), 400

        # Convert token symbol to address if needed
        token_address = token
        if not token.startswith("0x"):
            if token in token_addresses:
                token_address = token_addresses[token]
            else:
                return jsonify({"error": f"Unknown token symbol: {token}"}), 400

        # Check if blockchain is connected
        network_info = web3_client.get_network_info()
        if not network_info.get("connected", False):
            # Return mock data if blockchain is not connected
            logger.warning(f"Blockchain not connected. Returning mock price for {token}")

            # Mock prices for common tokens
            mock_prices = {
                "DAI": 1.0,
                "USDC": 1.0,
                "USDT": 1.0,
                "WETH": 3500.0,
                "WBTC": 65000.0,
                "AAVE": 95.0,
                "LINK": 15.0,
                "UNI": 8.0,
                "COMP": 45.0,
                "MKR": 1800.0,
                "SNX": 3.0,
                "YFI": 9500.0,
                "SUSHI": 1.2,
                "BUSD": 1.0,
                "TUSD": 1.0
            }

            # Default price if token not in mock prices
            mock_price = mock_prices.get(token.upper(), 10.0)

            return jsonify({
                "token": token,
                "address": token_address,
                "price": mock_price,
                "source": "mock_data"
            })

        # Get price from blockchain oracle
        price = web3_client.get_asset_price(token_address)

        if price is None:
            logger.warning(f"Failed to get price from blockchain for {token}. Using mock data.")
            # Return mock data as fallback
            mock_price = 1.0 if token.upper() in ["DAI", "USDC", "USDT"] else 100.0
            return jsonify({
                "token": token,
                "address": token_address,
                "price": mock_price,
                "source": "mock_data"
            })

        return jsonify({
            "token": token,
            "address": token_address,
            "price": price,
            "source": "blockchain"
        })

    except Exception as e:
        logger.error(f"Error getting token price: {str(e)}")
        # Return mock data in case of any error
        mock_price = 1.0 if token.upper() in ["DAI", "USDC", "USDT"] else 100.0
        return jsonify({
            "token": token,
            "address": token_address,
            "price": mock_price,
            "source": "mock_data",
            "error": str(e)
        })

@app.route("/api/blockchain/supported-tokens", methods=["GET"])
def get_supported_tokens():
    """Get list of supported tokens"""
    try:
        return jsonify({
            "tokens": token_addresses
        })

    except Exception as e:
        logger.error(f"Error getting supported tokens: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/blockchain/network", methods=["GET"])
def get_network_info():
    """Get blockchain network information"""
    try:
        network_info = web3_client.get_network_info()
        return jsonify(network_info)

    except Exception as e:
        logger.error(f"Error getting network info: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flash Loan Risk API...")
    app.run(host="0.0.0.0", port=5000, debug=True)

