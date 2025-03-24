import logging
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3, HTTPProvider

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONTRACTS_DIR = BASE_DIR / "contracts"

class Web3Integration:
    """Handles integration with blockchain networks"""
    
    def __init__(self):
        """Initialize Web3 integration"""
        self.web3 = None
        self.network = os.getenv("NETWORK", "mumbai")
        self.flash_loan_contract = None
        self.price_oracle_contract = None
        self._connect_to_network()
        self._load_contracts()
    
    def _connect_to_network(self):
        """Connect to the blockchain network"""
        try:
            # Get Infura API key from environment
            infura_key = os.getenv("INFURA_API_KEY", "")
            
            # Set up provider based on network
            if self.network == "mainnet":
                provider_url = f"https://mainnet.infura.io/v3/{infura_key}"
            elif self.network == "polygon":
                provider_url = f"https://polygon-mainnet.infura.io/v3/{infura_key}"
            elif self.network == "mumbai":
                provider_url = f"https://polygon-mumbai.infura.io/v3/{infura_key}"
                # Fallback to public RPC if no Infura key
                if not infura_key:
                    provider_url = "https://rpc-mumbai.maticvigil.com"
            else:
                provider_url = f"https://{self.network}.infura.io/v3/{infura_key}"
            
            # Connect to network
            self.web3 = Web3(HTTPProvider(provider_url))
            
            # Check connection
            if self.web3.is_connected():
                logger.info(f"Connected to {self.network} network")
            else:
                logger.warning(f"Failed to connect to {self.network} network")
        except Exception as e:
            logger.error(f"Error connecting to blockchain: {str(e)}")
            self.web3 = Web3()  # Empty Web3 instance
    
    def _load_contracts(self):
        """Load contract ABIs and addresses"""
        try:
            # Load deployment info
            deployment_file = DATA_DIR / "deployment-info.json"
            if deployment_file.exists():
                with open(deployment_file, "r") as f:
                    deployment_info = json.load(f)
                
                # Get contract info for current network
                network_info = deployment_info.get(self.network, {})
                
                # Load flash loan contract
                flash_loan_address = network_info.get("flashLoanContract", "")
                flash_loan_abi_file = CONTRACTS_DIR / "FlashLoanRisk.json"
                
                if flash_loan_address and flash_loan_abi_file.exists():
                    with open(flash_loan_abi_file, "r") as f:
                        contract_json = json.load(f)
                        flash_loan_abi = contract_json.get("abi", [])
                    
                    self.flash_loan_contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(flash_loan_address),
                        abi=flash_loan_abi
                    )
                    logger.info(f"Loaded flash loan contract at {flash_loan_address}")
                
                # Load price oracle contract
                oracle_address = network_info.get("priceOracleContract", "")
                oracle_abi_file = CONTRACTS_DIR / "PriceOracle.json"
                
                if oracle_address and oracle_abi_file.exists():
                    with open(oracle_abi_file, "r") as f:
                        contract_json = json.load(f)
                        oracle_abi = contract_json.get("abi", [])
                    
                    self.price_oracle_contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(oracle_address),
                        abi=oracle_abi
                    )
                    logger.info(f"Loaded price oracle contract at {oracle_address}")
            else:
                logger.warning(f"Deployment info file not found: {deployment_file}")
        except Exception as e:
            logger.error(f"Error loading contracts: {str(e)}")
    
    def get_network_info(self):
        """Get information about the connected network"""
        try:
            if not self.web3 or not self.web3.is_connected():
                return {
                    "connected": False,
                    "network": self.network
                }
            
            # Get network info
            chain_id = self.web3.eth.chain_id
            block_number = self.web3.eth.block_number
            gas_price = self.web3.eth.gas_price
            
            return {
                "connected": True,
                "network": self.network,
                "chain_id": chain_id,
                "block_number": block_number,
                "gas_price": gas_price
            }
        except Exception as e:
            logger.error(f"Error getting network info: {str(e)}")
            return {
                "connected": False,
                "network": self.network,
                "error": str(e)
            }
    
    def check_flash_loan(self, borrower, tokens, amounts):
        """Check flash loan risk using the blockchain contract"""
        try:
            if not self.web3 or not self.web3.is_connected() or not self.flash_loan_contract:
                logger.warning("Cannot check flash loan: Web3 not connected or contract not loaded")
                return None
            
            # Convert addresses to checksum format
            borrower = self.web3.to_checksum_address(borrower)
            tokens = [self.web3.to_checksum_address(token) for token in tokens]
            
            # Call contract method
            result = self.flash_loan_contract.functions.checkFlashLoanRisk(
                borrower,
                tokens,
                amounts
            ).call()
            
            # Parse result
            return {
                "safe": result[0],
                "risk_score": result[1] / 100.0,  # Convert from percentage
                "risk_level": self._risk_level_from_score(result[1] / 100.0)
            }
        except Exception as e:
            logger.error(f"Error checking flash loan: {str(e)}")
            return None
    
    def simulate_flash_loan(self, receiver, tokens, amounts, params=b''):
        """Simulate a flash loan using the blockchain contract"""
        try:
            if not self.web3 or not self.web3.is_connected() or not self.flash_loan_contract:
                logger.warning("Cannot simulate flash loan: Web3 not connected or contract not loaded")
                return None
            
            # Convert addresses to checksum format
            receiver = self.web3.to_checksum_address(receiver)
            tokens = [self.web3.to_checksum_address(token) for token in tokens]
            
            # Call contract method
            result = self.flash_loan_contract.functions.simulateFlashLoan(
                receiver,
                tokens,
                amounts,
                params
            ).call()
            
            # Parse result
            return {
                "success": result[0],
                "borrower": receiver,
                "tokens": tokens,
                "amounts": amounts,
                "fees": [amount * 0.0009 for amount in amounts],  # 0.09% fee
                "timestamp": self.web3.eth.get_block('latest').timestamp
            }
        except Exception as e:
            logger.error(f"Error simulating flash loan: {str(e)}")
            return None
    
    def get_asset_price(self, token_address):
        """Get asset price from the price oracle"""
        try:
            if not self.web3 or not self.web3.is_connected() or not self.price_oracle_contract:
                logger.warning("Cannot get asset price: Web3 not connected or contract not loaded")
                return None
            
            # Convert address to checksum format
            token_address = self.web3.to_checksum_address(token_address)
            
            # Call contract method
            price = self.price_oracle_contract.functions.getAssetPrice(token_address).call()
            
            # Convert to human-readable format (assuming 8 decimals)
            return price / 10**8
        except Exception as e:
            logger.error(f"Error getting asset price: {str(e)}")
            return None
    
    def get_supported_tokens(self):
        """Get list of supported tokens"""
        try:
            # Load token addresses from JSON file
            token_file = DATA_DIR / "token_addresses.json"
            if token_file.exists():
                with open(token_file, "r") as f:
                    token_addresses = json.load(f)
                return token_addresses
            else:
                # Default token addresses (for testing)
                return {
                    "DAI": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
                    "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                    "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
                    "WETH": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
                    "WBTC": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
                    "AAVE": "0xD6DF932A45C0f255f85145f286eA0b292B21C90B",
                    "LINK": "0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39",
                    "UNI": "0xb33EaAd8d922B1083446DC23f610c2567fB5180f",
                    "COMP": "0x8505b9d2254A7Ae468c0E9dd10Ccea3A837aef5c",
                    "MKR": "0x6f7C932e7684666C9fd1d44527765433e01fF61d",
                    "SNX": "0x50B728D8D964fd00C2d0AAD81718b71311feF68a",
                    "YFI": "0xDA537104D6A5edd53c6fBba9A898708E465260b6",
                    "SUSHI": "0x0b3F868E0BE5597D5DB7fEB59E1CADBb0fdDa50a",
                    "BUSD": "0x9C9e5fD8bbc25984B178FdCE6117Defa39d2db39",
                    "TUSD": "0x2e1AD108fF1D8C782fcBbB89AAd783aC49586756"
                }
        except Exception as e:
            logger.error(f"Error getting supported tokens: {str(e)}")
            return {}
    
    def _risk_level_from_score(self, risk_score):
        """Convert risk score to risk level"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"

# For testing
if __name__ == "__main__":
    web3_client = Web3Integration()
    
    # Test network connection
    network_info = web3_client.get_network_info()
    print("Network info:")
    for key, value in network_info.items():
        print(f"  {key}: {value}")
    
    # Test supported tokens
    tokens = web3_client.get_supported_tokens()
    print(f"Supported tokens: {len(tokens)}")
    for symbol, address in list(tokens.items())[:5]:
        print(f"  {symbol}: {address}")
