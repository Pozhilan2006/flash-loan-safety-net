import os
import json
import logging
import time
from pathlib import Path
import requests
from datetime import datetime, timedelta
import pandas as pd

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

# API endpoints and keys
# Note: Replace with your actual API keys in production
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "YOUR_ETHERSCAN_API_KEY")
INFURA_API_KEY = os.getenv("INFURA_API_KEY", "YOUR_INFURA_API_KEY")
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY", "YOUR_ALCHEMY_API_KEY")

# DeFi platform endpoints
AAVE_API_URL = "https://api.thegraph.com/subgraphs/name/aave/protocol-v2"
COMPOUND_API_URL = "https://api.thegraph.com/subgraphs/name/compound-finance/compound-v2"
UNISWAP_API_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

class DeFiDataCollector:
    """Collects real-time data from various DeFi platforms"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.last_update = {}
        self.load_last_update_times()
    
    def load_last_update_times(self):
        """Load the last update times for each data source"""
        update_file = self.data_dir / "last_update.json"
        if update_file.exists():
            with open(update_file, "r") as f:
                self.last_update = json.load(f)
        else:
            # Default to 24 hours ago if no file exists
            default_time = (datetime.now() - timedelta(days=1)).isoformat()
            self.last_update = {
                "aave": default_time,
                "compound": default_time,
                "uniswap": default_time,
                "ethereum": default_time
            }
            self.save_last_update_times()
    
    def save_last_update_times(self):
        """Save the last update times for each data source"""
        update_file = self.data_dir / "last_update.json"
        with open(update_file, "w") as f:
            json.dump(self.last_update, f, indent=4)
    
    def fetch_aave_data(self):
        """Fetch reserve data from Aave protocol"""
        logger.info("Fetching Aave reserve data...")
        
        # GraphQL query to get reserve data
        query = """
        {
          reserves(first: 100) {
            id
            symbol
            name
            decimals
            liquidityRate
            variableBorrowRate
            stableBorrowRate
            totalLiquidity
            totalBorrows
            utilizationRate
            liquidityIndex
            variableBorrowIndex
            aEmissionPerSecond
            vEmissionPerSecond
            sEmissionPerSecond
            lastUpdateTimestamp
          }
        }
        """
        
        try:
            response = requests.post(AAVE_API_URL, json={"query": query})
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and "reserves" in data["data"]:
                reserves = data["data"]["reserves"]
                logger.info(f"Successfully fetched {len(reserves)} Aave reserves")
                
                # Save to file
                aave_file = self.data_dir / "aave_reserves.json"
                with open(aave_file, "w") as f:
                    json.dump(data, f, indent=4)
                
                # Update last update time
                self.last_update["aave"] = datetime.now().isoformat()
                self.save_last_update_times()
                
                return reserves
            else:
                logger.error("Invalid response format from Aave API")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Aave data: {str(e)}")
            return None
    
    def fetch_compound_data(self):
        """Fetch market data from Compound protocol"""
        logger.info("Fetching Compound market data...")
        
        # GraphQL query to get market data
        query = """
        {
          markets(first: 100) {
            id
            symbol
            borrowRate
            supplyRate
            totalBorrows
            totalSupply
            exchangeRate
            underlyingPriceUSD
            accrualBlockNumber
            blockTimestamp
            borrowIndex
            reserveFactor
            underlyingName
            underlyingSymbol
            underlyingDecimals
            underlyingAddress
          }
        }
        """
        
        try:
            response = requests.post(COMPOUND_API_URL, json={"query": query})
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and "markets" in data["data"]:
                markets = data["data"]["markets"]
                logger.info(f"Successfully fetched {len(markets)} Compound markets")
                
                # Save to file
                compound_file = self.data_dir / "compound_markets.json"
                with open(compound_file, "w") as f:
                    json.dump(data, f, indent=4)
                
                # Update last update time
                self.last_update["compound"] = datetime.now().isoformat()
                self.save_last_update_times()
                
                return markets
            else:
                logger.error("Invalid response format from Compound API")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Compound data: {str(e)}")
            return None
    
    def fetch_uniswap_data(self):
        """Fetch pool data from Uniswap protocol"""
        logger.info("Fetching Uniswap pool data...")
        
        # GraphQL query to get pool data
        query = """
        {
          pools(first: 100, orderBy: volumeUSD, orderDirection: desc) {
            id
            token0 {
              id
              symbol
              name
            }
            token1 {
              id
              symbol
              name
            }
            feeTier
            liquidity
            volumeUSD
            txCount
            totalValueLockedUSD
            volumeToken0
            volumeToken1
          }
        }
        """
        
        try:
            response = requests.post(UNISWAP_API_URL, json={"query": query})
            response.raise_for_status()
            data = response.json()
            
            if "data" in data and "pools" in data["data"]:
                pools = data["data"]["pools"]
                logger.info(f"Successfully fetched {len(pools)} Uniswap pools")
                
                # Save to file
                uniswap_file = self.data_dir / "uniswap_pools.json"
                with open(uniswap_file, "w") as f:
                    json.dump(data, f, indent=4)
                
                # Update last update time
                self.last_update["uniswap"] = datetime.now().isoformat()
                self.save_last_update_times()
                
                return pools
            else:
                logger.error("Invalid response format from Uniswap API")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Uniswap data: {str(e)}")
            return None
    
    def fetch_ethereum_transactions(self, address=None, contract=None):
        """Fetch recent Ethereum transactions, optionally filtered by address or contract"""
        logger.info("Fetching Ethereum transactions...")
        
        # Etherscan API endpoint
        url = f"https://api.etherscan.io/api"
        
        params = {
            "module": "account",
            "action": "txlist",
            "apikey": ETHERSCAN_API_KEY,
            "sort": "desc",
            "page": 1,
            "offset": 100  # Fetch last 100 transactions
        }
        
        if address:
            params["address"] = address
        elif contract:
            params["address"] = contract
        else:
            # If no address specified, get transactions from known flash loan contracts
            # Example Aave V2 lending pool address
            params["address"] = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "status" in data and data["status"] == "1" and "result" in data:
                transactions = data["result"]
                logger.info(f"Successfully fetched {len(transactions)} Ethereum transactions")
                
                # Save to file
                eth_file = self.data_dir / "ethereum_transactions.json"
                with open(eth_file, "w") as f:
                    json.dump(data, f, indent=4)
                
                # Update last update time
                self.last_update["ethereum"] = datetime.now().isoformat()
                self.save_last_update_times()
                
                return transactions
            else:
                logger.error(f"Error in Etherscan API response: {data.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching Ethereum transactions: {str(e)}")
            return None
    
    def merge_defi_data(self):
        """Merge data from different DeFi platforms into a unified dataset"""
        logger.info("Merging DeFi data from multiple sources...")
        
        try:
            # Load data from files
            aave_file = self.data_dir / "aave_reserves.json"
            compound_file = self.data_dir / "compound_markets.json"
            
            if not aave_file.exists() or not compound_file.exists():
                logger.warning("Missing data files. Fetching fresh data...")
                self.fetch_aave_data()
                self.fetch_compound_data()
            
            # Read the data
            with open(aave_file, "r") as f:
                aave_data = json.load(f)
            
            with open(compound_file, "r") as f:
                compound_data = json.load(f)
            
            # Extract reserves and markets
            aave_reserves = aave_data.get("data", {}).get("reserves", [])
            compound_markets = compound_data.get("data", {}).get("markets", [])
            
            # Create DataFrames
            aave_df = pd.DataFrame(aave_reserves)
            compound_df = pd.DataFrame(compound_markets)
            
            # Normalize and combine data
            # This is a simplified example - in production you'd want more sophisticated merging
            merged_data = []
            
            # Process Aave data
            for _, reserve in aave_df.iterrows():
                asset_data = {
                    "source": "aave",
                    "symbol": reserve.get("symbol"),
                    "liquidityRate": float(reserve.get("liquidityRate", 0)),
                    "borrowRate": float(reserve.get("variableBorrowRate", 0)),
                    "totalLiquidity": float(reserve.get("totalLiquidity", 0)),
                    "totalBorrows": float(reserve.get("totalBorrows", 0)),
                    "utilizationRate": float(reserve.get("utilizationRate", 0)),
                    "timestamp": reserve.get("lastUpdateTimestamp")
                }
                merged_data.append(asset_data)
            
            # Process Compound data
            for _, market in compound_df.iterrows():
                asset_data = {
                    "source": "compound",
                    "symbol": market.get("underlyingSymbol"),
                    "liquidityRate": float(market.get("supplyRate", 0)),
                    "borrowRate": float(market.get("borrowRate", 0)),
                    "totalLiquidity": float(market.get("totalSupply", 0)),
                    "totalBorrows": float(market.get("totalBorrows", 0)),
                    "utilizationRate": float(market.get("totalBorrows", 0)) / 
                                      (float(market.get("totalSupply", 1)) + 0.000001),
                    "timestamp": market.get("blockTimestamp")
                }
                merged_data.append(asset_data)
            
            # Create a unified DataFrame
            unified_df = pd.DataFrame(merged_data)
            
            # Save to CSV
            unified_file = self.data_dir / "unified_defi_data.csv"
            unified_df.to_csv(unified_file, index=False)
            
            logger.info(f"Successfully merged data from multiple DeFi platforms. Saved to {unified_file}")
            return unified_df
            
        except Exception as e:
            logger.error(f"Error merging DeFi data: {str(e)}")
            return None
    
    def update_all_data(self):
        """Update all data sources"""
        logger.info("Updating all DeFi data sources...")
        
        # Fetch data from all sources
        aave_data = self.fetch_aave_data()
        compound_data = self.fetch_compound_data()
        uniswap_data = self.fetch_uniswap_data()
        ethereum_data = self.fetch_ethereum_transactions()
        
        # Merge the data
        unified_data = self.merge_defi_data()
        
        return {
            "aave": aave_data is not None,
            "compound": compound_data is not None,
            "uniswap": uniswap_data is not None,
            "ethereum": ethereum_data is not None,
            "unified": unified_data is not None
        }

# Main execution
if __name__ == "__main__":
    collector = DeFiDataCollector()
    results = collector.update_all_data()
    
    # Print results
    for source, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{source.capitalize()} data update: {status}")