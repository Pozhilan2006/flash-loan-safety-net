 import requests
import json

# Example: Fetch Aave lending pool data (Modify with actual API)
AAVE_API_URL = "https://api.thegraph.com/subgraphs/name/aave/protocol-v2"

QUERY = """
{
  reserves(first: 5) {
    id
    name
    liquidityRate
    variableBorrowRate
    stableBorrowRate
  }
}
"""

def fetch_aave_data():
    response = requests.post(AAVE_API_URL, json={"query": QUERY})
    if response.status_code == 200:
        data = response.json()
        with open("flash_loan_data.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Data collected successfully!")
    else:
        print("Error fetching data:", response.text)

if __name__ == "__main__":
    fetch_aave_data()
