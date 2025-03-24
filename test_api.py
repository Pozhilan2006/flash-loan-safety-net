import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def test_endpoint(method, endpoint, data=None, params=None):
    url = f"{BASE_URL}{endpoint}"
    print(f"Testing {method} {url}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"Unsupported method: {method}")
            return False
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            print("Test passed!")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        print("-" * 50)

def run_tests():
    tests = [
        {"method": "GET", "endpoint": "/api/health"},
        {"method": "POST", "endpoint": "/api/analyze", "data": {"symbol": "DAI"}},
        {"method": "GET", "endpoint": "/api/blockchain/supported-tokens"},
        {"method": "GET", "endpoint": "/api/blockchain/network"},
        {"method": "GET", "endpoint": "/api/blockchain/token-price", "params": {"token": "DAI"}},
        {"method": "POST", "endpoint": "/api/predict", "data": {
            "liquidityRate": 0.05,
            "variableBorrowRate": 0.08,
            "stableBorrowRate": 0.07
        }}
    ]
    
    success = 0
    total = len(tests)
    
    for test in tests:
        if test_endpoint(
            test["method"], 
            test["endpoint"], 
            data=test.get("data"), 
            params=test.get("params")
        ):
            success += 1
    
    print(f"Test Results: {success}/{total} tests passed")
    return success == total

if __name__ == "__main__":
    print("Testing Flash Loan Risk API...")
    success = run_tests()
    if not success:
        sys.exit(1)
