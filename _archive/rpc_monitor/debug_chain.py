from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

RPC_ENDPOINTS = {
    "infura": os.getenv("INFURA_RPC_URL"),
    "alchemy": os.getenv("ALCHEMY_RPC_URL"),
    "publicnode": "https://ethereum-sepolia.publicnode.com",
}

for name, url in RPC_ENDPOINTS.items():
    try:
        w3 = Web3(Web3.HTTPProvider(url))
        print(f"{name} -> chain_id: {w3.eth.chain_id}")
    except Exception as e:
        print(f"{name} -> ERROR: {e}")