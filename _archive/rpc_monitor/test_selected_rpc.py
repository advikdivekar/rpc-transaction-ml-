import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline_engine.lowest_latency import get_best_rpc
from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

RPCS = {
    "infura": os.getenv("INFURA_RPC_URL"),
    "alchemy": os.getenv("ALCHEMY_RPC_URL"),
    "publicnode": "https://ethereum-sepolia.publicnode.com"
}

rpc = get_best_rpc()

if rpc:
    w3 = Web3(Web3.HTTPProvider(RPCS[rpc]))
    print("Using:", rpc)
    print("Latest block:", w3.eth.block_number)
else:
    print("No RPC selected.")