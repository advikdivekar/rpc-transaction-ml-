import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Secrets
DATABASE_URL = os.getenv("DATABASE_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
MY_ADDRESS = os.getenv("MY_ADDRESS")

# Constants
MODEL_PATH = "rpc_model.json"
CHAIN_ID = 11155111  # Sepolia

# The RPC List (Single Source of Truth)
RPC_PROVIDERS = {
    "infura": "https://sepolia.infura.io/v3/f6dccf73ccd64c06a5e7734325927bb9",
    "alchemy": "https://eth-sepolia.g.alchemy.com/v2/demo",
    "drpc": "https://sepolia.drpc.org",
    "publicnode": "https://ethereum-sepolia-rpc.publicnode.com"
}