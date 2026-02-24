import time
import os
from datetime import datetime
import pandas as pd
from web3 import Web3
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# RPC endpoints
RPC_ENDPOINTS = {
    "infura": os.getenv("INFURA_RPC_URL"),
    "alchemy": os.getenv("ALCHEMY_RPC_URL"),
    "publicnode": "https://ethereum-sepolia.publicnode.com",
}

# MariaDB connection
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)

POLL_INTERVAL = 7  # seconds


def measure_rpc(rpc_id, url):
    try:
        w3 = Web3(Web3.HTTPProvider(url))
        start = time.time()
        block_number = w3.eth.block_number
        latency = (time.time() - start) * 1000
        return {
            "rpc_id": rpc_id,
            "latency_ms": latency,
            "block_number": block_number,
            "failure_flag": 0
        }
    except Exception:
        return {
            "rpc_id": rpc_id,
            "latency_ms": None,
            "block_number": None,
            "failure_flag": 1
        }


def main():
    print("Starting RPC monitor (MariaDB mode)...")

    while True:
        results = []
        timestamp = datetime.utcnow()

        for rpc_id, url in RPC_ENDPOINTS.items():
            if not url:
                continue

            data = measure_rpc(rpc_id, url)
            data["timestamp"] = timestamp
            results.append(data)

        if results:
            df = pd.DataFrame(results)

            max_block = df["block_number"].max()
            df["block_lag"] = max_block - df["block_number"]

            df = df[[
                "timestamp",
                "rpc_id",
                "latency_ms",
                "failure_flag",
                "block_number",
                "block_lag"
            ]]

            df.to_sql("rpc_metrics", engine, if_exists="append", index=False)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()