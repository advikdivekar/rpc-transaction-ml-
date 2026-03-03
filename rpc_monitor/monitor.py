import time
import os
import pandas as pd
from datetime import datetime
from web3 import Web3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Define RPCs to monitor
RPC_ENDPOINTS = {
    "infura": os.getenv("INFURA_RPC_URL"),
    "alchemy": os.getenv("ALCHEMY_RPC_URL"),
    "publicnode": "https://ethereum-sepolia.publicnode.com",
    "drpc": "https://sepolia.drpc.org"
}

# Connect to Neon DB
engine = create_engine(os.getenv("DATABASE_URL"), pool_pre_ping=True)

def measure_rpc(rpc_id, url):
    try:
        w3 = Web3(Web3.HTTPProvider(url, request_kwargs={'timeout': 5}))
        start = time.time()
        
        if w3.is_connected():
            block = w3.eth.block_number
            latency = (time.time() - start) * 1000
            return {"rpc_id": rpc_id, "latency_ms": latency, "block_number": block, "failure": 0}
            
    except Exception:
        return {"rpc_id": rpc_id, "latency_ms": 5000, "block_number": 0, "failure": 1}

def main():
    print("Starting RPC Monitor...")
    
    # Create Table if not exists
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rpc_metrics (
                timestamp TIMESTAMP, rpc_id TEXT, latency_ms FLOAT, 
                block_number BIGINT, block_lag BIGINT, failure INT
            );
        """))
        conn.commit()

    while True:
        results = []
        timestamp = datetime.utcnow()

        # 1. Measure all RPCs
        for rpc_id, url in RPC_ENDPOINTS.items():
            if url:
                data = measure_rpc(rpc_id, url)
                data['timestamp'] = timestamp
                results.append(data)

        if results:
            df = pd.DataFrame(results)
            # Calculate Block Lag (Difference from the best block seen)
            df['block_lag'] = df['block_number'].max() - df['block_number']
            df.loc[df['failure'] == 1, 'block_lag'] = 100 # Penalty for failure
            
            # Save to DB
            try:
                df.to_sql("rpc_metrics", engine, if_exists="append", index=False)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved {len(df)} rows.")
            except Exception as e:
                print(f"DB Error: {e}")

        time.sleep(30)

if __name__ == "__main__":
    main()