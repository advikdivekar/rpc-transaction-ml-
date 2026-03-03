import time
import os
import random
import pandas as pd
from datetime import datetime
from web3 import Web3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Setup
RPC_LIST = [
    os.getenv("INFURA_RPC_URL"),
    "https://ethereum-sepolia.publicnode.com",
    "https://sepolia.drpc.org"
]
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
MY_ADDRESS = os.getenv("MY_ADDRESS")
engine = create_engine(os.getenv("DATABASE_URL"))

def send_tx():
    # 1. Pick a random RPC
    rpc_url = random.choice([url for url in RPC_LIST if url])
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    print(f"Attempting TX via {rpc_url}...")
    
    try:
        # 2. Build TX
        nonce = w3.eth.get_transaction_count(MY_ADDRESS)
        tx = {
            'nonce': nonce,
            'to': MY_ADDRESS,
            'value': w3.to_wei(0.00001, 'ether'),
            'gas': 21000,
            'gasPrice': w3.eth.gas_price,
            'chainId': 11155111
        }
        
        # 3. Sign & Send
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        start_time = time.time()
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        # 4. Wait for Receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        duration = time.time() - start_time
        
        print(f"✅ Confirmed in {duration:.2f}s")
        return {
            "timestamp": datetime.utcnow(),
            "rpc_url": rpc_url,
            "tx_hash": tx_hash.hex(),
            "duration_sec": duration,
            "status": 1
        }

    except Exception as e:
        print(f"❌ Failed: {e}")
        return {
            "timestamp": datetime.utcnow(),
            "rpc_url": rpc_url,
            "tx_hash": "FAILED",
            "duration_sec": 120, # Penalty
            "status": 0
        }

def main():
    # Create Table
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tx_outcomes (
                timestamp TIMESTAMP, rpc_url TEXT, tx_hash TEXT, 
                duration_sec FLOAT, status INT
            );
        """))
        conn.commit()

    # Run Loop (Send 200 TXs)
    for i in range(200):
        data = send_tx()
        
        # Save to DB
        df = pd.DataFrame([data])
        df.to_sql("tx_outcomes", engine, if_exists="append", index=False)
        
        print("Waiting 3s...")
        time.sleep(3)

if __name__ == "__main__":
    main()