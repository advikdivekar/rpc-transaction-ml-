import time
import requests
import pandas as pd
import xgboost as xgb
import json
from web3 import Web3
from dotenv import load_dotenv
import os

# 1. Setup
load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
MY_ADDRESS = os.getenv("MY_ADDRESS")

# Define our RPCs (The "Candidates")
RPC_LIST = {
    "infura": "https://sepolia.infura.io/v3/f6dccf73ccd64c06a5e7734325927bb9",
    "alchemy": "https://eth-sepolia.g.alchemy.com/v2/demo",
    "drpc": "https://sepolia.drpc.org",
    "publicnode": "https://ethereum-sepolia-rpc.publicnode.com"
}

def get_realtime_metrics():
    """Pings all RPCs to get current conditions (Features)"""
    print("📡 Pinging networks for live data...")
    live_data = []
    
    for name, url in RPC_LIST.items():
        start = time.time()
        try:
            # We fetch block number to calculate "Block Lag"
            response = requests.post(url, json={"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}, timeout=3)
            latency = (time.time() - start) * 1000 # Convert to ms
            
            # Parse Block Number
            result = response.json().get('result')
            block_num = int(result, 16) if result else 0
            
            live_data.append({
                "rpc_id": name,
                "url": url,
                "latency_ms": latency,
                "block_number": block_num
            })
        except:
            print(f"⚠️ {name} is down!")
            continue
            
    # Calculate Lag (Distance from the highest block found)
    df = pd.DataFrame(live_data)
    highest_block = df['block_number'].max()
    df['block_lag'] = highest_block - df['block_number']
    
    return df

def predict_fastest(df):
    """Uses the AI model to predict duration"""
    print("\n🧠 AI Analyzing current conditions...")
    
    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("rpc_model.json")
    
    # Prepare inputs (Must match training order: latency_ms, block_lag)
    X_live = df[['latency_ms', 'block_lag']]
    
    # Predict!
    predictions = model.predict(X_live)
    df['predicted_duration'] = predictions
    
    # Sort by fastest (lowest duration)
    df_sorted = df.sort_values("predicted_duration")
    
    return df_sorted

def send_transaction(best_rpc_url):
    """Sends a real transaction using the WINNER"""
    print(f"\n🚀 Routing Transaction via WINNER: {best_rpc_url}...")
    
    w3 = Web3(Web3.HTTPProvider(best_rpc_url))
    
    if not w3.is_connected():
        print("❌ Failed to connect to winner.")
        return

    nonce = w3.eth.get_transaction_count(MY_ADDRESS)
    
    tx = {
        'nonce': nonce,
        'to': MY_ADDRESS,
        'value': w3.to_wei(0.00001, 'ether'),
        'gas': 21000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'chainId': 11155111
    }
    
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    
    start_time = time.time()
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"✅ Sent! Hash: {tx_hash.hex()}")
    
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    duration = time.time() - start_time
    print(f"🎉 Confirmed in {duration:.2f} seconds")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Get Live Data
    live_df = get_realtime_metrics()
    
    # 2. AI Prediction
    results = predict_fastest(live_df)
    
    print("\n🏆 AI LEADERBOARD (Who is fastest right now?)")
    print(results[['rpc_id', 'latency_ms', 'block_lag', 'predicted_duration']].to_string(index=False))
    
    # 3. Select Winner
    winner = results.iloc[0]
    print(f"\n🥇 The AI chose: {winner['rpc_id'].upper()}")
    print(f"   Reason: Latency {winner['latency_ms']:.0f}ms | Lag {winner['block_lag']}")
    
    # 4. Execute
    send_transaction(winner['url'])