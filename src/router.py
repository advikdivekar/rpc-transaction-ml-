import time
import requests
import pandas as pd
from web3 import Web3
from config.settings import RPC_PROVIDERS, PRIVATE_KEY, MY_ADDRESS, CHAIN_ID
from src.model import RPCLatencyPredictor

class SmartRouter:
    """
    Orchestrates the selection and routing of transactions.
    Queries network status, invokes the predictive model, and executes transactions.
    """
    
    def __init__(self):
        self.predictor = RPCLatencyPredictor()

    def get_realtime_metrics(self) -> pd.DataFrame:
        """
        Probes all configured RPC endpoints to gather latency and block height data.
        """
        print("[INFO] Pinging RPC networks for real-time metrics...")
        live_data = []
        
        for name, url in RPC_PROVIDERS.items():
            start = time.time()
            try:
                # Perform a lightweight JSON-RPC call (eth_blockNumber).
                response = requests.post(
                    url, 
                    json={"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}, 
                    timeout=3
                )
                
                # Calculate latency (ms).
                latency = (time.time() - start) * 1000 
                
                # Parse block number.
                result = response.json().get('result')
                block_num = int(result, 16) if result else 0
                
                live_data.append({
                    "rpc_id": name,
                    "url": url,
                    "latency_ms": latency,
                    "block_number": block_num
                })
            except Exception as e:
                print(f"[WARNING] {name} is unreachable. Skipping.")
                continue
                
        df = pd.DataFrame(live_data)
        
        # Calculate Block Lag relative to the most advanced node.
        if not df.empty:
            df['block_lag'] = df['block_number'].max() - df['block_number']
            
        return df

    def find_best_route(self):
        """
        Determines the optimal RPC provider based on predictive analysis.
        """
        # 1. Acquire current network state.
        live_df = self.get_realtime_metrics()
        
        if live_df.empty:
            print("[ERROR] No RPC endpoints are currently available.")
            return None
            
        # 2. Predict confirmation times.
        results = self.predictor.predict(live_df)
        
        return results

    def send_transaction(self, best_rpc_url):
        """
        Submits a transaction via the selected optimal provider.
        """
        print(f"[ACTION] Routing transaction via optimal provider: {best_rpc_url}...")
        
        w3 = Web3(Web3.HTTPProvider(best_rpc_url))
        
        if not w3.is_connected():
            print("[ERROR] Failed to establish connection with the selected provider.")
            return

        # Construct the transaction object.
        nonce = w3.eth.get_transaction_count(MY_ADDRESS)
        tx = {
            'nonce': nonce,
            'to': MY_ADDRESS, # Self-transfer for testing
            'value': w3.to_wei(0.00001, 'ether'),
            'gas': 21000,
            'gasPrice': w3.to_wei('10', 'gwei'),
            'chainId': CHAIN_ID
        }
        
        # Sign transaction locally.
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        
        # Broadcast transaction.
        start_time = time.time()
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"[INFO] Transaction broadcasted. Hash: {tx_hash.hex()}")
        
        # Wait for on-chain confirmation.
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        duration = time.time() - start_time
        
        print(f"[SUCCESS] Transaction confirmed in {duration:.2f} seconds")