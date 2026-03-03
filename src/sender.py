import time
import random
from web3 import Web3
from config.settings import RPC_PROVIDERS, PRIVATE_KEY, MY_ADDRESS, CHAIN_ID
from src.database import db

class TransactionProber:
    def __init__(self, max_tx=100):
        self.max_tx = max_tx

    def send_probe(self):
        """Sends 1 transaction to a random RPC and records duration."""
        # 1. Pick a random network
        rpc_name = random.choice(list(RPC_PROVIDERS.keys()))
        rpc_url = RPC_PROVIDERS[rpc_name]
        
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return

        try:
            # 2. Build Transaction
            nonce = w3.eth.get_transaction_count(MY_ADDRESS)
            tx = {
                'nonce': nonce,
                'to': MY_ADDRESS,
                'value': w3.to_wei(0.00001, 'ether'),
                'gas': 21000,
                'gasPrice': w3.to_wei('50', 'gwei'), # High gas to ensure it's not stuck
                'chainId': CHAIN_ID
            }
            
            # 3. Send & Time it
            signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
            start_time = time.time()
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            print(f"[SENDER] Sent via {rpc_name.upper()} | Hash: {tx_hash.hex()[:10]}...")
            
            # 4. Wait for receipt
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            duration = time.time() - start_time
            
            # 5. Save to DB
            if receipt.status == 1:
                print(f"[SENDER] ✅ Confirmed in {duration:.2f}s")
                db.save_metric("tx_outcomes", {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "rpc_url": rpc_url,
                    "tx_hash": tx_hash.hex(),
                    "duration_sec": duration,
                    "status": 1
                })
            else:
                print("[SENDER] ❌ Failed (Reverted)")

        except Exception as e:
            print(f"[SENDER] Error: {e}")

    def run_batch(self):
        """Runs the probing loop."""
        for i in range(self.max_tx):
            print(f"\n--- TX #{i+1}/{self.max_tx} ---")
            self.send_probe()
            
            # Random sleep to capture different network conditions
            sleep_time = random.randint(5, 15)
            print(f"Sleeping {sleep_time}s...")
            time.sleep(sleep_time)