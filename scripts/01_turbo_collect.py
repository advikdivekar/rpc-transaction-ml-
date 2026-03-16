import sys
import os
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from web3 import Web3
from dotenv import load_dotenv

# 1. PATH FIX & LOAD ENV
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
from src.database import db

# 2. EXTRACT KEYS
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
MY_ADDRESS = os.getenv("MY_ADDRESS")
CHAIN_ID = 11155111

if not PRIVATE_KEY or not MY_ADDRESS:
    print("[ERROR] PRIVATE_KEY or MY_ADDRESS missing in .env")
    sys.exit(1)

SAFE_ADDRESS = Web3.to_checksum_address(MY_ADDRESS)

# 3. HARDCODE URLS (Bypasses settings.py completely for stability)
RPC_PROVIDERS = {
    "alchemy": os.getenv("ALCHEMY_RPC_URL", "https://eth-sepolia.g.alchemy.com/v2/TBURRsb3KoDLo1oJXSyBj"),
    "publicnode": "https://ethereum-sepolia-rpc.publicnode.com",
    "drpc": "https://sepolia.drpc.org",
    "infura": "https://sepolia.infura.io/v3/f6dccf73ccd64c06a5e7734325927bb9",
}

# 4. DYNAMIC MIDDLEWARE
def inject_poa_middleware(w3):
    try:
        from web3.middleware import ExtraDataToExternalDataMiddleware
        w3.middleware_onion.inject(ExtraDataToExternalDataMiddleware, layer=0)
        return
    except ImportError: pass
    try:
        from web3.middleware.geth_poa_middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        return
    except ImportError: pass

# 5. INLINE NETWORK MONITOR (Bypasses src/monitor.py)
class StandaloneMonitor:
    def __init__(self, interval=10):
        self.interval = interval
        self.running = False

    def fetch_metrics(self):
        for name, url in RPC_PROVIDERS.items():
            start = time.time()
            try:
                response = requests.post(url, json={"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}, timeout=5)
                latency = (time.time() - start) * 1000
                result = response.json().get('result')
                block_num = int(result, 16) if result else 0
                
                db.save_metric("rpc_metrics", {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                    "rpc_id": name,
                    "latency_ms": latency,
                    "block_number": block_num,
                    "block_lag": 0
                })
            except Exception as e:
                pass # Fail silently to keep console clean

    def start(self):
        self.running = True
        while self.running:
            self.fetch_metrics()
            time.sleep(self.interval)

# 6. MASTER HARVESTER LOGIC
print("[INFO] Initializing Master Harvester...")
primary_w3 = Web3(Web3.HTTPProvider(RPC_PROVIDERS["alchemy"]))
inject_poa_middleware(primary_w3)

class NonceManager:
    def __init__(self, address):
        self.nonce = primary_w3.eth.get_transaction_count(address)
        self.lock = threading.Lock()
    def get_next(self):
        with self.lock:
            current = self.nonce
            self.nonce += 1
            return current

nonce_manager = NonceManager(SAFE_ADDRESS)

def send_and_wait(rpc_name, rpc_url):
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        inject_poa_middleware(w3)
        
        start_time = time.time()
        nonce = nonce_manager.get_next()
        
        tx = {
            'nonce': nonce, 'to': SAFE_ADDRESS, 'value': 0, 'gas': 21000,
            'maxFeePerGas': w3.to_wei(20, 'gwei'),
            'maxPriorityFeePerGas': w3.to_wei(2, 'gwei'),
            'chainId': CHAIN_ID
        }
        
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
        duration = time.time() - start_time
        
        db.save_metric("tx_outcomes", {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)),
            "rpc_url": rpc_url, "tx_hash": tx_hash.hex(),
            "duration_sec": duration, "status": receipt.status
        })
        print(f"[TX SUCCESS] {rpc_name.upper()} | Time: {duration:.2f}s | Nonce: {nonce}")
        
    except Exception as e:
        err_msg = str(e)
        if "insufficient funds" in err_msg.lower():
            print(f"[CRITICAL] {rpc_name.upper()}: Out of Sepolia ETH!")
        else:
            print(f"[TX ERROR] {rpc_name.upper()}: {err_msg[:60]}")

def main():
    print(f"[INFO] Collection Address: {SAFE_ADDRESS}")
    print(f"[INFO] Initial Nonce: {nonce_manager.nonce}")

    # Background Monitor
    monitor = StandaloneMonitor(interval=7)
    threading.Thread(target=monitor.start, daemon=True).start()
    
    # Foreground Transaction Sender
    with ThreadPoolExecutor(max_workers=10) as executor:
        try:
            while True:
                for name, url in RPC_PROVIDERS.items():
                    executor.submit(send_and_wait, name, url)
                time.sleep(4) 
        except KeyboardInterrupt:
            print("\n[INFO] Harvester safely stopped.")

if __name__ == "__main__":
    main()