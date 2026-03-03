import time
import requests
import threading
from config.settings import RPC_PROVIDERS
from src.database import db

class NetworkHealthMonitor:
    def __init__(self, interval=30):
        self.interval = interval
        self.running = False

    def fetch_metrics(self):
        """Pings all RPCs and saves health data to DB."""
        print(f"[MONITOR] Pinging {len(RPC_PROVIDERS)} networks...")
        
        for name, url in RPC_PROVIDERS.items():
            start = time.time()
            try:
                # 1. Measure Latency
                payload = {"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}
                response = requests.post(url, json=payload, timeout=5)
                latency = (time.time() - start) * 1000
                
                # 2. Measure Block Height
                result = response.json().get('result')
                block_num = int(result, 16) if result else 0
                
                # 3. Save to DB
                db.save_metric("rpc_metrics", {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "rpc_id": name,
                    "latency_ms": latency,
                    "block_number": block_num,
                    "block_lag": 0 # Calculated later during analysis
                })
            except Exception as e:
                print(f"[MONITOR] Error pinging {name}: {e}")

    def start(self):
        """Runs the monitor loop in a background thread."""
        self.running = True
        while self.running:
            self.fetch_metrics()
            time.sleep(self.interval)

    def stop(self):
        self.running = False