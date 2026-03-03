import sys
import os

# Append project root to system path to allow module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.router import SmartRouter

def main():
    router = SmartRouter()
    
    # 1. Evaluate Network and Select Provider
    results = router.find_best_route()
    
    if results is not None:
        print("\n[INFO] Predictive Leaderboard (Lowest Latency First):")
        print(results[['rpc_id', 'latency_ms', 'predicted_duration']].to_string(index=False))
        
        winner = results.iloc[0]
        print(f"\n[DECISION] Optimal Provider Selected: {winner['rpc_id'].upper()}")
        print(f"           Metrics: Latency {winner['latency_ms']:.0f}ms | Lag {winner['block_lag']}")
        
        # 2. Execute Transaction
        router.send_transaction(winner['url'])

if __name__ == "__main__":
    main()