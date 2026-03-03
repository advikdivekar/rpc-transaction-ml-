import sys
import os

# Append project root to system path to allow module imports.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import db
from src.model import RPCLatencyPredictor

def main():
    print("[INFO] Initializing training pipeline...")
    
    # 1. Load Data
    print("[INFO] Fetching historical data from database...")
    tx_df = db.load_data("SELECT * FROM tx_outcomes")
    rpc_df = db.load_data("SELECT * FROM rpc_metrics")
    
    # 2. Train Model
    predictor = RPCLatencyPredictor()
    predictor.train(tx_df, rpc_df)
    
    print("[SUCCESS] Training pipeline completed successfully.")

if __name__ == "__main__":
    main()