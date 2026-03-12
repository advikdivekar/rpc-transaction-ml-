import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from config.settings import RPC_PROVIDERS, MODEL_PATH
import os

class RPCLatencyPredictor:
    """
    Implements the XGBoost regression pipeline for predicting transaction confirmation latency.
    This class handles data preprocessing, feature engineering, model training, and inference.
    """
    
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.features = ['latency_ms', 'block_lag']
        self.target = 'duration_sec'

    def _clean_rpc_name(self, url: str) -> str:
        """
        Normalizes provider URLs to standard identifiers.
        Hardcoded to ensure historical data matches even if settings change.
        """
        url = str(url).lower()
        if "infura" in url: return "infura"
        if "alchemy" in url: return "alchemy"
        if "drpc" in url: return "drpc"
        if "publicnode" in url: return "publicnode"
        return "unknown"

    def train(self, tx_df: pd.DataFrame, rpc_df: pd.DataFrame):
        print("[INFO] Preprocessing data for training...")
        print(f"[DEBUG] Starting rows -> TX: {len(tx_df)}, RPC: {len(rpc_df)}")
        
        # PEEK AT THE RAW DATA to see what Excel did
        print(f"[DEBUG] Raw TX Timestamp Example: {tx_df['timestamp'].iloc[0]} (Type: {type(tx_df['timestamp'].iloc[0])})")
        
        # 1. Force numeric types safely
        rpc_df['latency_ms'] = pd.to_numeric(rpc_df['latency_ms'], errors='coerce')
        rpc_df['block_lag'] = pd.to_numeric(rpc_df['block_lag'], errors='coerce')
        tx_df['duration_sec'] = pd.to_numeric(tx_df['duration_sec'], errors='coerce')
        tx_df['status'] = pd.to_numeric(tx_df['status'], errors='coerce')
        
        # 2. Clean RPC names
        tx_df['rpc_id'] = tx_df['rpc_url'].apply(self._clean_rpc_name)
        
        # 3. THE ULTIMATE DATETIME FIX:
        # Convert to string first (in case Excel made them floats/serial numbers)
        tx_df['timestamp'] = tx_df['timestamp'].astype(str)
        rpc_df['timestamp'] = rpc_df['timestamp'].astype(str)
        
        # Parse using format='mixed' to handle almost any date string, and force to UTC
        tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], format='mixed', errors='coerce', utc=True)
        rpc_df['timestamp'] = pd.to_datetime(rpc_df['timestamp'], format='mixed', errors='coerce', utc=True)
        
        # Drop rows with corrupted dates and sort
        tx_df = tx_df.dropna(subset=['timestamp']).sort_values("timestamp")
        rpc_df = rpc_df.dropna(subset=['timestamp']).sort_values("timestamp")
        
        print(f"[DEBUG] After date parsing -> TX: {len(tx_df)}, RPC: {len(rpc_df)}")

        # 4. Merge Data
        print("[INFO] Aligning transaction outcomes with network state...")
        df = pd.merge_asof(
            tx_df, rpc_df, 
            on="timestamp", 
            by="rpc_id", 
            direction="backward",
            tolerance=pd.Timedelta("15m") # Added 15m tolerance so it only matches relevant times
        )
        
        print(f"[DEBUG] Rows after Merge -> {len(df)}")
        
        # 5. Final Filtering
        df = df.dropna(subset=self.features + [self.target])
        print(f"[DEBUG] Rows after dropping NaNs -> {len(df)}")
        
        df = df[df['status'] == 1]
        print(f"[DEBUG] Rows after status==1 filter -> {len(df)}")
        
        print(f"[INFO] Final Training dataset size: {len(df)} matched samples.")
        
        if len(df) == 0:
            print("[ERROR] 0 matched samples remaining.")
            return
            
        # --- Model Fitting ---
        X = df[self.features]
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # --- Evaluation ---
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"[RESULT] Model Mean Absolute Error (MAE): +/- {mae:.4f} seconds")
        self.model.save_model(MODEL_PATH)
        print(f"[SUCCESS] Model saved to {MODEL_PATH}")