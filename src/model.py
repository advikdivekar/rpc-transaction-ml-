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
        # Initialize the XGBoost Regressor with squared error objective.
        # n_estimators=100 provides a balance between underfitting and overfitting.
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        
        # input features:
        # 1. latency_ms: Network round-trip time.
        # 2. block_lag: Sync status relative to the network head.
        self.features = ['latency_ms', 'block_lag']
        
        # target variable: Actual confirmation duration in seconds.
        self.target = 'duration_sec'

    def _clean_rpc_name(self, url: str) -> str:
        """
        Normalizes provider URLs to standard identifiers (e.g., 'infura').
        This ensures consistency between the metrics table and the transaction table.
        """
        url = str(url).lower()
        for name in RPC_PROVIDERS.keys():
            if name in url:
                return name
        return "unknown"

    def train(self, tx_df: pd.DataFrame, rpc_df: pd.DataFrame):
        """
        Executes the training pipeline.
        
        Args:
            tx_df: Historical transaction outcomes (ground truth).
            rpc_df: Historical network health metrics (features).
        """
        print("[INFO] Preprocessing data for training...")
        
        # --- Data Standardization ---
        # Map full URLs to short identifiers for merging.
        tx_df['rpc_id'] = tx_df['rpc_url'].apply(self._clean_rpc_name)
        
        # Convert timestamps to UTC to ensure accurate temporal alignment.
        tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], utc=True)
        rpc_df['timestamp'] = pd.to_datetime(rpc_df['timestamp'], utc=True)
        
        # Sort data by timestamp (required for merge_asof).
        tx_df = tx_df.sort_values("timestamp")
        rpc_df = rpc_df.sort_values("timestamp")

        # --- Temporal Alignment ---
        # Merge transaction data with the most recent RPC metric recorded prior to the transaction.
        # tolerance='10m' discards metrics older than 10 minutes to prevent using stale data.
        df = pd.merge_asof(
            tx_df, rpc_df, 
            on="timestamp", 
            by="rpc_id", 
            direction="backward", 
            tolerance=pd.Timedelta("10m")
        )
        
        # --- Filtering ---
        # Remove rows with missing features or failed transactions.
        df = df.dropna(subset=self.features)
        df = df[df['status'] == 1]
        
        print(f"[INFO] Training dataset size: {len(df)} matched samples.")
        
        # --- Model Fitting ---
        X = df[self.features]
        y = df[self.target]
        
        # Split dataset: 80% Training, 20% Testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # --- Evaluation ---
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"[RESULT] Model Mean Absolute Error (MAE): +/- {mae:.4f} seconds")
        
        # Serialize model to disk.
        self.model.save_model(MODEL_PATH)
        print(f"[SUCCESS] Model saved to {MODEL_PATH}")

    def predict(self, live_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Performs inference on real-time data.
        
        Args:
            live_metrics: A DataFrame containing current network status.
            
        Returns:
            DataFrame with an appended 'predicted_duration' column.
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file not found. Please run the training script first.")
            
        self.model.load_model(MODEL_PATH)
        
        # Generate predictions using the trained features.
        predictions = self.model.predict(live_metrics[self.features])
        
        live_metrics['predicted_duration'] = predictions
        return live_metrics.sort_values("predicted_duration")