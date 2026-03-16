import os
import pandas as pd
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class RPCLatencyPredictor:
    def __init__(self, model_path=os.path.join(PROJECT_ROOT, "rpc_model.json")):
        # Note: Changed to XGBClassifier to match the new model
        self.model = XGBClassifier()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
        else:
            print(f"[WARN] Model not found at {model_path}. Please train first.")

    def predict(self, live_df: pd.DataFrame) -> pd.DataFrame:
        df = live_df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        if 'rolling_latency_5' not in df.columns:
             df['rolling_latency_5'] = df['latency_ms'] 

        features = ['latency_ms', 'block_lag', 'hour_of_day', 'day_of_week', 'rolling_latency_5']
        
        # predict_proba returns the probability of the transaction being slow
        probabilities = self.model.predict_proba(df[features])[:, 1]
        df['prob_slow'] = probabilities
        
        return df