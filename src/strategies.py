import pandas as pd
from src.model import RPCLatencyPredictor

class StaticStrategy:
    """The baseline Web2 approach: Hardcode a single provider."""
    def __init__(self, default_provider="infura"):
        self.default_provider = default_provider

    def get_best_rpc(self, live_df: pd.DataFrame) -> str:
        return self.default_provider

class PingBasedStrategy:
    """The standard Web3 Load Balancer: Pick the lowest latency (Ping)."""
    def get_best_rpc(self, live_df: pd.DataFrame) -> str:
        if live_df.empty:
            return "infura" # Fallback
        # Sort strictly by the lowest ping
        best = live_df.sort_values(by="latency_ms", ascending=True).iloc[0]
        return best["rpc_id"]

class SmartRouterStrategy:
    """Your Research Model: XGBoost prediction using Latency + Block Lag."""
    def __init__(self):
        self.predictor = RPCLatencyPredictor()

    def get_best_rpc(self, live_df: pd.DataFrame) -> str:
        if live_df.empty:
            return "infura" # Fallback
        
        # Ask the AI for the lowest predicted confirmation time
        predictions = self.predictor.predict(live_df)
        best = predictions.sort_values(by="predicted_duration", ascending=True).iloc[0]
        return best["rpc_id"]
        