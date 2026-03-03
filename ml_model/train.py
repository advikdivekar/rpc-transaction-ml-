import os
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

def train():
    print("Loading data from Neon DB...")
    
    # 1. Load Data
    metrics_df = pd.read_sql("SELECT * FROM rpc_metrics", engine)
    tx_df = pd.read_sql("SELECT * FROM tx_outcomes", engine)
    
    if tx_df.empty:
        print("❌ No transactions found! Run sender.py first.")
        return

    # 2. Merge Data (As-Of Merge)
    # We want the network metrics *closest* to when the TX happened
    metrics_df = metrics_df.sort_values("timestamp")
    tx_df = tx_df.sort_values("timestamp")
    
    df = pd.merge_asof(
        tx_df, metrics_df, 
        on="timestamp", 
        by_left="rpc_url", by_right="rpc_id", # Match RPCs
        direction="backward"
    )
    
    # 3. Prepare Features
    features = ['latency_ms', 'block_lag']
    target = 'duration_sec'
    
    df = df.dropna(subset=features)
    X = df[features]
    y = df[target]
    
    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = xgb.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    # 5. Save Model
    model.save_model("rpc_model.json")
    print("✅ Model trained and saved as rpc_model.json")
    
    # Feature Importance (For your paper)
    print("Feature Importance:", model.feature_importances_)

if __name__ == "__main__":
    train()