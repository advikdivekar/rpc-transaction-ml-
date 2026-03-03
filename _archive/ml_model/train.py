import os
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# 1. Setup
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def clean_rpc_name(url):
    """Converts full URL to short ID to match rpc_metrics table"""
    url = str(url).lower()
    if "infura" in url: return "infura"
    if "alchemy" in url: return "alchemy"
    if "drpc" in url: return "drpc"
    if "publicnode" in url: return "publicnode"
    return "unknown"

def train_model():
    print("⏳ Connecting to Neon DB...")
    
    # 2. Load Data
    try:
        rpc_df = pd.read_sql("SELECT * FROM rpc_metrics", engine)
        tx_df = pd.read_sql("SELECT * FROM tx_outcomes", engine)
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    print(f"✅ Loaded Data: {len(tx_df)} Transactions, {len(rpc_df)} RPC Logs")

    # 3. Cleaning & Normalization (THE FIX)
    # Convert timestamps to UTC to avoid timezone mismatches
    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], utc=True)
    rpc_df['timestamp'] = pd.to_datetime(rpc_df['timestamp'], utc=True)
    
    # FIX: Map full URLs in tx_df to short IDs in rpc_df
    tx_df['rpc_id'] = tx_df['rpc_url'].apply(clean_rpc_name)
    
    # Sort for merge
    tx_df = tx_df.sort_values("timestamp")
    rpc_df = rpc_df.sort_values("timestamp")

    # 4. Merging
    print("⏳ Merging datasets...")
    
    # Now we merge on 'rpc_id' which exists in both!
    df = pd.merge_asof(
        tx_df, 
        rpc_df, 
        on="timestamp", 
        by="rpc_id", 
        direction="backward",
        tolerance=pd.Timedelta("10m") # Look back up to 10 mins
    )

    # Remove rows where we couldn't find a match or TX failed
    df = df.dropna(subset=['latency_ms', 'block_lag'])
    df = df[df['status'] == 1]
    
    print(f"✅ Training Data Ready: {len(df)} matched samples")

    if len(df) < 10:
        print("⚠️ Still 0 matches? Check your timestamps!")
        print("Sample TX Time:", tx_df['timestamp'].iloc[0])
        print("Sample RPC Time:", rpc_df['timestamp'].iloc[0])
        return

    # 5. Train Model
    features = ['latency_ms', 'block_lag']
    target = 'duration_sec'

    X = df[features]
    y = df[target]

    print("🧠 Training XGBoost Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # 6. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"🎯 Model Accuracy (MAE): +/- {mae:.2f} seconds")
    
    # 7. Save
    model.save_model("rpc_model.json")
    print("💾 Model saved to 'rpc_model.json'")
    
    # 8. Feature Importance
    print("\n📊 Feature Importance:")
    print(pd.Series(model.feature_importances_, index=features))

if __name__ == "__main__":
    train_model()