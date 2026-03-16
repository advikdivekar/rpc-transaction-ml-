import sys
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Path Fix
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.database import db

def train_classifier():
    print("[INFO] Initializing State-Aware Classification pipeline...")
    
    # 1. Fetch Data
    tx_query = "SELECT timestamp, rpc_url, duration_sec FROM tx_outcomes WHERE status = 1"
    rpc_query = "SELECT timestamp, rpc_id, latency_ms, block_number FROM rpc_metrics" # Pulled block_number instead of lag
    
    tx_df = db.load_data(tx_query)
    rpc_df = db.load_data(rpc_query)
    
    if tx_df.empty or rpc_df.empty:
        print("[ERROR] Insufficient data in database.")
        return

    # 2. Preprocess & Merge
    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])
    rpc_df['timestamp'] = pd.to_datetime(rpc_df['timestamp'])
    
    # --- CALCULATE REAL BLOCK LAG ---
    # Group by 15-second windows to find the "Global Max Block" at that moment
    rpc_df['time_window'] = rpc_df['timestamp'].dt.floor('15s')
    rpc_df['global_max_block'] = rpc_df.groupby('time_window')['block_number'].transform('max')
    rpc_df['block_lag'] = rpc_df['global_max_block'] - rpc_df['block_number']
    rpc_df['block_lag'] = rpc_df['block_lag'].clip(lower=0) # Prevent negative anomalies
    
    tx_df = tx_df.sort_values('timestamp')
    rpc_df = rpc_df.sort_values('timestamp')
    
    url_to_id = {
        "https://sepolia.infura.io/v3/f6dccf73ccd64c06a5e7734325927bb9": "infura",
        "https://sepolia.drpc.org": "drpc",
        "https://ethereum-sepolia-rpc.publicnode.com": "publicnode",
        "https://eth-sepolia.g.alchemy.com/v2/TBURRsb3KoDLo1oJXSyBj": "alchemy"
    }
    tx_df['rpc_id'] = tx_df['rpc_url'].map(lambda x: url_to_id.get(x, "unknown"))
    
    merged_df = pd.merge_asof(
        tx_df, rpc_df, on='timestamp', by='rpc_id', 
        direction='backward', tolerance=pd.Timedelta('2 minutes')
    ).dropna()

    # --- FEATURE ENGINEERING ---
    merged_df['hour_of_day'] = merged_df['timestamp'].dt.hour
    merged_df['day_of_week'] = merged_df['timestamp'].dt.dayofweek
    
    merged_df = merged_df.sort_values(['rpc_id', 'timestamp'])
    merged_df['rolling_latency_5'] = merged_df.groupby('rpc_id')['latency_ms'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    merged_df = merged_df.sort_values('timestamp')
    
    # --- BINARY TARGET CREATION (Shifted to 20 seconds for true tail risk) ---
    merged_df['is_slow'] = (merged_df['duration_sec'] > 20.0).astype(int)

    print(f"\n[INFO] Dataset Size: {len(merged_df)} matched samples")
    print(f"[INFO] Severe Tail-Risk Base Rate (>20s): {merged_df['is_slow'].mean()*100:.1f}%")

    # 3. Train the Model (CLASSIFIER)
    features = ['latency_ms', 'block_lag', 'hour_of_day', 'day_of_week', 'rolling_latency_5']
    X = merged_df[features]
    y = merged_df['is_slow']

    split_idx = int(len(merged_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Calculate weight to balance the strict 20s threshold
    num_fast = (y_train == 0).sum()
    num_slow = (y_train == 1).sum()
    scale_weight = num_fast / num_slow if num_slow > 0 else 1.0

    model = XGBClassifier(
        n_estimators=150, 
        max_depth=5, 
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- CLASSIFICATION PERFORMANCE (Temporal Split) ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}") # Ensure this outputs 0.532 to match your new text!
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fast (<20s)", "Slow (>20s)"]))

    # --- NEW: PRINT FEATURE IMPORTANCES (For Fix 3) ---
    print("\n--- FEATURE IMPORTANCES ---")
    importances = model.feature_importances_
    # Sort them descending so you see the top drivers first
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"{features[idx]}: {importances[idx]:.3f}")

    # 5. Save Model
    model_path = os.path.join(PROJECT_ROOT, "rpc_model.json")
    model.save_model(model_path)
    print(f"\n[SUCCESS] AI Classifier saved to {model_path}")

if __name__ == "__main__":
    train_classifier()