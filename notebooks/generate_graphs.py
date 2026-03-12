import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CRITICAL FIX: Use insert(0, ...) so it finds your config folder first
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.database import db
from src.model import RPCLatencyPredictor

# --- CONFIGURATION FOR ACADEMIC STYLE ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6)
})

def clean_rpc_name(url):
    url = str(url).lower()
    if "infura" in url: return "infura"
    if "alchemy" in url: return "alchemy"
    if "drpc" in url: return "drpc"
    if "publicnode" in url: return "publicnode"
    return "unknown"

def load_and_merge_data():
    print("[INFO] Fetching historical data from database...")
    tx_df = db.load_data("SELECT * FROM tx_outcomes")
    rpc_df = db.load_data("SELECT * FROM rpc_metrics")
    
    print("[INFO] Preprocessing data for graphing...")
    # Force Numeric
    rpc_df['latency_ms'] = pd.to_numeric(rpc_df['latency_ms'], errors='coerce')
    tx_df['duration_sec'] = pd.to_numeric(tx_df['duration_sec'], errors='coerce')
    tx_df['status'] = pd.to_numeric(tx_df['status'], errors='coerce')

    # Clean Names
    tx_df['rpc_id'] = tx_df['rpc_url'].apply(clean_rpc_name)
    rpc_df['rpc_id'] = rpc_df['rpc_id'].astype(str).str.strip().str.lower()
    
    # Robust Date Parsing (Same as the training script)
    tx_df['timestamp'] = tx_df['timestamp'].astype(str)
    rpc_df['timestamp'] = rpc_df['timestamp'].astype(str)
    
    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], format='mixed', errors='coerce').dt.tz_localize(None)
    rpc_df['timestamp'] = pd.to_datetime(rpc_df['timestamp'], format='mixed', errors='coerce').dt.tz_localize(None)
    
    tx_df = tx_df.dropna(subset=['timestamp']).sort_values("timestamp")
    rpc_df = rpc_df.dropna(subset=['timestamp']).sort_values("timestamp")
    
    print("[INFO] Aligning datasets...")
    df = pd.merge_asof(
        tx_df, rpc_df, 
        on="timestamp", by="rpc_id", 
        direction="backward", tolerance=pd.Timedelta("2d")
    )
    
    df = df.dropna(subset=['latency_ms', 'block_lag', 'duration_sec'])
    df = df[df['status'] == 1]
    
    print(f"[SUCCESS] Generated graph dataset with {len(df)} matched samples.")
    return df

def plot_latency_correlation(df):
    print("[GRAPH] Generating Figure 1: Latency Correlation...")
    plt.figure(figsize=(10, 6))
    
    sns.regplot(
        data=df, x="latency_ms", y="duration_sec",
        scatter_kws={'alpha':0.5}, line_kws={'color':'red'}
    )
    
    plt.title("Impact of Network Latency on Transaction Confirmation Time")
    plt.xlabel("RPC Latency (ms) - [Ping Time]")
    plt.ylabel("Confirmation Duration (sec) - [On-Chain]")
    plt.tight_layout()
    plt.savefig("notebooks/Fig1_Latency_vs_Duration.png", dpi=300)

def plot_provider_performance(df):
    print("[GRAPH] Generating Figure 2: Provider Performance...")
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df, x="rpc_id", y="duration_sec", palette="viridis")
    
    plt.title("Performance Comparison by RPC Provider")
    plt.xlabel("RPC Provider")
    plt.ylabel("Transaction Duration (Seconds)")
    plt.tight_layout()
    plt.savefig("notebooks/Fig2_Provider_Performance.png", dpi=300)

def plot_feature_importance():
    print("[GRAPH] Generating Figure 3: Feature Importance...")
    
    import xgboost as xgb
    model = xgb.XGBRegressor()
    try:
        model.load_model("rpc_model.json")
    except:
        print("[ERROR] Could not find rpc_model.json. Did you train the model?")
        return
        
    importance = model.feature_importances_
    features = ['Latency (ms)', 'Block Lag']
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=features, y=importance, palette="magma")
    
    plt.title("Feature Importance: What drives Transaction Delay?")
    plt.ylabel("Relative Importance (0-1)")
    plt.tight_layout()
    plt.savefig("notebooks/Fig3_Feature_Importance.png", dpi=300)

if __name__ == "__main__":
    plt.switch_backend('Agg')
    df = load_and_merge_data()
    if len(df) > 0:
        plot_latency_correlation(df)
        plot_provider_performance(df)
        plot_feature_importance()
        print("\n[SUCCESS] All 3 figures saved to the 'notebooks/' folder!")
    else:
        print("[ERROR] Cannot generate graphs without data.")