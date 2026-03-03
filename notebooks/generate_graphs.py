import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    if "infura" in url: return "infura"
    if "alchemy" in url: return "alchemy"
    if "drpc" in url: return "drpc"
    if "publicnode" in url: return "publicnode"
    return "unknown"

def load_and_merge_data():
    print("[INFO] Loading data from Neon DB...")
    tx_df = db.load_data("SELECT * FROM tx_outcomes")
    rpc_df = db.load_data("SELECT * FROM rpc_metrics")
    
    # Preprocessing
    tx_df['rpc_id'] = tx_df['rpc_url'].apply(clean_rpc_name)
    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], utc=True)
    rpc_df['timestamp'] = pd.to_datetime(rpc_df['timestamp'], utc=True)
    
    tx_df = tx_df.sort_values("timestamp")
    rpc_df = rpc_df.sort_values("timestamp")
    
    # Merge
    df = pd.merge_asof(
        tx_df, rpc_df, 
        on="timestamp", by="rpc_id", 
        direction="backward", tolerance=pd.Timedelta("10m")
    )
    
    df = df.dropna(subset=['latency_ms', 'block_lag'])
    df = df[df['status'] == 1]
    return df

def plot_latency_correlation(df):
    """
    Figure 1: Scatter plot showing that higher latency = slower transactions.
    """
    print("[GRAPH] Generating Figure 1: Latency Correlation...")
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with regression line
    sns.regplot(
        data=df, x="latency_ms", y="duration_sec",
        scatter_kws={'alpha':0.5}, line_kws={'color':'red'}
    )
    
    plt.title("Impact of Network Latency on Transaction Confirmation Time")
    plt.xlabel("RPC Latency (ms) - [Ping Time]")
    plt.ylabel("Confirmation Duration (sec) - [On-Chain]")
    plt.tight_layout()
    plt.savefig("notebooks/Fig1_Latency_vs_Duration.png", dpi=300)
    print("   -> Saved to notebooks/Fig1_Latency_vs_Duration.png")

def plot_provider_performance(df):
    """
    Figure 2: Boxplot comparing the different providers.
    """
    print("[GRAPH] Generating Figure 2: Provider Performance...")
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df, x="rpc_id", y="duration_sec", palette="viridis")
    
    plt.title("Performance Comparison by RPC Provider")
    plt.xlabel("RPC Provider")
    plt.ylabel("Transaction Duration (Seconds)")
    plt.tight_layout()
    plt.savefig("notebooks/Fig2_Provider_Performance.png", dpi=300)
    print("   -> Saved to notebooks/Fig2_Provider_Performance.png")

def plot_feature_importance():
    """
    Figure 3: Feature Importance from the XGBoost Model.
    """
    print("[GRAPH] Generating Figure 3: Feature Importance...")
    
    # Load model
    predictor = RPCLatencyPredictor()
    # We need to trick the class to load the model without training
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model("rpc_model.json")
    
    importance = model.feature_importances_
    features = ['Latency (ms)', 'Block Lag']
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=features, y=importance, palette="magma")
    
    plt.title("Feature Importance: What drives Transaction Delay?")
    plt.ylabel("Relative Importance (0-1)")
    plt.tight_layout()
    plt.savefig("notebooks/Fig3_Feature_Importance.png", dpi=300)
    print("   -> Saved to notebooks/Fig3_Feature_Importance.png")

if __name__ == "__main__":
    # Ensure matplotlib doesn't try to open windows
    plt.switch_backend('Agg')
    
    df = load_and_merge_data()
    
    plot_latency_correlation(df)
    plot_provider_performance(df)
    plot_feature_importance()
    
    print("\n[SUCCESS] All figures generated in 'notebooks/' folder.")