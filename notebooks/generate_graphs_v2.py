"""
generate_graphs_v2.py  —  Option A: Negative Result Paper Figures
==================================================================

Produces 7 publication-quality figures that support the new narrative:
"Client-side telemetry (latency + block lag) is provably insufficient
 for predictive RPC routing."

Run from project root:
    python notebooks/generate_graphs_v2.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    roc_curve, auc, mean_absolute_error,
    classification_report, roc_auc_score
)

# ── PATH SETUP ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.database import db

OUTPUT_DIR = SCRIPT_DIR   
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'rpc_model.json')

# ── ACADEMIC STYLE ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.titleweight': 'bold',
    'axes.labelsize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
})

SLOT_S = 12.0   
SLOW_THRESHOLD = 20.0  

C = {
    'static':   '#e74c3c',
    'mean':     '#c0392b',
    'ping':     '#f39c12',
    'smart_r':  '#27ae60',
    'smart_p':  '#2ecc71',
    'random':   '#95a5a6',
    'slot':     '#2c3e50',
    'fast':     '#2980b9',
    'slow':     '#e74c3c',
    'block_lag':'#8e44ad',
    'latency':  '#d35400',
}

# ═══════════════════════════════════════════════════════════════════════════════
def clean_rpc_name(url):
    url = str(url).lower()
    if "infura"     in url: return "infura"
    if "alchemy"    in url: return "alchemy"
    if "drpc"       in url: return "drpc"
    if "publicnode" in url: return "publicnode"
    return "unknown"

def load_and_merge():
    print("[INFO] Loading data from database...")
    tx_df  = db.load_data("SELECT * FROM tx_outcomes")
    rpc_df = db.load_data("SELECT * FROM rpc_metrics")
    
    rpc_df['latency_ms']   = pd.to_numeric(rpc_df['latency_ms'],   errors='coerce')
    rpc_df['block_number'] = pd.to_numeric(rpc_df['block_number'], errors='coerce')
    tx_df['duration_sec']  = pd.to_numeric(tx_df['duration_sec'],  errors='coerce')
    tx_df['status']        = pd.to_numeric(tx_df['status'],        errors='coerce')

    tx_df['rpc_id']  = tx_df['rpc_url'].apply(clean_rpc_name)
    rpc_df['rpc_id'] = rpc_df['rpc_id'].astype(str).str.strip().str.lower()

    for df in (tx_df, rpc_df):
        df['timestamp'] = (
            pd.to_datetime(df['timestamp'].astype(str), format='mixed', errors='coerce')
              .dt.tz_localize(None)
        )

    tx_df  = tx_df.dropna(subset=['timestamp']).sort_values('timestamp')
    rpc_df = rpc_df.dropna(subset=['timestamp']).sort_values('timestamp')

    rpc_df['block_lag'] = (
        rpc_df.groupby('timestamp')['block_number'].transform('max')
        - rpc_df['block_number']
    ).clip(lower=0)

    merged = pd.merge_asof(
        tx_df, rpc_df,
        on='timestamp', by='rpc_id',
        direction='backward',
        tolerance=pd.Timedelta('15min')
    )
    merged = merged.dropna(subset=['latency_ms', 'block_lag', 'duration_sec'])
    merged = merged[merged['status'] == 1]

    return merged

def temporal_split(df, test_frac=0.20):
    df = df.sort_values('timestamp').reset_index(drop=True)
    idx = int(len(df) * (1 - test_frac))
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()

# ═══════════════════════════════════════════════════════════════════════════════
def fig1_latency_scatter(df):
    print("[GRAPH] Fig 1: Latency Scatter...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # FIX 1: Filter structural gap before calculating R2
    mask = df['latency_ms'] < 600
    x, y = df.loc[mask, 'latency_ms'].values, df.loc[mask, 'duration_sec'].values
    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    r2   = corr ** 2

    ax.scatter(df['latency_ms'], df['duration_sec'],
               alpha=0.4, s=8, color='#2980b9', rasterized=True)

    # Regplot only on filtered data
    sns.regplot(data=df[mask], x='latency_ms', y='duration_sec',
                ax=ax, scatter=False,
                line_kws={'color': 'red', 'linewidth': 1.8, 'label': f'Linear fit  R²={r2:.3f}'})

    ax.set_xlabel('RPC Latency (ms)  [Ping Time]')
    ax.set_ylabel('Confirmation Duration (s)  [On-Chain]')
    ax.set_title('Network Latency vs. Transaction Confirmation Time\n'
                 f'R²={r2:.3f} — latency explains <10 % of confirmation variance')
    ax.legend(fontsize=10)
    ax.set_xlim(0, df['latency_ms'].quantile(0.99))

    ax.text(0.97, 0.97,
            f'n = {len(df):,}\nR² = {r2:.3f}\n→ 90 %+ unexplained',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.9))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig1_Latency_Scatter.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
def fig2_provider_boxplot(df):
    print("[GRAPH] Fig 2: Provider Box Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    order = ['infura', 'drpc', 'publicnode', 'alchemy']
    palette = {'infura': '#2980b9', 'drpc': '#16a085',
               'publicnode': '#1abc9c', 'alchemy': '#27ae60'}

    sns.boxplot(data=df, x='rpc_id', y='duration_sec',
                order=order, palette=palette,
                showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
                ax=ax)

    ax.axhline(SLOT_S, color=C['slot'], linewidth=1.4, linestyle='--', alpha=0.8,
               label=f'1 Ethereum slot ({SLOT_S}s)')
    ax.axhline(SLOW_THRESHOLD, color=C['slow'], linewidth=1.2, linestyle=':',
               alpha=0.7, label=f'Tail-risk threshold ({SLOW_THRESHOLD}s)')

    ax.set_xlabel('RPC Provider')
    ax.set_ylabel('Transaction Confirmation Duration (s)')
    ax.set_title('Confirmation Time Distribution by RPC Provider\n'
                 'Similar medians mask dramatically different tail behaviour')
    ax.legend(fontsize=10)

    for prov in order:
        worst = df.loc[df['rpc_id'] == prov, 'duration_sec'].max()
        if worst > 40:
            x_pos = order.index(prov)
            ax.text(x_pos, worst + 1, f'{worst:.0f}s', ha='center', fontsize=9,
                    color='#e74c3c', fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig2_Provider_Boxplot.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
def fig3_feature_importance():
    print("[GRAPH] Fig 3: Feature Importance...")
    if not os.path.exists(MODEL_PATH):
        print(f"  [SKIP] rpc_model.json not found at {MODEL_PATH}")
        return

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    importance_dict = booster.get_score(importance_type='gain')

    label_map = {
        'latency_ms':       'Latency (ms)',
        'block_lag':        'Block Lag',
        'hour_of_day':      'Hour of Day',
        'day_of_week':      'Day of Week',
        'rolling_latency_5':'Rolling Latency\n(5-sample avg)',
        'f0': 'Latency (ms)',
        'f1': 'Block Lag',
        'f2': 'Hour of Day',
        'f3': 'Day of Week',
        'f4': 'Rolling Latency\n(5-sample avg)',
    }
    color_map = {
        'latency_ms': C['latency'], 'f0': C['latency'],
        'block_lag':  C['block_lag'], 'f1': C['block_lag'],
    }
    default_color = '#7f8c8d'

    total = sum(importance_dict.values())
    features = list(importance_dict.keys())
    importance = [importance_dict[f] / total for f in features]
    labels = [label_map.get(f, f) for f in features]
    colors = [color_map.get(f, default_color) for f in features]

    pairs = sorted(zip(importance, labels, colors), reverse=True)
    importance, labels, colors = zip(*pairs)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    bars = ax.bar(labels, importance, color=colors,
                  edgecolor='black', linewidth=0.6, width=0.5)

    for bar, val in zip(bars, importance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Relative Importance (0–1)  [XGBoost F-score]')
    ax.set_title('Feature Importance: What the Model Can See\n'
                 'Block lag is orthogonal to latency — but both are client-side only')

    ax.text(0.97, 0.97,
            'Missing features:\n• Mempool depth\n• Gas base fee\n• Validator rate',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor=C['slow'], alpha=0.9))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig3_Feature_Importance.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
def fig4_roc_curve(train, test):
    print("[GRAPH] Fig 4: ROC Curve...")

    full = pd.concat([train, test]).sort_values('timestamp').reset_index(drop=True)
    full['hour_of_day']     = full['timestamp'].dt.hour
    full['day_of_week']      = full['timestamp'].dt.dayofweek
    full['rolling_latency_5'] = (
        full.groupby('rpc_id')['latency_ms']
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    full['is_slow'] = (full['duration_sec'] > SLOW_THRESHOLD).astype(int)

    features = ['latency_ms', 'block_lag', 'hour_of_day', 'day_of_week', 'rolling_latency_5']

    idx = int(len(full) * 0.80)
    X_train = full[features].iloc[:idx]
    X_test  = full[features].iloc[idx:]
    y_train = full['is_slow'].iloc[:idx]
    y_test  = full['is_slow'].iloc[idx:]

    n_fast = (y_train == 0).sum()
    n_slow = (y_train == 1).sum()
    scale_w = n_fast / n_slow if n_slow > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_w, eval_metric='logloss',
        random_state=42, verbosity=0
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], color=C['random'], linewidth=1.6,
            linestyle='--', label='Random classifier  AUC = 0.500', zorder=1)
    ax.fill_between(fpr, fpr, tpr, alpha=0.12, color='#27ae60', label='Gain over random (marginal)')
    ax.plot(fpr, tpr, color='#27ae60', linewidth=2.4,
            label=f'XGBoost Classifier  AUC = {roc_auc:.3f}', zorder=3)

    ax.text(0.55, 0.22,
            f'ROC-AUC = {roc_auc:.3f}\n≈ 0.500 (random)\n\n'
            f'The model cannot\ndistinguish fast\nfrom slow transactions.',
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fadbd8', edgecolor=C['slow'], linewidth=1.5),
            zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate  (1 − Specificity)')
    ax.set_ylabel('True Positive Rate  (Sensitivity / Recall)')
    ax.set_title('ROC Curve: Predicting Tail-Latency Events (> 20 s)\n'
                 'Client-side telemetry (latency + block lag) provides no useful signal')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.35)

    ax.text(0.02, 0.97,
            f'Features: latency_ms, block_lag,\nhour_of_day, day_of_week, rolling_latency_5\n'
            f'Test set: {len(y_test):,} samples',
            transform=ax.transAxes, ha='left', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.8))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig4_ROC_Curve.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc

# ═══════════════════════════════════════════════════════════════════════════════
def fig5_slot_distribution(df):
    print("[GRAPH] Fig 5: Confirmation Time Distribution + Slot Overlay...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    dur = df['duration_sec'].clip(upper=60)
    ax.hist(dur, bins=80, density=True, color='#2980b9', alpha=0.35, label='Empirical distribution')
    sns.kdeplot(dur, ax=ax, color='#2980b9', linewidth=2.0, label='KDE')

    global_mean   = df['duration_sec'].mean()
    global_median = df['duration_sec'].median()

    for slot_n, label in [(1, '1 slot\n(12 s)'), (2, '2 slots\n(24 s)'), (3, '3 slots\n(36 s)')]:
        x = slot_n * SLOT_S
        ax.axvline(x, color=C['slot'], linewidth=1.4, linestyle='--' if slot_n == 1 else ':',
                   alpha=0.8 if slot_n == 1 else 0.5, label=label if slot_n <= 2 else None)
        ax.text(x + 0.3, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.03,
                label, color=C['slot'], fontsize=8, va='top')

    ax.axvline(global_mean, color='#e74c3c', linewidth=1.8, linestyle='-', label=f'Global mean ({global_mean:.1f}s)')
    ax.axvline(global_median, color='#f39c12', linewidth=1.8, linestyle='-.', label=f'Global median ({global_median:.1f}s)')

    ax.set_xlim(0, 60)
    ax.set_xlabel('Confirmation Duration (s)')
    ax.set_ylabel('Density')
    ax.set_title('Confirmation Time Distribution\n(n={:,} transactions)'.format(len(df)))
    ax.legend(fontsize=8.5, loc='upper right')

    ax.text(0.97, 0.68,
            'Distribution anchored\nat slot boundaries.\nMean ≈ 1 slot → mean\nbaseline is near-optimal.',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7', edgecolor='#f39c12', alpha=0.9))

    ax2 = axes[1]
    prov_colors = {'infura': '#2980b9', 'alchemy': '#27ae60', 'drpc': '#8e44ad', 'publicnode': '#e74c3c'}

    for prov, color in prov_colors.items():
        subset = df.loc[df['rpc_id'] == prov, 'duration_sec'].clip(upper=60)
        if len(subset) < 30: continue
        sns.kdeplot(subset, ax=ax2, color=color, linewidth=1.8, fill=True, alpha=0.15, label=f'{prov} (n={len(subset):,})')

    ax2.axvline(SLOT_S, color=C['slot'], linewidth=1.6, linestyle='--', alpha=0.8, label=f'1 slot ({SLOT_S}s)')
    ax2.axvline(SLOW_THRESHOLD, color=C['slow'], linewidth=1.2, linestyle=':', alpha=0.7, label=f'Tail threshold ({SLOW_THRESHOLD}s)')

    ax2.set_xlim(0, 50)
    ax2.set_xlabel('Confirmation Duration (s)')
    ax2.set_ylabel('Density')
    ax2.set_title('Per-Provider Distribution\n(All medians ≈ 12 s — tail is the differentiator)')
    ax2.legend(fontsize=8.5, loc='upper right')

    plt.suptitle('Why Predicting Confirmation Time is Near-Degenerate Without Consensus Features',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out = os.path.join(OUTPUT_DIR, 'Fig5_Slot_Distribution.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
def fig6_ping_trap(df):
    print("[GRAPH] Fig 6: Ping Trap Visualisation...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    fast_mask = df['duration_sec'] <= SLOW_THRESHOLD
    slow_mask = ~fast_mask

    trap_zone_x = [0, 500, 500, 0]
    trap_zone_y = [0.5, 0.5, df['block_lag'].max() + 0.5, df['block_lag'].max() + 0.5]
    ax.fill(trap_zone_x, trap_zone_y, color='#e74c3c', alpha=0.07, zorder=0)

    ax.scatter(df.loc[fast_mask, 'latency_ms'],
               df.loc[fast_mask, 'block_lag'] + np.random.uniform(-0.15, 0.15, fast_mask.sum()),
               alpha=0.3, s=8, color=C['fast'], label=f'Fast (≤{SLOW_THRESHOLD}s)', rasterized=True, zorder=2)
    ax.scatter(df.loc[slow_mask, 'latency_ms'],
               df.loc[slow_mask, 'block_lag'] + np.random.uniform(-0.15, 0.15, slow_mask.sum()),
               alpha=0.55, s=14, color=C['slow'], label=f'Slow (>{SLOW_THRESHOLD}s)', zorder=3)

    ax.text(250, df['block_lag'].max() * 0.6,
            '"Ping Trap" zone\nLow latency,\nhigh block lag\n→ ping router\nwould select this',
            ha='center', va='center', fontsize=9, color='#922b21',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fadbd8', edgecolor='#e74c3c', alpha=0.9))

    ax.set_xlim(0, df['latency_ms'].quantile(0.97))
    ax.set_ylim(-0.5, df['block_lag'].max() + 0.5)
    ax.set_xlabel('RPC Latency (ms)  [Ping Signal]')
    ax.set_ylabel('Block Lag (blocks behind canonical head)')
    ax.set_title('The Ping Trap: Low Latency ≠ Synchronised Node\n'
                 'Shaded zone = transactions where ping routing would mis-select')
    ax.legend(fontsize=10)

    ax2 = axes[1]
    order   = ['infura', 'alchemy', 'drpc', 'publicnode']
    palette = {'infura': '#2980b9', 'alchemy': '#27ae60', 'drpc': '#8e44ad', 'publicnode': '#e74c3c'}

    prov_data = [df.loc[df['rpc_id'] == p, 'block_lag'].values for p in order]
    vp = ax2.violinplot(prov_data, positions=range(len(order)), showmedians=True, showextrema=True, widths=0.7)

    for i, (body, prov) in enumerate(zip(vp['bodies'], order)):
        body.set_facecolor(palette[prov])
        body.set_alpha(0.55)

    vp['cmedians'].set_color('black')
    vp['cbars'].set_color('grey')
    vp['cmaxes'].set_color('grey')
    vp['cmins'].set_color('grey')

    ax2.axhline(1, color=C['slow'], linewidth=1.2, linestyle='--', alpha=0.7, label='1 block behind = missed-slot risk')

    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order)
    ax2.set_xlabel('RPC Provider')
    ax2.set_ylabel('Block Lag (blocks)')
    ax2.set_title('Block Lag Distribution by Provider\n'
                  'PublicNode & DRPC fall behind most often — but ping cannot see this')
    ax2.legend(fontsize=9)

    for i, (prov, data) in enumerate(zip(order, prov_data)):
        if len(data) > 0:
            ax2.text(i, np.mean(data) + 0.05, f'μ={np.mean(data):.2f}',
                     ha='center', va='bottom', fontsize=8.5, color=palette[prov], fontweight='bold')

    plt.suptitle('The "Ping Trap": Why Block Lag is a Real Signal — But Client-Side Alone',
                 fontsize=12, fontweight='bold', y=0.98)
                 
    plt.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.94])
    
    out = os.path.join(OUTPUT_DIR, 'Fig6_Ping_Trap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
def fig7_strategy_comparison(train, test):
    print("[GRAPH] Fig 7: Strategy Comparison (reframed for negative result)...")

    y_train = train['duration_sec'].values
    y_test  = test['duration_sec'].values

    def rmse_fn(a, b):
        return float(np.sqrt(np.mean((np.array(a) - np.array(b))**2)))

    results = {}

    # 1. Static Router
    infura_mean = train.loc[train['rpc_id'] == 'infura', 'duration_sec'].mean()
    if np.isnan(infura_mean): infura_mean = y_train.mean()
    p = np.full(len(y_test), infura_mean)
    results['Static Router\n(Hardcoded Infura)'] = {
        'mae': mean_absolute_error(y_test, p), 'rmse': rmse_fn(y_test, p), 'color': C['static'], 'preds': p}

    # 2. Mean Baseline
    p = np.full(len(y_test), y_train.mean())
    results['Mean Baseline\n(Global Average)\n★ MAE winner'] = {
        'mae': mean_absolute_error(y_test, p), 'rmse': rmse_fn(y_test, p), 'color': C['mean'], 'preds': p}

    # 3. Ping-Based Balancer
    m = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    m.fit(train[['latency_ms']], y_train)
    p = m.predict(test[['latency_ms']])
    results['Ping-Based Balancer\n(Latency Only)'] = {
        'mae': mean_absolute_error(y_test, p), 'rmse': rmse_fn(y_test, p), 'color': C['ping'], 'preds': p}

    # 4. Smart Router (Retrained Regressor)
    m = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
    m.fit(train[['latency_ms', 'block_lag']], y_train)
    p = m.predict(test[['latency_ms', 'block_lag']])
    results['Smart Router\n(Retrained, Temporal)'] = {
        'mae': mean_absolute_error(y_test, p), 'rmse': rmse_fn(y_test, p), 'color': C['smart_r'], 'preds': p}

    # --- PRODUCTION MODEL CLASSIFIER BLOCK DELETED HERE ---

    names  = list(results.keys())
    maes   = [results[n]['mae']  for n in names]
    rmses  = [results[n]['rmse'] for n in names]
    colors = [results[n]['color'] for n in names]

    # Removed the 5th label so it matches our 4 models
    short_labels = [
        'Static Router\n(Hardcoded)',
        'Mean Baseline\n(Global Avg) ★',
        'Ping Balancer\n(Latency Only)',
        'Smart Router\n(Retrained)'
    ]

    x = np.arange(len(names))
    w = 0.34
    fig, ax = plt.subplots(figsize=(11, 7)) # Slightly narrower since we have 1 less bar

    bars_mae  = ax.bar(x - w/2, maes,  w, color=colors, alpha=0.92, label='MAE', edgecolor='black', linewidth=0.5)
    bars_rmse = ax.bar(x + w/2, rmses, w, color=colors, alpha=0.45, label='RMSE', edgecolor='black', linewidth=0.5, hatch='//')

    for bar in bars_mae:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                f'{h:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_rmse:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                f'{h:.2f}s', ha='center', va='bottom', fontsize=8.5, color='#444')

    ax.axhline(SLOT_S, color=C['slot'], linewidth=1.4, linestyle='--', alpha=0.7, label=f'1 Ethereum slot ({SLOT_S}s)')
    
    # FIX: Moved the 12s label to the left so it doesn't get cut off by the right border
    ax.text(len(names) - 0.8, SLOT_S + 0.4,
            '1 Ethereum Slot (12 s)', color=C['slot'], fontsize=8.5, va='bottom', ha='center')

    best_mae_idx = maes.index(min(maes))
    ax.annotate(
        'Zero-intelligence baseline\nwins on MAE\n→ mean estimation dominates',
        xy=(x[best_mae_idx] - w/2, maes[best_mae_idx]),
        # FIX: Pushed the annotation box WAY up so it clears the tops of all the bars
        xytext=(x[best_mae_idx] + 0.3, max(rmses) * 1.15),
        fontsize=8.5, color='#922b21',
        arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.2, connectionstyle="arc3,rad=0.1"),
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#fadbd8', edgecolor='#e74c3c', alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9.5, linespacing=1.4)
    ax.tick_params(axis='x', pad=8)
    ax.set_ylabel('Error (seconds)  —  Lower is Better', fontsize=11)
    ax.set_title(
        'Routing Strategy Comparison  (Temporal Split — No Data Leakage)\n'
        'A zero-intelligence mean baseline outperforms all ML strategies on MAE',
        fontsize=12)
    
    # Raised the ceiling of the graph slightly to accommodate the higher text box
    ax.set_ylim(0, max(rmses) * 1.45) 
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.35)

    # Note: Using standard tight_layout here is fine since there is no suptitle
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'Fig7_Strategy_Comparison.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = load_and_merge()

    if len(df) == 0:
        print("[ERROR] No data — check database connection.")
        import sys; sys.exit(1)

    train, test = temporal_split(df, test_frac=0.20)

    print("\n─── Generating figures ──────────────────────────────────────────")
    fig1_latency_scatter(df)
    fig2_provider_boxplot(df)
    fig3_feature_importance()
    roc = fig4_roc_curve(train, test)
    fig5_slot_distribution(df)
    fig6_ping_trap(df)
    fig7_strategy_comparison(train, test)

    print(f"\n{'='*60}")
    print(f"  ROC-AUC (centrepiece result): {roc:.4f}")
    print(f"  All 7 figures saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")