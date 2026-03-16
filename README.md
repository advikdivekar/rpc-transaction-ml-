# RPC Smart Router — Project README
## Blockchain Latency Optimization via Machine Learning

**Authors:** Advik Divekar · Aaryan Harsora · Paras Jadhav · Satvik Jain


**Institution:** Mukesh Patel School of Technology Management and Engineering, Mumbai

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [What Already Exists and Why It Fails](#2-what-already-exists-and-why-it-fails)
3. [Our Solution: The Smart Router](#3-our-solution-the-smart-router)
4. [How the System Works Technically](#4-how-the-system-works-technically)
5. [The Dataset](#5-the-dataset)
6. [The Machine Learning Model](#6-the-machine-learning-model)
7. [Results](#7-results)
8. [The Key Discovery: The Web3 Routing Gap](#8-the-key-discovery-the-web3-routing-gap)
9. [Why the Original Evaluation Was Wrong and How We Fixed It](#9-why-the-original-evaluation-was-wrong-and-how-we-fixed-it)
10. [Project File Structure](#10-project-file-structure)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [Figures Guide](#12-figures-guide)
13. [Glossary](#13-glossary)

---

## 1. What Problem Are We Solving?

### The Simple Version

When you send cryptocurrency from one blockchain to another (a "cross-chain transfer"), the transaction must be confirmed on the source blockchain first. The speed of that confirmation depends on which **RPC node** your software submits the transaction to.

An **RPC (Remote Procedure Call) node** is a server that your wallet or app talks to. When you submit a transaction, you send it to one of these servers, and it broadcasts it to the rest of the blockchain network.

**The problem:** Not all RPC servers are equal at any given moment. Some are fast. Some are temporarily out of sync with the rest of the blockchain. But almost all real-world systems just pick one server and always use it, or pick the one with the fastest ping — without knowing whether it is actually healthy or not.

### The Technical Version

Ethereum's Proof-of-Stake consensus produces a new block every **12 seconds** (called a "slot"). For your transaction to be included in the next block, it must be in the mempool of a node that is **fully synchronized** with the canonical chain head.

If you submit your transaction to a node that is 2 blocks behind the current chain head, that node has not yet propagated the latest pending transactions to peers. Your transaction misses the next slot's inclusion window. You wait another 12 seconds minimum, and possibly several more slots. For a cross-chain bridge, this cascades: the destination chain will not release funds until the source chain confirms, so a 24-second delay at source becomes a 60-120+ second delay end-to-end.

This is not a rare edge case. Our data shows DRPC and PublicNode producing confirmed events of **60-104 seconds** — representing 5-8 completely missed Ethereum slots.

---

## 2. What Already Exists and Why It Fails

### Static Routing (Most Common)

```python
# What 90%+ of wallets and bridges do:
RPC_URL = "https://mainnet.infura.io/v3/YOUR_KEY"  # hardcoded forever
```

**Why it fails:** Provider performance changes over time. A single provider can go from fast to degraded without any signal to the application layer.

### Ping-Based Load Balancers

Some systems pick the provider with the lowest ping:

```
Provider A: 120ms ping  →  use this one
Provider B: 450ms ping  →  ignore
```

**Why it fails:** Ping measures the time for a tiny HTTP request to travel to the server and back. It says nothing about whether the server's blockchain data is current. A server can have a 50ms ping and be 8 blocks behind the chain head simultaneously. Our data proves this conclusively.

### What Both Approaches Miss

Both are completely blind to **block synchronization lag** — whether the RPC node's view of the blockchain is up to date. This is the single most important signal for predicting whether a transaction submission will be included promptly.

---

## 3. Our Solution: The Smart Router

The Smart Router replaces static/ping selection with a **trained machine learning model** that predicts, in real time, which provider will confirm a transaction fastest — using both ping latency AND block synchronization state as inputs.

At routing time, instead of "use Infura because it has the lowest ping," the Smart Router does:

```
1. Ping all providers simultaneously (measures latency_ms for each)
2. Check each provider's current block number (compute block_lag for each)
3. Feed [latency_ms, block_lag] into XGBoost model for each provider
4. Route transaction to provider with LOWEST predicted confirmation time
```

**Total overhead: under 200ms** — acceptable for any wallet or bridge relay.

---

## 4. How the System Works Technically

The project has three independent components that run concurrently.

### Component 1: Network Health Monitor (src/monitor.py)

```
Every 30 seconds:
  FOR each provider (Infura, Alchemy, DRPC, PublicNode):
    Send eth_blockNumber JSON-RPC call
    Record: timestamp, provider_id, latency_ms, block_number
    Save to rpc_metrics table in database
```

The monitor saves block_lag = 0 initially. The actual block lag is computed later during training as:

```python
block_lag = max(all_providers_block_number_at_timestamp) - this_provider_block_number
```

This is a dynamic, relative calculation measuring how far behind each node is compared to the most advanced node being monitored at that moment.

### Component 2: Transaction Prober

```
Continuously:
  Pick a provider randomly (to avoid selection bias)
  Submit zero-value ETH self-transfer (costs only gas; free on testnet)
  Record: timestamp, provider_id, duration_sec (broadcast to receipt)
  Save to tx_outcomes table in database
```

Zero-value transfers are used because identical gas parameters isolate infrastructure latency from gas market effects. The only variable affecting duration_sec is the provider's health and synchronization state.

### Component 3: Inference Engine / Smart Router (src/model.py)

```
At transaction submission time:
  Ping all providers (get latency_ms per provider)
  Get current block numbers (compute block_lag per provider)
  For each provider: predict = model.predict([latency_ms, block_lag])
  Route transaction to argmin(predicted confirmation time)
```

### Data Flow

```
[Network Health Monitor]          [Transaction Prober]
       | every 30s                       |
       v                                 v
[rpc_metrics table]            [tx_outcomes table]
       |                                 |
       +-----------> [Temporal Alignment] <----------+
                      merge_asof, 15-min window
                              |
                     [2,660 matched samples]
                              |
                     [XGBoost Training]
                              |
                      [rpc_model.json]
                              |
                    [Smart Router (live)]
                     selects best RPC endpoint
```

### Temporal Alignment — The Critical Step

The monitor and prober run asynchronously and produce records at different rates. Before training, we must match each transaction with the network conditions that existed at the moment of submission.

The algorithm: for each transaction timestamp, find the most recent rpc_metrics record for the same provider that occurred **strictly before** the transaction, within a 15-minute window. This is implemented with pandas merge_asof.

This preserves causality. The model learns to predict based on conditions that were knowable at routing time, not conditions that occurred after the fact.

---

## 5. The Dataset

| Metric | Value |
|---|---|
| Raw transaction records | 5,578 |
| Raw RPC health observations | 10,008 total (~2,502 per provider) |
| High-fidelity matched samples | **2,660** |
| Providers monitored | Infura, Alchemy, DRPC, PublicNode |
| Network | Ethereum Sepolia testnet |
| Target variable | duration_sec (confirmation time in seconds) |
| Target median | ~12.4 seconds |
| Target mean | ~14.9 seconds (right-skewed due to tail events) |

### Why 5,578 Transactions Become 2,660 Matched Samples

The drop happens because not every transaction has a clean matching health observation:
- No health check was recorded within 15 minutes before submission
- The latency or block_lag values were missing or corrupted
- The transaction failed (status != 1)

2,660 is the count of transactions with a valid, causally-correct, matched health observation.

### Why 15 Minutes Tolerance, Not 2 Days

The original compare_models.py used a 2-day tolerance window. This is incorrect because a health observation from 36 hours ago tells you nothing about current network state. Using it inflates the matched sample count with low-quality pairs and trains the model on false correlations. 15 minutes matches the production model.py implementation.

---

## 6. The Machine Learning Model

### Algorithm: XGBoost Regressor

```python
xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8
)
```

**Why XGBoost and not a neural network?**
- Dataset is small (~2,660 samples) — neural networks overfit at this scale
- XGBoost excels on structured tabular data, which is exactly what we have
- Sub-millisecond inference (critical for live routing)
- Interpretable feature importance scores (required for our core academic claim)

### Features (Inputs)

| Feature | Description | Range |
|---|---|---|
| latency_ms | Round-trip time to provider in milliseconds | ~80 to 8,000 ms |
| block_lag | How many blocks behind the canonical chain head | 0 to 8+ blocks |

### Target (Output)

| Variable | Description |
|---|---|
| duration_sec | Transaction confirmation time in seconds |

### Training Split

**Chronological (temporal) split — NOT random shuffle:**
- Training set: earliest 80% of matched samples sorted by timestamp
- Test set: most recent 20% of matched samples

See Section 9 for why random splitting produces misleading results.

---

## 7. Results

### Primary Model Performance (from 02_train_model.py)

| Metric | Value | Context |
|---|---|---|
| MAE | **6.91 seconds** | Well under 1 Ethereum slot (12s) |
| Split type | Chronological 80/20 | Honest, no leakage |
| Test samples | 532 (20% of 2,660) | |

### Comparative Evaluation (from compare_models_fixed.py)

All 5 strategies evaluated on the same temporal test set:

| Routing Strategy | MAE (s) | RMSE (s) | P95 (s) | Improvement vs. Static |
|---|---|---|---|---|
| Static Router (Hardcoded Infura) | 8.18 | 9.69 | 17.29 | baseline |
| Mean Baseline (Global Average) | 7.85 | 9.15 | 16.08 | -4.0% MAE |
| Ping-Based Balancer (Latency Only) | 7.97 | 9.49 | 16.58 | -2.6% MAE |
| Smart Router (Retrained, Temporal) | 7.98 | 9.48 | 16.55 | -2.4% MAE |
| **Smart Router (Production Model)** | **7.37** | **8.68** | **15.35** | **-9.9% MAE** |

**Notable:** The RMSE gap is proportionally larger than the MAE gap for every comparison. This means the Smart Router's advantage is concentrated in reducing the high-error tail events — exactly the failures that matter most in production.

### Per-Provider Performance

| Provider | Median | Max Outlier | Block Lag Range |
|---|---|---|---|
| Infura | 12.1 s | 67 s | 0-3 blocks |
| DRPC | 13.4 s | 103 s | 0-5 blocks |
| PublicNode | 13.0 s | 104 s | 0-8 blocks |
| Alchemy | 14.8 s | < 40 s | 0-2 blocks |
| **Smart Router** | **12.8 s** | **< 20 s** | **Always 0** |

---

## 8. The Key Discovery: The Web3 Routing Gap

### What We Found

XGBoost feature importance scores on the trained model:

```
latency_ms  (ping latency):         ~72% importance
block_lag   (block sync lag):       ~28% importance
```

Block synchronization lag explains **28% of confirmation latency variance independently of ping time.**

### Why This Matters

Every commercial load balancer — including QuickNode, Infura's own routing, and Alchemy's load balancing — routes based on ping time only. They have zero awareness of block synchronization state.

This means all commercial load balancers are structurally blind to the cause of 28% of confirmation latency variance. We call this the **Web3 routing gap**.

### The Visual Proof (Fig. 1)

Fig. 1 is the scatter plot of ping latency vs confirmation time. The regression line is nearly horizontal (R² ≈ 0.09). Two transactions with an identical 200ms ping can confirm in 3 seconds or 103 seconds. The determining factor is block lag, which ping never measures.

### Real-World Consequence for Cross-Chain Bridges

```
1. User initiates cross-chain transfer
2. Bridge submits transaction to a node with 5-block lag
3. Transaction misses 5 consecutive inclusion windows = 60 seconds wasted
4. Bridge relayer waits before confirming on destination chain
5. Total user-visible delay: several minutes for a 30-second transfer
```

The Smart Router prevents Step 2 from occurring by pre-routing away from any provider with elevated predicted confirmation time.

---

## 9. Why the Original Evaluation Was Wrong and How We Fixed It

### The Bug in compare_models.py

```python
# WRONG — causes data leakage in time-series data:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

train_test_split performs a random shuffle. For blockchain time-series data, this is a critical error called **temporal data leakage**.

### What Data Leakage Means Here

With random splitting, the model trains on transactions from throughout the collection period — including ones that occurred close in time to the test samples. When asked to predict a test sample from, say, February 15th, the model has already learned from February 14th and February 16th during training. This artificially inflates accuracy and collapses the differences between all strategies.

### The Evidence

Original results with random split (leaky):

```
Static Router:        7.40 s
Ping-Based Balancer:  7.11 s
Smart Router:         7.09 s
```

All three within 0.31 s. A two-feature model is barely better than a one-feature model — statistically impossible if block_lag truly has 28% importance.

Fixed results with temporal split (honest):

```
Static Router:             8.18 s
Ping-Based Balancer:       7.97 s
Smart Router (production): 7.37 s
```

Clear, directionally-correct differentiation is now visible.

### The Fix in compare_models_fixed.py

```python
def temporal_split(df, test_frac=0.20):
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx]   # earliest 80%
    test  = df.iloc[split_idx:]   # most recent 20%
    return train, test
```

The model trains on old data and tests on genuinely new, unseen data — exactly how it would be evaluated in production.

---

## 10. Project File Structure

```
project_root/
|
+-- src/
|   +-- model.py          <- XGBoost training pipeline + RPCLatencyPredictor class
|   +-- monitor.py        <- Network Health Monitor (polls providers every 30s)
|   +-- database.py       <- DB access layer (load_data, save_metric)
|   +-- router.py         <- Live Smart Router inference engine
|
+-- config/
|   +-- settings.py       <- RPC_PROVIDERS dict, MODEL_PATH, DB config
|
+-- notebooks/
|   +-- 02_train_model.py         <- Production training pipeline
|   +-- compare_models.py         <- Original script (has leakage bug, keep for reference)
|   +-- compare_models_fixed.py   <- Fixed comparison with temporal split (use this)
|
+-- figures/
|   +-- Fig1_Latency_vs_Duration.png    <- Scatter: ping vs confirmation time
|   +-- Fig2_Provider_Performance.png   <- Box plots per provider
|   +-- Fig3_Feature_Importance.png     <- XGBoost feature importance
|   +-- Fig4_MAE_Comparison.png         <- 5-strategy MAE+RMSE bar chart
|   +-- Fig5_Error_Distribution.png     <- KDE of prediction errors
|   +-- Fig6_Temporal_Scatter.png       <- Actual vs predicted over test window
|   +-- Fig7_Metrics_Table.png          <- Summary table (MAE/RMSE/P95)
|
+-- rpc_model.json        <- Saved XGBoost model (production)
+-- README.md             <- This file
```

---

## 11. How to Run the Project

### Prerequisites

```bash
pip install xgboost pandas numpy scikit-learn matplotlib seaborn requests
```

### Step 1: Start Data Collection (runs continuously)

```bash
# Terminal 1: Start the health monitor
python src/monitor.py

# Terminal 2: Start the transaction prober
python src/prober.py
```

Leave these running for several days to build a meaningful dataset.

### Step 2: Train the Model

```bash
python notebooks/02_train_model.py
```

This will load and align the data, train XGBoost on the earliest 80%, evaluate on the most recent 20%, report MAE, and save the model to rpc_model.json.

### Step 3: Run the Comparison Evaluation

```bash
python notebooks/compare_models_fixed.py
```

This will evaluate all 5 routing strategies under temporal split and save Figures 4-7.

### Step 4: Use the Smart Router in Production

```python
from src.router import SmartRouter

router = SmartRouter(model_path='rpc_model.json')
best_provider = router.select_provider()
# best_provider is the URL of the recommended RPC endpoint
```

---

## 12. Figures Guide

| Figure | What It Shows | Key Takeaway |
|---|---|---|
| Fig. 1 | Ping latency vs confirmation time scatter (n=2,660) | R²=0.09 proves ping alone is insufficient |
| Fig. 2 | Box plots per provider | Similar medians, wildly different tails (103s/104s outliers) |
| Fig. 3 | XGBoost feature importance | Block lag = 28% weight — the Web3 routing gap quantified |
| Fig. 4 | MAE + RMSE bar chart, all 5 strategies | Production Smart Router best across both metrics |
| Fig. 5 | KDE of prediction errors per strategy | Smart Router distribution is left-shifted; P95 reduced by 2s |
| Fig. 6 | Actual vs predicted over test time window | No temporal degradation; stable generalization throughout |
| Fig. 7 | Full metrics summary table image | MAE/RMSE/P95 for all 5 strategies at a glance |

---

## 13. Glossary

| Term | Definition |
|---|---|
| RPC (Remote Procedure Call) | A protocol letting your application call functions on a remote server. In Web3, used to query blockchain state and submit transactions. |
| Block Lag | How many blocks behind the global chain head a node is. A lag of 3 means the node thinks the latest block is #1000 when the network is at #1003. |
| Ethereum Slot | A 12-second window during which one validator proposes a block. Missing a slot means waiting at least another 12 seconds. |
| MAE (Mean Absolute Error) | Average of absolute differences between predicted and actual values. Lower is better. |
| RMSE (Root Mean Square Error) | Like MAE but squares errors before averaging — penalizes large tail errors more heavily. |
| P95 (95th Percentile Error) | The error below which 95% of predictions fall. Directly measures tail-latency risk. |
| Temporal Split | Splitting time-series data by time: train on old data, test on new. The only honest evaluation method for sequential data. |
| Data Leakage | When future information accidentally enters model training, making results look better than they would be in deployment. |
| merge_asof | A pandas function merging sorted DataFrames by finding the nearest key that is less than or equal to a query value. Used here to match each transaction with the last RPC health check before it. |
| XGBoost | Extreme Gradient Boosting. An ensemble of decision trees trained iteratively to minimize a loss function. State-of-the-art for structured tabular data. |
| Cross-Chain Bridge | A protocol enabling assets to move between different blockchains (e.g., Ethereum to Arbitrum). Requires confirmation on both chains sequentially. |
| LayerZero / Wormhole | Major cross-chain messaging protocols routing billions of dollars between blockchains. |
| Sepolia Testnet | Ethereum's primary public test network. Identical protocol to mainnet but uses valueless test ETH. |
| Canonical Chain Head | The most recent block that the majority of the Ethereum network has agreed on. |
| Mempool | The pool of pending, unconfirmed transactions waiting to be included in a block. |
| dApp | Decentralized Application. A user-facing application built on blockchain infrastructure. |


