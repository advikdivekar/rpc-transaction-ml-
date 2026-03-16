from generate_graphs_v2 import load_and_merge, temporal_split
from scipy import stats
import numpy as np

# 1. Load data exactly like the graph script
df = load_and_merge()
train, test = temporal_split(df, test_frac=0.20)

# 2. Get the actual test durations
actuals = test['duration_sec'].values

# 3. Calculate the Mean Baseline predictions (Global Average)
mean_baseline_preds = np.full(len(actuals), train['duration_sec'].mean())

# 4. We know from your graph the production model RMSE/MAE is heavily influenced 
# by the mean, but let's test a random baseline vs mean baseline for the paper text
# Calculate absolute errors
baseline_errors = np.abs(actuals - mean_baseline_preds)
static_errors = np.abs(actuals - np.full(len(actuals), train.loc[train['rpc_id'] == 'infura', 'duration_sec'].mean()))

# 5. Run paired t-test
t_stat, p_val = stats.ttest_rel(baseline_errors, static_errors)

print("\n" + "="*50)
print("SENTENCE FOR YOUR PAPER:")
print(f'"The MAE difference between the mean baseline and the static routing strategy was tested for statistical significance (paired t-test, p = {p_val:.4e})."')
print("="*50 + "\n")