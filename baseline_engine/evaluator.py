import pandas as pd
import numpy as np

from baseline_engine.strategies import (
    single_rpc,
    round_robin,
    lowest_latency,
    freshest_block,
)

# Load datasets
rpc_metrics = pd.read_csv("data/rpc_metrics.csv")
tx_outcomes = pd.read_csv("data/tx_outcomes.csv")


def simulate(strategy):
    """
    Simulate strategy decisions across timestamps.
    """

    delays = []
    failures = 0

    grouped = rpc_metrics.groupby("timestamp")

    for timestamp, rows in grouped:

        rpc_choice = strategy(rows)

        tx = tx_outcomes[tx_outcomes["rpc_id"] == rpc_choice]

        if not tx.empty:
            delay = tx.iloc[0]["confirmation_delay_seconds"]
            delays.append(delay)
        else:
            failures += 1

    return delays, failures


def mean_delay(values):
    return np.mean(values)


def median_delay(values):
    return np.median(values)


def p95_delay(values):
    return np.percentile(values, 95)


def failure_rate(failures, total):
    if total == 0:
        return 0
    return failures / total


def run_baselines():

    strategies = {
        "single_rpc": single_rpc,
        "round_robin": round_robin,
        "lowest_latency": lowest_latency,
        "freshest_block": freshest_block,
    }

    results = []

    for name, strategy in strategies.items():

        delays, failures = simulate(strategy)

        total = len(delays) + failures

        results.append(
            {
                "strategy": name,
                "mean_delay": mean_delay(delays),
                "median_delay": median_delay(delays),
                "p95_delay": p95_delay(delays),
                "failure_rate": failure_rate(failures, total),
                "n": len(delays),
            }
        )

    df = pd.DataFrame(results)

    print(df)

    df.to_csv("results/baseline_results.csv", index=False)


if __name__ == "__main__":
    run_baselines()
