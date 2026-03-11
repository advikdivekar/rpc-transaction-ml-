# baseline_engine/strategies.py

RPC_LIST = ["cloudflare", "publicnode", "llamarpc"]

rr_index = 0


def single_rpc(rows):
    """
    Always select the same RPC.
    """
    return "publicnode"


def round_robin(rows):
    """
    Rotate through RPC providers.
    """
    global rr_index

    rpc = RPC_LIST[rr_index]
    rr_index = (rr_index + 1) % len(RPC_LIST)

    return rpc


def lowest_latency(rows):
    """
    Select RPC with lowest latency.
    """
    best = rows.loc[rows["latency_ms"].idxmin()]
    return best["rpc_id"]


def freshest_block(rows):
    """
    Select RPC with smallest block lag.
    """
    best = rows.loc[rows["block_lag"].idxmin()]
    return best["rpc_id"]
