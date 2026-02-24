"""
receipt_tracker.py

Purpose:
Tracks confirmations of transactions sent by sender.py.
Computes actual mining delay using block timestamps.

Input:
data/sent_transactions.csv

Output:
data/confirmed_transactions.csv
"""

import os
import csv
import time
from web3 import Web3
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================

load_dotenv()
RPC_URL = os.getenv("RPC_URL")
INPUT_FILE = "data/sent_transactions.csv"
OUTPUT_FILE = "data/confirmed_transactions.csv"

# =========================
# WEB3 SETUP
# =========================

w3 = Web3(Web3.HTTPProvider(RPC_URL))

os.makedirs("data", exist_ok=True)

# =========================
# HELPER FUNCTION
# =========================

def wait_for_receipt(tx_hash):
    """
    Poll until transaction receipt is available.
    """
    while True:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if receipt:
            return receipt
        time.sleep(3)

# =========================
# MAIN PROCESS
# =========================

with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w", newline="") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    writer.writerow([
        "rpc_url",
        "tx_hash",
        "send_time",
        "block_timestamp",
        "block_number",
        "mining_delay_seconds"
    ])

    for row in reader:
        tx_hash = row["tx_hash"]
        send_time = row["send_time"]

        print("Waiting for confirmation:", tx_hash)

        try:
            receipt = wait_for_receipt(tx_hash)
            block = w3.eth.get_block(receipt.blockNumber)
            block_timestamp = block.timestamp

            mining_delay = block_timestamp - int(datetime.fromisoformat(send_time).timestamp())

            writer.writerow([
                row["rpc_url"],
                tx_hash,
                send_time,
                block_timestamp,
                receipt.blockNumber,
                mining_delay
            ])

            print(f"Confirmed in block {receipt.blockNumber}, delay: {mining_delay} seconds")

        except Exception as e:
            print("Error while tracking receipt:", e)
            continue

print("\nFinished tracking all transactions.")
