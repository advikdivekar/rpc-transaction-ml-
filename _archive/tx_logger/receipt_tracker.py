import csv
import os
import time
from web3 import Web3
from datetime import datetime
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()
RPC_URL = os.getenv("RPC_URL")


CSV_PATH = os.getenv("TX_OUTCOMES_CSV_PATH")

# =========================
# WEB3 SETUP
# =========================
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# =========================
# TRACKER PROCESS
# =========================
print("Starting receipt tracker...")

def process_receipts():
    """
    Reads the CSV and updates rows that are still 'PENDING'.
    """
    try:
        # Load existing data
        rows = []
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row["block_number"] == "PENDING":
                    print(f"Tracking receipt for: {row['tx_hash']}")
                    try:
                        # Wait for receipt (same logic as wait_for_transaction_receipt)
                        receipt = w3.eth.wait_for_transaction_receipt(row["tx_hash"], timeout=120)
                        confirmation_time = datetime.utcnow()
                        
                        # Calculate delay
                        send_time = datetime.fromisoformat(row["send_time"])
                        delay = (confirmation_time - send_time).total_seconds()

                        # Update the row data
                        row["block_number"] = receipt.blockNumber
                        row["confirmation_time"] = confirmation_time.isoformat()
                        row["confirmation_delay_seconds"] = delay
                        print(f"Confirmed in block: {receipt.blockNumber}")
                    except Exception as e:
                        print(f"Could not confirm {row['tx_hash']}: {e}")
                
                rows.append(row)

        # Write updated data back to CSV
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    except FileNotFoundError:
        print("CSV file not found. Wait for sender to create it.")

# Run the tracker periodically
while True:
    process_receipts()
    print("Cycle complete. Checking for new pending transactions in 30 seconds...")
    time.sleep(30)
