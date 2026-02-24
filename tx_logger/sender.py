import csv
import os
import time
import random
from web3 import Web3
from datetime import datetime
from dotenv import load_dotenv

# =========================
# CONFIG
# =========================
load_dotenv()
RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
CHAIN_ID = 11155111  # Sepolia

TOTAL_TRANSACTIONS = 500

CSV_PATH = os.getenv("TX_OUTCOMES_CSV_PATH")

# =========================
# WEB3 SETUP
# =========================
w3 = Web3(Web3.HTTPProvider(RPC_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)

print("Using RPC:", RPC_URL)
print("Wallet address:", account.address)

os.makedirs("data", exist_ok=True)

# Create CSV header if file does not exist
if not os.path.isfile(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rpc_id",
            "send_time",
            "tx_hash",
            "block_number",
            "confirmation_time",
            "confirmation_delay_seconds"
        ])

# =========================
# SENDER LOOP
# =========================
for i in range(TOTAL_TRANSACTIONS):
    try:
        print(f"\nSending transaction {i+1}/{TOTAL_TRANSACTIONS}")

        nonce = w3.eth.get_transaction_count(account.address)

        tx = {
            "nonce": nonce,
            "to": account.address,
            "value": w3.to_wei(0, "ether"),
            "gas": 21000,
            "gasPrice": w3.eth.gas_price,
            "chainId": CHAIN_ID,
        }

        send_time = datetime.utcnow()

        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        print("Transaction sent:", tx_hash.hex())

        # Log the transaction as 'pending' or partially filled
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                RPC_URL,
                send_time.isoformat(),
                tx_hash.hex(),
                "PENDING",
                "PENDING",
                "PENDING"
            ])

        # Wait 30–35 seconds (per your code)
        sleep_time = random.randint(30, 35)
        print(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)

    except Exception as e:
        print("Error occurred in sender:", e)
        time.sleep(60)

print("\nFinished sending transactions.")
