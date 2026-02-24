import random


class TransactionListener:

    def get_transaction(self):

        tx = {
            "value": random.randint(100, 20000),
            "gas_price": random.randint(20, 200),
            "address_age_days": random.uniform(0, 30),
            "tx_count_last_min": random.randint(1, 15)
        }

        return tx