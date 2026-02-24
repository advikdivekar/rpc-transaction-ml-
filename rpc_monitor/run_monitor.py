from rpc_monitor.transaction_listener import TransactionListener
from baseline_engine.evaluator import Evaluator


def run():

    listener = TransactionListener()
    evaluator = Evaluator()

    tx = listener.get_transaction()

    result = evaluator.evaluate_transaction(tx)

    print("Transaction detected:")
    print(result)


if __name__ == "__main__":
    run()