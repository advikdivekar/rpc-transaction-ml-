class Strategy:
    def evaluate(self, features):
        raise NotImplementedError


class LargeTransactionStrategy(Strategy):

    def __init__(self, threshold=10000):
        self.threshold = threshold

    def evaluate(self, features):
        if features["value"] > self.threshold:
            return 0.7
        return 0.0


class HighFrequencyStrategy(Strategy):

    def __init__(self, limit=10):
        self.limit = limit

    def evaluate(self, features):
        if features["tx_count_last_min"] > self.limit:
            return 0.6
        return 0.0


class NewAddressStrategy(Strategy):

    def evaluate(self, features):
        if features["address_age_days"] < 1:
            return 0.5
        return 0.0