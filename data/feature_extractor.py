class FeatureExtractor:

    def extract_features(self, tx):

        features = {}

        features["value"] = tx.get("value", 0)
        features["gas_price"] = tx.get("gas_price", 0)
        features["address_age_days"] = tx.get("address_age_days", 0)
        features["tx_count_last_min"] = tx.get("tx_count_last_min", 0)

        features["high_value"] = 1 if features["value"] > 10000 else 0
        features["new_wallet"] = 1 if features["address_age_days"] < 1 else 0

        return features