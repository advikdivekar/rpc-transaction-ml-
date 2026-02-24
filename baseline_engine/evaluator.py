from baseline_engine.strategies import (
    LargeTransactionStrategy,
    HighFrequencyStrategy,
    NewAddressStrategy
)

from data.feature_extractor import FeatureExtractor


class Evaluator:

    def __init__(self):

        self.feature_extractor = FeatureExtractor()

        self.strategies = [
            LargeTransactionStrategy(),
            HighFrequencyStrategy(),
            NewAddressStrategy()
        ]

    def evaluate_transaction(self, tx):

        features = self.feature_extractor.extract_features(tx)

        risk_score = 0

        for strategy in self.strategies:
            risk_score += strategy.evaluate(features)

        risk_score = min(risk_score, 1.0)

        result = {
            "transaction": tx,
            "features": features,
            "risk_score": risk_score,
            "flagged": risk_score > 0.7
        }

        return result