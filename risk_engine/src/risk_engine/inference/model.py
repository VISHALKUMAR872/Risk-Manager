from abc import ABC, abstractmethod

from risk_engine.contracts import FeatureVector


class RiskModel(ABC):
    """
    Abstraction for a risk model.

    Production implementations can wrap CatBoost,
    Isolation Forest, or another validated model.
    """

    @property
    @abstractmethod
    def model_version(self) -> str:
        ...

    @abstractmethod
    def predict_probability(self, features: FeatureVector) -> float:
        """
        Return a raw fraud probability in [0, 1].
        """
        ...