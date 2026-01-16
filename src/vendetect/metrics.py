"""Comparison metrics for ranking detection results."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .comparison import Comparison


class ComparisonMetric(ABC):
    """Abstract base class for comparison metrics."""

    @abstractmethod
    def score(self, comparison: "Comparison") -> float:
        """Calculate a score for the comparison. Higher scores indicate better matches."""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """Return the name of this metric."""
        raise NotImplementedError


class SumSimilarityMetric(ComparisonMetric):
    """Default metric: sum of both similarity scores."""

    def score(self, comparison: "Comparison") -> float:
        return comparison.similarity1 + comparison.similarity2

    def name(self) -> str:
        return "sum"


class AverageSimilarityMetric(ComparisonMetric):
    """Average of both similarity scores."""

    def score(self, comparison: "Comparison") -> float:
        return (comparison.similarity1 + comparison.similarity2) / 2

    def name(self) -> str:
        return "average"


class MinSimilarityMetric(ComparisonMetric):
    """Minimum of both similarity scores (most conservative)."""

    def score(self, comparison: "Comparison") -> float:
        return min(comparison.similarity1, comparison.similarity2)

    def name(self) -> str:
        return "min"


class MaxSimilarityMetric(ComparisonMetric):
    """Maximum of both similarity scores (most aggressive)."""

    def score(self, comparison: "Comparison") -> float:
        return max(comparison.similarity1, comparison.similarity2)

    def name(self) -> str:
        return "max"


class TokenOverlapMetric(ComparisonMetric):
    """Raw token overlap count."""

    def score(self, comparison: "Comparison") -> float:
        return float(comparison.token_overlap)

    def name(self) -> str:
        return "token_overlap"


class WeightedSimilarityMetric(ComparisonMetric):
    """Weighted combination of similarities and token overlap."""

    def __init__(self, sim_weight: float = 0.8, token_weight: float = 0.2):
        self.sim_weight = sim_weight
        self.token_weight = token_weight

    def score(self, comparison: "Comparison") -> float:
        sim_score = (comparison.similarity1 + comparison.similarity2) / 2
        # Normalize token overlap to 0-1 range (assuming max 1000 tokens for normalization)
        normalized_tokens = min(comparison.token_overlap / 1000.0, 1.0)
        return self.sim_weight * sim_score + self.token_weight * normalized_tokens

    def name(self) -> str:
        return "weighted"


# Registry of available metrics
METRICS = {
    "sum": SumSimilarityMetric(),
    "average": AverageSimilarityMetric(),
    "min": MinSimilarityMetric(),
    "max": MaxSimilarityMetric(),
    "token_overlap": TokenOverlapMetric(),
    "weighted": WeightedSimilarityMetric(),
}


def get_metric(name: str) -> ComparisonMetric:
    """Get a metric by name."""
    if name not in METRICS:
        available = ", ".join(METRICS.keys())
        msg = f"Unknown metric: {name}. Available metrics: {available}"
        raise ValueError(msg)
    return METRICS[name]
