"""Comparison metrics for ranking and filtering detection results."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from .errors import VendetectError

if TYPE_CHECKING:
    from .comparison import Comparison


class UnknownMetricError(VendetectError):
    """Raised when a metric is requested by a name that is not registered."""


class ComparisonMetric(ABC):
    """Scores a `Comparison` so detections can be ranked and filtered.

    Higher scores always indicate a better match, so a metric fully determines both the
    order detections are reported in and which of them clear `--min-score`.
    """

    name: ClassVar[str]
    """The value accepted by `--metric`."""

    label: ClassVar[str]
    """Human-readable column heading for this metric's score."""

    default_threshold: ClassVar[float]
    """Threshold applied when `--min-score` is omitted, in this metric's own units."""

    integral: ClassVar[bool] = False
    """Whether scores are counts rather than ratios, which changes how they are rendered."""

    @abstractmethod
    def score(self, comparison: "Comparison") -> float:
        """Score a comparison, where higher is a better match."""
        raise NotImplementedError

    def format_score(self, score: float) -> str:
        """Render a score for machine-readable output (CSV and JSON)."""
        return str(int(score)) if self.integral else f"{score:.4f}"

    def display_score(self, score: float) -> str:
        """Render a score for human-readable output."""
        return f"{int(score)} tokens" if self.integral else f"{score:.1%}"


class SumSimilarityMetric(ComparisonMetric):
    """Sum of both similarity scores, in the range 0.0-2.0.

    Ranks identically to `average`; it is retained because it is the ordering Vendetect
    used before metrics were selectable.
    """

    name = "sum"
    label = "Similarity Sum"
    default_threshold = 1.0

    def score(self, comparison: "Comparison") -> float:
        return comparison.similarity1 + comparison.similarity2


class AverageSimilarityMetric(ComparisonMetric):
    """Average of both similarity scores, in the range 0.0-1.0."""

    name = "average"
    label = "Similarity"
    default_threshold = 0.5

    def score(self, comparison: "Comparison") -> float:
        return (comparison.similarity1 + comparison.similarity2) / 2


class MinSimilarityMetric(ComparisonMetric):
    """Lesser of the two similarity scores, requiring both files to match well."""

    name = "min"
    label = "Similarity"
    default_threshold = 0.5

    def score(self, comparison: "Comparison") -> float:
        return min(comparison.similarity1, comparison.similarity2)


class MaxSimilarityMetric(ComparisonMetric):
    """Greater of the two similarity scores, requiring only one file to match well.

    Useful when a small file is vendored into a much larger one, which drags the other
    direction's similarity down.
    """

    name = "max"
    label = "Similarity"
    default_threshold = 0.5

    def score(self, comparison: "Comparison") -> float:
        return max(comparison.similarity1, comparison.similarity2)


class TokenOverlapMetric(ComparisonMetric):
    """Raw count of overlapping tokens, favoring large matches over proportional ones.

    Ranking by token overlap surfaces plenty of false positives, because whitespace and
    punctuation overlap between unrelated files of the same language. The threshold is
    correspondingly blunt: prefer a high `--min-score` and treat the results as leads
    rather than findings.
    """

    name = "token_overlap"
    label = "Token Overlap"
    default_threshold = 1000
    integral = True

    def score(self, comparison: "Comparison") -> float:
        return float(comparison.token_overlap)


METRICS: dict[str, ComparisonMetric] = {
    metric.name: metric
    for metric in (
        SumSimilarityMetric(),
        AverageSimilarityMetric(),
        MinSimilarityMetric(),
        MaxSimilarityMetric(),
        TokenOverlapMetric(),
    )
}

DEFAULT_METRIC = "average"


def get_metric(name: str) -> ComparisonMetric:
    """Look up a metric by its `--metric` name.

    Args:
        name: The registered name of the metric.

    Returns:
        The metric registered under `name`.

    Raises:
        UnknownMetricError: If no metric is registered under `name`.

    """
    try:
        return METRICS[name]
    except KeyError:
        msg = f"unknown metric {name!r}; expected one of {', '.join(sorted(METRICS))}"
        raise UnknownMetricError(msg) from None
