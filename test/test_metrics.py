"""Tests for comparison metrics and the metric-aware output paths.

These pin two behaviors that were regressions in the original version of this feature:
the default metric must keep `--min-score` on a 0.0-1.0 scale (a `sum` default silently
halved the effective threshold), and every output format must filter and display using
the metric the user selected rather than a hardcoded average.
"""

import csv
import io
import json
from typing import TYPE_CHECKING, cast

import pytest
from rich.console import Console

from vendetect._cli import main, output_csv, output_json, output_rich
from vendetect.comparison import Comparison, Slice
from vendetect.detector import Detection
from vendetect.metrics import (
    DEFAULT_METRIC,
    METRICS,
    UnknownMetricError,
    get_metric,
)

if TYPE_CHECKING:
    from vendetect.repo import File

DEFAULT_TOKEN_OVERLAP = 100


class FakePath:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class FakeFile:
    def __init__(self, name: str):
        self.relative_path = FakePath(name)


def comparison(similarity1: float, similarity2: float, token_overlap: int = DEFAULT_TOKEN_OVERLAP) -> Comparison:
    return Comparison(
        token_overlap=token_overlap,
        similarity1=similarity1,
        similarity2=similarity2,
        slices1=(Slice(0, 10),),
        slices2=(Slice(5, 15),),
    )


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("sum", 1.2),
        ("average", 0.6),
        ("min", 0.4),
        ("max", 0.8),
        ("token_overlap", 3568.0),
    ],
)
def test_metric_scores(metric_name: str, expected: float) -> None:
    score = get_metric(metric_name).score(comparison(0.8, 0.4, token_overlap=3568))
    assert score == pytest.approx(expected)


def test_registry_keys_match_metric_names() -> None:
    for name, metric in METRICS.items():
        assert metric.name == name


def test_default_metric_is_registered() -> None:
    assert DEFAULT_METRIC in METRICS


def test_default_metric_is_on_a_unit_scale() -> None:
    # A default whose scores exceed 1.0 makes `--min-score 0.5` mean something other
    # than "50% similar", which is what the flag and its default imply.
    metric = get_metric(DEFAULT_METRIC)
    assert metric.score(comparison(1.0, 1.0)) == pytest.approx(1.0)
    assert metric.default_threshold == pytest.approx(0.5)


def test_get_metric_rejects_unknown_name() -> None:
    with pytest.raises(UnknownMetricError, match="unknown metric 'nope'"):
        get_metric("nope")


def test_token_overlap_formats_as_a_count() -> None:
    metric = get_metric("token_overlap")
    assert metric.integral
    assert metric.format_score(3568.0) == "3568"
    assert metric.display_score(3568.0) == "3568 tokens"


def test_similarity_metrics_format_as_ratios() -> None:
    metric = get_metric("average")
    assert not metric.integral
    assert metric.format_score(0.25) == "0.2500"
    assert metric.display_score(0.25) == "25.0%"


def detection(similarity1: float, similarity2: float, token_overlap: int = DEFAULT_TOKEN_OVERLAP) -> Detection:
    file = cast("File", FakeFile("unused"))
    return Detection(file, file, comparison(similarity1, similarity2, token_overlap))


@pytest.mark.parametrize(
    ("metric_name", "expected_order"),
    [
        # A balanced pair, a lopsided pair, and a mediocre pair rank in three different
        # orders depending on the metric, which is the whole point of making it selectable.
        ("average", [0.9, 1.0, 0.55]),
        ("min", [0.9, 0.55, 1.0]),
        ("max", [1.0, 0.9, 0.55]),
    ],
)
def test_metrics_rank_differently(metric_name: str, expected_order: list[float]) -> None:
    # Each detection is identified by similarity1 so the resulting order is readable.
    detections = [detection(0.9, 0.9), detection(1.0, 0.2), detection(0.55, 0.5)]
    metric = get_metric(metric_name)
    ranked = sorted(detections, key=lambda d: metric.score(d.comparison), reverse=True)
    assert [d.comparison.similarity1 for d in ranked] == expected_order


def named_detection(
    name: str, similarity1: float, similarity2: float, token_overlap: int = DEFAULT_TOKEN_OVERLAP
) -> Detection:
    return Detection(
        cast("File", FakeFile(f"test/{name}")),
        cast("File", FakeFile(f"source/{name}")),
        comparison(similarity1, similarity2, token_overlap),
    )


def ranked_detections() -> list[Detection]:
    # Already ordered by `average`, which is how VenDetector reports them.
    return [
        named_detection("a", 0.9, 0.9),
        named_detection("b", 0.4, 0.3),
        named_detection("c", 0.3, 0.25),
    ]


def csv_rows(detections: list[Detection], metric_name: str, min_score: float) -> list[dict[str, str]]:
    out = io.StringIO()
    output_csv(detections, get_metric(metric_name), min_score, out)
    out.seek(0)
    return list(csv.DictReader(out))


def test_default_threshold_admits_only_genuinely_similar_files() -> None:
    # Regression: with a `sum`-scaled default, all three of these cleared `--min-score
    # 0.5`, because a 0.35-average pair scores 0.7 as a sum. Only "a" should survive.
    rows = csv_rows(ranked_detections(), DEFAULT_METRIC, get_metric(DEFAULT_METRIC).default_threshold)
    assert [row["Test File"] for row in rows] == ["test/a"]
    assert rows[0]["Score"] == "0.9000"
    assert rows[0]["Metric"] == "average"


def test_csv_reports_the_selected_metric() -> None:
    rows = csv_rows(ranked_detections(), "min", 0.3)
    assert [row["Metric"] for row in rows] == ["min", "min"]
    assert [row["Score"] for row in rows] == ["0.9000", "0.3000"]


def test_csv_stops_at_the_first_sub_threshold_detection() -> None:
    # `_scored` must stop rather than skip. Detections are pre-ranked, so a sub-threshold
    # score means every later one is below too, and skipping instead of stopping forces
    # `detect()` to walk git history for every remaining pair.
    #
    # The trailing high scorer cannot occur in a real ranked stream; it is here so that
    # stopping and skipping produce different output, which is what this pins.
    detections = [
        named_detection("a", 0.9, 0.9),
        named_detection("b", 0.4, 0.3),
        named_detection("c", 0.95, 0.95),
    ]
    rows = csv_rows(detections, "average", 0.5)
    assert [row["Test File"] for row in rows] == ["test/a"]


def test_csv_schema_is_stable_across_metrics() -> None:
    similarity = csv_rows(ranked_detections(), "average", 0.0)
    tokens = csv_rows(ranked_detections(), "token_overlap", 0.0)
    assert list(similarity[0].keys()) == list(tokens[0].keys())
    assert tokens[0]["Score"] == str(DEFAULT_TOKEN_OVERLAP)


def json_results(detections: list[Detection], metric_name: str, min_score: float) -> list[dict]:
    out = io.StringIO()
    output_json(detections, get_metric(metric_name), min_score, out)
    return json.loads(out.getvalue())


def test_json_schema_is_stable_across_metrics() -> None:
    similarity = json_results(ranked_detections(), "average", 0.0)
    tokens = json_results(ranked_detections(), "token_overlap", 0.0)
    assert set(similarity[0]) == set(tokens[0])
    # Both raw similarities and the raw token count are always present, so a consumer
    # never has to infer which metric produced the file.
    for result in (similarity[0], tokens[0]):
        assert {"metric", "score", "similarity_test", "similarity_source", "token_overlap"} <= set(result)


def test_json_reports_score_in_the_metric_units() -> None:
    assert json_results(ranked_detections(), "average", 0.0)[0]["score"] == pytest.approx(0.9)
    assert json_results(ranked_detections(), "token_overlap", 0.0)[0]["score"] == DEFAULT_TOKEN_OVERLAP


def test_rich_output_honors_the_selected_metric() -> None:
    # average is 0.85 and clears the threshold, but min is 0.7 and does not. The rich
    # format used to filter on a hardcoded average, so it reported this detection even
    # when the user asked to rank by `min`.
    detections = [named_detection("a", 1.0, 0.7)]
    out = io.StringIO()
    console = Console(file=out, width=200)
    output_rich(detections, console, get_metric("min"), 0.8)
    assert "test/a" not in out.getvalue()

    out_passing = io.StringIO()
    output_rich(detections, Console(file=out_passing, width=200), get_metric("max"), 0.8)
    assert "test/a" in out_passing.getvalue()


def test_cli_help_lists_every_registered_metric(monkeypatch) -> None:
    # Pins `--metric`'s choices to the registry, so adding a metric cannot leave the
    # CLI silently rejecting it.
    out = io.StringIO()
    monkeypatch.setattr("sys.argv", ["vendetect", "--help"])
    monkeypatch.setattr("sys.stdout", out)
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    help_text = out.getvalue()
    for name in METRICS:
        assert name in help_text


def test_json_serializes_numpy_scalars() -> None:
    # copydetect returns numpy scalars. np.float64 subclasses float so it serializes,
    # but np.int64 does not subclass int: emitting token overlaps or slice bounds
    # straight from a real Comparison used to abort `--format json` mid-write.
    numpy = pytest.importorskip("numpy")
    overlap = 135
    real_shaped = Comparison(
        token_overlap=numpy.int64(overlap),
        similarity1=numpy.float64(1.0),
        similarity2=numpy.float64(1.0),
        slices1=(Slice(numpy.int64(0), numpy.int64(426)),),
        slices2=(Slice(numpy.int64(0), numpy.int64(426)),),
    )
    detections = [
        Detection(
            cast("File", FakeFile("test/a")),
            cast("File", FakeFile("source/a")),
            real_shaped,
        )
    ]
    out = io.StringIO()
    output_json(detections, get_metric("token_overlap"), 0.0, out)
    result = json.loads(out.getvalue())[0]
    assert result["token_overlap"] == overlap
    assert result["slices"][0]["test_slice"] == {"start": 0, "end": 426}
