import os
import subprocess
from pathlib import Path
from typing import TypeVar

import pytest

from vendetect.comparison import Comparator, Comparison
from vendetect.detector import Detection, VenDetector
from vendetect.metrics import get_metric
from vendetect.repo import GIT_PATH, File, Repository

Fingerprint = TypeVar("Fingerprint")

pytestmark = pytest.mark.skipif(GIT_PATH is None, reason="requires git")

MATCHING_CODE = """\
def transform(values):
    result = []
    for index, value in enumerate(values):
        if value > 0:
            result.append(value * index + len(values))
        else:
            result.append(abs(value) - index)
    return sorted(result)
"""
TEST_HEAD_CODE = """\
class Collector:
    def __init__(self):
        self.items = {}

    def add(self, label, amount):
        if label not in self.items:
            self.items[label] = 0
        self.items[label] += amount
        return self.items[label]
"""
SOURCE_HEAD_CODE = """\
import re

PATTERN = re.compile(r"[a-z]+")

def parse_lines(document):
    for line in document.splitlines():
        fields = PATTERN.findall(line)
        if len(fields) > 2:
            yield tuple(fields)
"""


class WordComparator(Comparator[frozenset[str]]):
    def fingerprint(self, path: Path) -> frozenset[str]:
        tokens = frozenset(path.read_text(encoding="utf-8").split())
        if not tokens:
            msg = f"Cannot fingerprint empty fixture {path}"
            raise ValueError(msg)
        return tokens

    def compare(self, fp1: frozenset[str], fp2: frozenset[str]) -> Comparison:
        overlap = len(fp1 & fp2)
        return Comparison(overlap, overlap / len(fp1), overlap / len(fp2), (), ())


def git(path: Path, *args: str) -> str:
    assert GIT_PATH is not None
    env = dict(
        os.environ,
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_DATE="2000-01-01T00:00:00+00:00",
        GIT_COMMITTER_DATE="2000-01-01T00:00:00+00:00",
    )
    return subprocess.check_output(  # noqa: S603 - fixture commands use fixed Git arguments.
        [
            GIT_PATH,
            "-c",
            "user.name=Vendetect tests",
            "-c",
            "user.email=vendetect-tests@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=path,
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def create_repository(
    tmp_path: Path,
    name: str,
    versions: tuple[str, ...],
) -> tuple[Repository, list[str]]:
    path = tmp_path / name
    path.mkdir()
    git(path, "init", "--initial-branch=main")
    revisions = []
    for index, content in enumerate(versions):
        (path / "sample.py").write_text(content, encoding="utf-8")
        git(path, "add", "sample.py")
        git(path, "commit", "-m", f"{name} revision {index}")
        revisions.append(git(path, "rev-parse", "HEAD"))
    return Repository(path), revisions


def compare_heads(
    detector: VenDetector[Fingerprint],
    test_repo: Repository,
    source_repo: Repository,
) -> Detection:
    test_file = File(Path("sample.py"), test_repo)
    source_file = File(Path("sample.py"), source_repo)
    return next(detector.compare((test_file,), (source_file,)))


@pytest.mark.parametrize(
    ("test_versions", "source_versions"),
    [
        ((MATCHING_CODE, TEST_HEAD_CODE), (MATCHING_CODE,)),
        ((MATCHING_CODE,), (MATCHING_CODE, SOURCE_HEAD_CODE)),
        ((MATCHING_CODE, TEST_HEAD_CODE), (MATCHING_CODE, SOURCE_HEAD_CODE)),
    ],
    ids=["test-history", "source-history", "both-histories"],
)
def test_probable_copy_uses_historical_files(
    tmp_path: Path, test_versions: tuple[str, ...], source_versions: tuple[str, ...]
) -> None:
    test_repo, test_revisions = create_repository(tmp_path, "test", test_versions)
    source_repo, source_revisions = create_repository(tmp_path, "source", source_versions)
    detector = VenDetector()
    original = compare_heads(detector, test_repo, source_repo)
    assert detector.metric.score(original.comparison) < 1.0

    probable = detector.find_probable_copy(original)

    assert detector.metric.score(probable.comparison) == pytest.approx(1.0)
    assert probable.test.repo.rev == test_revisions[0]
    assert probable.source.repo.rev == source_revisions[0]
    assert probable.test.relative_path == probable.source.relative_path == Path("sample.py")


@pytest.mark.parametrize(
    ("metric_name", "expected_index", "expected_score"),
    [("average", -1, 0.75), ("token_overlap", 0, 4.0)],
)
def test_probable_copy_uses_selected_metric(
    tmp_path: Path, metric_name: str, expected_index: int, expected_score: float
) -> None:
    test_repo, _ = create_repository(tmp_path, "test", ("a b c d",))
    source_versions = ("a b c d e f g h i j k l", "a b")
    source_repo, revisions = create_repository(tmp_path, "source", source_versions)
    detector = VenDetector(comparator=WordComparator(), metric=get_metric(metric_name))
    original = compare_heads(detector, test_repo, source_repo)

    probable = detector.find_probable_copy(original)

    assert probable.source.repo.rev == revisions[expected_index]
    assert detector.metric.score(probable.comparison) == pytest.approx(expected_score)


def test_probable_copy_keeps_original_on_equal_scores(tmp_path: Path) -> None:
    test_repo, _ = create_repository(tmp_path, "test", ("a b",))
    source_repo, _ = create_repository(tmp_path, "source", ("a c", "b c"))
    detector = VenDetector(comparator=WordComparator())
    original = compare_heads(detector, test_repo, source_repo)

    assert detector.find_probable_copy(original) is original


def test_probable_copy_without_history_keeps_original(tmp_path: Path) -> None:
    test_repo, _ = create_repository(tmp_path, "test", (MATCHING_CODE,))
    source_repo, _ = create_repository(tmp_path, "source", (SOURCE_HEAD_CODE,))
    detector = VenDetector()
    original = compare_heads(detector, test_repo, source_repo)

    assert detector.find_probable_copy(original) is original


@pytest.mark.parametrize(
    ("default_depth", "max_depth", "expected_index"),
    [
        (None, 0, 2),
        (None, 1, 2),
        (None, 2, 1),
        (None, 3, 0),
        (None, None, 0),
        (2, None, 1),
        (2, 0, 2),
        (1, 3, 0),
        (2, -1, 1),
    ],
)
def test_probable_copy_respects_history_depth(
    tmp_path: Path, default_depth: int | None, max_depth: int | None, expected_index: int
) -> None:
    test_repo, _ = create_repository(tmp_path, "test", ("a b c d",))
    source_versions = ("a b c d", "a b x y", "w x y z")
    source_repo, revisions = create_repository(tmp_path, "source", source_versions)
    detector = VenDetector(comparator=WordComparator(), max_history_depth=default_depth)
    original = compare_heads(detector, test_repo, source_repo)

    probable = detector.find_probable_copy(original, max_depth=max_depth)

    assert probable.source.repo.rev == revisions[expected_index]


def test_history_does_not_change_head_comparisons(tmp_path: Path) -> None:
    test_repo, _ = create_repository(tmp_path, "test", ("a b c d",))
    source_repo, _ = create_repository(tmp_path, "source", ("a b c d", "w x y z"))
    detector = VenDetector(comparator=WordComparator())
    original = compare_heads(detector, test_repo, source_repo)

    probable = detector.find_probable_copy(original)
    current = compare_heads(detector, test_repo, source_repo)

    assert detector.metric.score(probable.comparison) > detector.metric.score(current.comparison)
    assert current.comparison == original.comparison
    assert current.source.repo.rev == source_repo.rev
    with probable.source.repo:
        assert probable.source.path.read_text(encoding="utf-8") == "a b c d"


def test_unavailable_historical_fingerprint_keeps_original(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_repo, _ = create_repository(tmp_path, "test", ("a b",))
    source_repo, _ = create_repository(tmp_path, "source", ("", "a b"))
    detector = VenDetector(comparator=WordComparator())
    original = compare_heads(detector, test_repo, source_repo)

    probable = detector.find_probable_copy(original)

    assert probable is original
    assert "Cannot fingerprint empty fixture" in caplog.text
