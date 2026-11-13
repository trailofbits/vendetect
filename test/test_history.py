import os
import subprocess
from pathlib import Path

import pytest

from vendetect.detector import VenDetector
from vendetect.repo import GIT_PATH, File, Repository

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
    test_file = File(Path("sample.py"), test_repo)
    source_file = File(Path("sample.py"), source_repo)
    original = next(detector.compare((test_file,), (source_file,)))
    assert detector.metric.score(original.comparison) < 1.0

    probable = detector.find_probable_copy(original)

    assert detector.metric.score(probable.comparison) == pytest.approx(1.0)
    assert probable.test.repo.rev == test_revisions[0]
    assert probable.source.repo.rev == source_revisions[0]
    assert probable.test.relative_path == probable.source.relative_path == Path("sample.py")
