from pathlib import Path
from unittest import TestCase

import pytest

from vendetect.detector import VenDetector
from vendetect.repo import GIT_PATH, RemoteGitRepository, Repository


class TestVenDetect(TestCase):
    @pytest.mark.skipif(GIT_PATH is None, reason="requires git")
    def test_detect(self):  # noqa: ANN201
        with (
            Repository(Path(__file__).parent.parent) as test_repo,
            RemoteGitRepository("https://github.com/trailofbits/cookiecutter-python") as source_repo,
        ):
            vend = VenDetector()
            self.assertTrue(
                any(
                    d.test.relative_path == Path("README.md")
                    and d.source.relative_path == Path("{{cookiecutter.project_slug}}") / "README.md"
                    for d in vend.detect(
                        test_repo,
                        source_repo,
                        file_filter=lambda f: f.relative_path.name in ("Makefile", "LICENSE", "README.md"),
                    )
                )
            )
