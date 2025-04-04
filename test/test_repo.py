import contextlib
import os
from pathlib import Path
from unittest import TestCase

from vendetect.repo import Repository, RepositoryCommit


REPO_ROOT = Path(__file__).parent.parent


class TestRepo(TestCase):
    def test_files(self):
        repo = Repository(REPO_ROOT)
        old_cwd = REPO_ROOT.cwd()
        try:
            os.chdir(REPO_ROOT)
            files = list(repo.files())
            self.assertEqual(len(files), len(frozenset(files)), "Paths are not unique")
            for expected in (
                ".gitignore",
                "LICENSE",
                "Makefile",
                "README.md"
            ):
                self.assertTrue(any(
                    p.relative_path.name == expected
                    for p in files
                ), f"Missing expected file: {expected}")
        finally:
            os.chdir(old_cwd)

    def test_prev_version(self):
        repo = Repository(REPO_ROOT)
        prev: RepositoryCommit | None = repo.previous_version(Path("./pyproject.toml"))
        if prev is None:
            pct = contextlib.nullcontext
        else:
            pct = prev
        with pct:
            while prev is not None:
                with prev:
                    pv = prev.previous_version(Path("./pyproject.toml"))
                if pv is None:
                    break
                prev = pv
        self.assertIsNotNone(prev)
        self.assertIsNone(prev.previous_version(Path("./README.md")))
        self.assertEqual("16385f50f79fe3aa44b6a8e4ef131626be700b38", prev.rev)
