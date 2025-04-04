import os
from pathlib import Path
from unittest import TestCase

import pytest

from vendetect.repo import GIT_PATH, Repository

REPO_ROOT = Path(__file__).parent.parent


class TestRepo(TestCase):
    def test_files(self):  # noqa: ANN201
        repo = Repository(REPO_ROOT)
        old_cwd = REPO_ROOT.cwd()
        try:
            os.chdir(REPO_ROOT)
            files = list(repo.files())
            self.assertEqual(len(files), len(frozenset(files)), "Paths are not unique")
            for expected in (".gitignore", "LICENSE", "Makefile", "README.md"):
                self.assertTrue(
                    any(p.relative_path.name == expected for p in files),
                    f"Missing expected file: {expected}",
                )
        finally:
            os.chdir(old_cwd)

    @pytest.mark.skipif(GIT_PATH is None, reason="requires git")
    def test_prev_version(self):  # noqa: ANN201
        repo: Repository | None = Repository(REPO_ROOT)
        with repo:
            prev: Repository | None = None
            while repo:
                repo = repo.previous_version(Path("./pyproject.toml"))
                if repo is None:
                    break
                prev = repo
        self.assertIsNotNone(prev)
        self.assertIsNone(prev.previous_version(Path("./README.md")))
        self.assertEqual("16385f50f79fe3aa44b6a8e4ef131626be700b38", prev.rev)
