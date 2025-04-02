import os
from pathlib import Path
from unittest import TestCase

from vendetect.repo import Repository


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
                    p.name == expected
                    for p in files
                ), f"Missing expected file: {expected}")
        finally:
            os.chdir(old_cwd)
