from pathlib import Path
from unittest import TestCase

from vendetect.detector import VenDetector
from vendetect.repo import RemoteGitRepository, Repository


class TestVenDetect(TestCase):
    def test_detect(self):
        with Repository(Path(__file__).parent.parent) as test_repo, \
                RemoteGitRepository("https://github.com/trailofbits/cookiecutter-python") as source_repo:
            vend = VenDetector()
            for d in vend.detect(test_repo, source_repo):
                print(f"{d.test_repo!s}/{d.test_file!s} <-- {d.source_repo!s}/{d.source_file!s}")
