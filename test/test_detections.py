from pathlib import Path
from unittest import TestCase

from vendetect.detector import VenDetector
from vendetect.repo import RemoteGitRepository, Repository


class TestVenDetect(TestCase):
    def test_detect(self):
        with Repository(Path(__file__).parent.parent) as test_repo, \
                RemoteGitRepository("https://github.com/trailofbits/cookiecutter-python") as source_repo:
            d = VenDetector(test_repo, source_repo)
            d.detect()
