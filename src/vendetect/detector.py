from dataclasses import dataclass
from heapq import heappush, heappop
from pathlib import Path

from .comparison import Comparator, Comparison
from .copydetect import CopyDetectComparator
from .repo import Repository


@dataclass(frozen=True, unsafe_hash=True)
class Detection:
    test_repo: Repository
    source_repo: Repository
    test_file: Path
    source_file: Path
    comparison: Comparison

    def __lt__(self, other: "Detection"):
        return self.comparison < other.comparison


class VenDetector:
    def __init__(self, test_repo: Repository, source_repo: Repository, comparator: Comparator | None = None):
        self.test_repo: Repository = test_repo
        self.source_repo: Repository = source_repo
        if comparator is None:
            comparator = CopyDetectComparator()
        self.comparator: Comparator = comparator

    def detect(self):
        history: set[Repository] = set()
        detections: list[Detection] = []

        for test_path in self.test_repo.files():
            fp1 = self.comparator.fingerprint(test_path)
            if test_path.is_absolute():
                rel_test_path = test_path.relative_to(self.test_repo.root_path)
            else:
                rel_test_path = test_path
            for source_path in self.source_repo.files():
                fp2 = self.comparator.fingerprint(source_path)
                if source_path.is_absolute():
                    rel_source_path = source_path.relative_to(self.source_repo.root_path)
                else:
                    rel_source_path = source_path
                d = self.comparator.compare(fp1, fp2)
                heappush(detections, Detection(self.test_repo, self.source_repo, rel_test_path, rel_source_path, d))
