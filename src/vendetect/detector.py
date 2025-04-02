from dataclasses import dataclass
from heapq import heappush, heappop
from pathlib import Path
from typing import Iterable, Iterator

from numpy import ndarray

from .comparison import Comparator, Comparison
from .copydetect import CopyDetectComparator
from .repo import Repository


@dataclass(frozen=True)
class Source:
    file: Path
    source_slices: ndarray

    def __hash__(self) -> int:
        return hash((self.file, self.source_slices.tobytes()))

    def __lt__(self, other: "Source"):
        return self.file < other.file or (self.file == other.file and self.source_slices < other.source_slices)


@dataclass(frozen=True, unsafe_hash=True)
class Detection:
    test_repo: Repository
    source_repo: Repository
    test_file: Path
    source_file: Path
    comparison: Comparison

    def __lt__(self, other: "Detection"):
        return self.comparison < other.comparison

    @property
    def test_source(self) -> Source:
        return Source(self.test_file, self.comparison.slices1)


class VenDetector:
    def __init__(self, comparator: Comparator | None = None):
        if comparator is None:
            comparator = CopyDetectComparator()
        self.comparator: Comparator = comparator

    def compare(self, test_repo: Repository, source_repo: Repository, test_paths: Iterable[Path] = (),
                source_paths: Iterable[Path] = ()) -> Iterator[Detection]:
        explored_sources: set[Source] = set()
        detections: list[Detection] = []

        test_paths = tuple(test_paths)
        if not test_paths:
            test_paths = test_repo.files()

        source_paths = tuple(source_paths)

        for test_path in test_paths:
            try:
                fp1 = self.comparator.fingerprint(test_path)
            except Exception as e:
                print(e)
                continue
            if test_path.is_absolute():
                rel_test_path = test_path.relative_to(test_repo.root_path)
            else:
                rel_test_path = test_path

            if source_paths:
                sp = source_paths
            else:
                sp = source_repo.files()
            for source_path in sp:
                try:
                    fp2 = self.comparator.fingerprint(source_path)
                except Exception as e:
                    print(e)
                    continue
                if source_path.is_absolute():
                    rel_source_path = source_path.relative_to(source_repo.root_path)
                else:
                    rel_source_path = source_path
                cmp = self.comparator.compare(fp1, fp2)
                d = Detection(test_repo, source_repo, rel_test_path, rel_source_path, cmp)
                heappush(detections, d)

        while detections:
            d = heappop(detections)

            if d.test_source in explored_sources:
                # we already found a different provenance for this source slice with a better similarity
                continue

            yield d

            explored_sources.add(d.test_source)

    def find_probable_copy(self, detection: Detection) -> Detection:
        """Finds the most probable point in the test repo and source repo when the given detection was vendored"""
        best: Detection = detection
        to_test: list[tuple[Repository, Repository]] = [(detection.test_repo, detection.source_repo)]
        history: set[tuple[Repository | None, Repository | None]] = set()
        while to_test:
            test_repo, source_repo = to_test.pop()
            if history:
                new_detections = tuple(self.compare(test_repo, source_repo, test_paths=(detection.test_file,),
                                       source_paths=(detection.source_file,)))
                if new_detections:
                    best = min(best, min(new_detections))
            pv = test_repo.previous_version(detection.test_file)
            spv = source_repo.previous_version(detection.source_file)
            if (pv, spv) in history:
                continue
            history.add((pv, spv))
            if pv is not None and spv is not None:
                to_test.append((pv, spv))
            if pv is not None:
                to_test.append((pv, source_repo))
            if spv is not None:
                to_test.append((spv, source_repo))
        return best

    def detect(self, test_repo: Repository, source_repo: Repository) -> Iterator[Detection]:
        for d in self.compare(test_repo, source_repo):
            yield self.find_probable_copy(d)
