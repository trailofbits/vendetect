import types
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from heapq import heappop, heappush
from logging import getLogger

from numpy import ndarray
from pygments import lexer, lexers
from pygments.util import ClassNotFound

from .comparison import Comparator, Comparison
from .copydetect import CopyDetectComparator
from .repo import File, Repository

log = getLogger(__name__)


def get_lexer_for_filename(filename: str) -> lexer.Lexer | None:
    try:
        return lexers.get_lexer_for_filename(filename)
    except ClassNotFound:
        return None


class Status:
    def on_compare(self, test_files: Iterable[File], source_files: Iterable[File]):
        pass

    def compare_completed(self, test_files: Iterable[File], source_files: Iterable[File]):
        pass

    def update_num_comparisons(self, num: int):
        pass

    def update_compare_progress(self, file: File | None = None):
        pass


@dataclass(frozen=True)
class Source:
    file: File
    source_slices: ndarray

    def __hash__(self) -> int:
        return hash((self.file, self.source_slices.tobytes()))

    def __eq__(self, other):
        if not isinstance(other, Source):
            return False
        return self.file == other.file and (self.source_slices == other.source_slices).all()

    def __lt__(self, other: "Source"):
        return self.file.relative_path < other.file.relative_path or (
            self.file == other.file and (self.source_slices < other.source_slices).all()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file!r}, {self.source_slices!r})"

    def __str__(self):
        file_str = str(self.file)
        if file_str.startswith("http"):
            slices = ";".join(
                (f"L{from_pos}-L{to_pos}" for from_pos, to_pos in zip(*self.source_slices.tolist(), strict=False))
            )
            return f"{self.file!s}#{slices}"
        slices = ";".join(
            (f"{from_pos}-{to_pos}" for from_pos, to_pos in zip(*self.source_slices.tolist(), strict=False))
        )
        return f"{self.file!s}:{slices}"


@dataclass(frozen=True, unsafe_hash=True)
class Detection:
    test: File
    source: File
    comparison: Comparison

    def __lt__(self, other: "Detection"):
        return self.comparison < other.comparison

    @property
    def test_source(self) -> Source:
        return Source(self.test, self.comparison.slices1)

    @property
    def test_repo(self) -> Repository:
        return self.test.repo

    @property
    def source_repo(self) -> Repository:
        return self.source.repo


class VenDetector:
    def __init__(self, comparator: Comparator | None = None, status: Status | None = None):
        if comparator is None:
            comparator = CopyDetectComparator()
        self.comparator: Comparator = comparator
        if status is None:
            self.status: Status = Status()
        else:
            self.status = status

    @staticmethod
    def callback(func):
        @wraps(func)
        def wrapper(self: "VenDetector", *args, **kwargs):
            if not hasattr(self.status, f"on_{func.__name__}"):
                raise TypeError(
                    f"{self.status.__class__.__name__}.on_{func.__name__} is not defined; required for "
                    f"@callback on {self.__class__.__name__}.{func.__name__}"
                )
            callback_func = getattr(self.status, f"on_{func.__name__}")
            callback_func(*args, **kwargs)
            ret = func(self, *args, **kwargs)
            is_generator = isinstance(ret, types.GeneratorType)
            if is_generator:
                yield from ret
            if hasattr(self.status, f"{func.__name__}_completed"):
                getattr(self.status, f"{func.__name__}_completed")(*args, **kwargs)
            if not is_generator:
                return ret

        return wrapper

    @callback
    def compare(self, test_files: Iterable[File], source_files: Iterable[File]):
        test_files = tuple(test_files)
        source_files = tuple(source_files)

        with ExitStack() as stack:
            for repo in {f.repo for f in test_files} | {f.repo for f in source_files}:
                stack.enter_context(repo)

            tf = []
            sf = []

            for lst, files in ((tf, test_files), (sf, source_files)):
                for file in files:
                    if get_lexer_for_filename(file.path.name) is None:
                        log.warning(
                            f"Ignoring {file!s} because we do not have a lexer for its filetype"
                        )
                    else:
                        lst.append(file)

            test_files = tf
            source_files = sf

            self.status.update_num_comparisons(len(test_files) * len(source_files))

            explored_sources: set[Source] = set()
            detections: list[Detection] = []

            for test_file in test_files:
                self.status.update_compare_progress(test_file)

                try:
                    fp1 = self.comparator.fingerprint(test_file.path)
                except Exception as e:
                    log.warning(f"Error fingerprinting {test_file!s}: {e!s}")
                    continue

                for source_file in source_files:
                    self.status.update_compare_progress()

                    try:
                        fp2 = self.comparator.fingerprint(source_file.path)
                    except Exception as e:
                        log.warning(f"Error fingerprinting {source_file!s}: {e!s}")
                        continue
                    cmp = self.comparator.compare(fp1, fp2)
                    d = Detection(test_file, source_file, cmp)
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
        log.info(f"Finding the most probable commit in which {detection.test_source!s} was copied…")
        best: Detection = detection
        to_test: list[tuple[Repository, Repository]] = [
            (detection.test_repo, detection.source_repo)
        ]
        history: set[tuple[Repository | None, Repository | None]] = set()
        while to_test:
            test_repo, source_repo = to_test.pop()
            if history:
                new_detections = tuple(self.compare((detection.test,), (detection.source,)))
                if new_detections:
                    best = min(best, min(new_detections))
            pv = test_repo.previous_version(detection.test.relative_path)
            spv = source_repo.previous_version(detection.source.relative_path)
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

    def detect(
        self,
        test_repo: Repository,
        source_repo: Repository,
        file_filter: Callable[[File], bool] = lambda _: True,
    ) -> Iterator[Detection]:
        for d in self.compare(
            (f for f in test_repo.files() if file_filter(f)),
            (f for f in source_repo.files() if file_filter(f)),
        ):
            yield self.find_probable_copy(d)
