import types
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from heapq import heappop, heappush
from logging import getLogger
from typing import TypeVar

from pygments import lexer, lexers
from pygments.util import ClassNotFound

from .comparison import Comparator, Comparison, Slice
from .copydetect import CopyDetectComparator
from .repo import File, Repository

log = getLogger(__name__)
F = TypeVar("F")


def get_lexer_for_filename(filename: str) -> lexer.Lexer | None:
    try:
        return lexers.get_lexer_for_filename(filename)
    except ClassNotFound:
        return None


class Status:
    def on_compare(
        self,
        test_files: Iterable[File],  # noqa: ARG002
        source_files: Iterable[File],  # noqa: ARG002
    ) -> None | tuple[Iterable[File], Iterable[File]]:
        return None

    def compare_completed(self, test_files: Iterable[File], source_files: Iterable[File]) -> None:
        pass

    def update_num_comparisons(self, num: int) -> None:
        pass

    def update_compare_progress(self, file: File | None = None) -> None:
        pass


@dataclass(frozen=True, unsafe_hash=True)
class Source:
    file: File
    source_slices: tuple[Slice, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Source):
            return False
        return self.file == other.file and self.source_slices == other.source_slices

    def __lt__(self, other: "Source") -> bool:
        return self.file.relative_path < other.file.relative_path or (
            self.file == other.file and self.source_slices < other.source_slices
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.file!r}, {self.source_slices!r})"

    def __str__(self) -> str:
        file_str = str(self.file)
        if file_str.startswith("http"):
            slices = ";".join(f"L{s.from_index}-L{s.to_index}" for s in self.source_slices)
            return f"{self.file!s}#{slices}"
        slices = ";".join(f"{s.from_index}-{s.to_index}" for s in self.source_slices)
        return f"{self.file!s}:{slices}"


@dataclass(frozen=True, unsafe_hash=True)
class Detection:
    test: File
    source: File
    comparison: Comparison

    def __lt__(self, other: "Detection") -> bool:
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
    def __init__(
        self,
        comparator: Comparator[F] | None = None,
        status: Status | None = None,
        batch_size: int = 100,
        max_history_depth: int | None = None,
        *,
        incremental: bool = False,
    ):
        if comparator is None:
            comparator = CopyDetectComparator()
        self.comparator: Comparator[F] = comparator
        if status is None:
            self.status: Status = Status()
        else:
            self.status = status
        self.incremental = incremental
        self.batch_size = batch_size  # Process files in batches
        self.max_history_depth = (
            max_history_depth if max_history_depth is not None and max_history_depth >= 0 else None
        )  # Limit history traversal depth
        self._fingerprint_cache: dict[File, F] = {}  # Cache fingerprints

    @staticmethod
    def callback(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "VenDetector", *args: tuple, **kwargs: dict) -> Iterator | object:
            if not hasattr(self.status, f"on_{func.__name__}"):
                msg = (
                    f"{self.status.__class__.__name__}.on_{func.__name__} is not defined; required "
                    f"for @callback on {self.__class__.__name__}.{func.__name__}"
                )
                raise TypeError(msg)
            callback_func = getattr(self.status, f"on_{func.__name__}")
            modified_args = callback_func(*args, **kwargs)
            if modified_args is None:
                modified_args = args
            ret = func(self, *modified_args, **kwargs)
            is_generator = isinstance(ret, types.GeneratorType)
            if is_generator:
                yield from ret
            if hasattr(self.status, f"{func.__name__}_completed"):
                getattr(self.status, f"{func.__name__}_completed")(*modified_args, **kwargs)
            if not is_generator:
                return ret  # type: ignore

        return wrapper

    def _get_fingerprint(self, file: File) -> tuple[object, bool]:
        """Get fingerprint from cache or compute it, returning tuple of (fingerprint, is_cached)."""
        if file in self._fingerprint_cache:
            return self._fingerprint_cache[file], True

        try:
            fp = self.comparator.fingerprint(file.path)
            self._fingerprint_cache[file] = fp
            return fp, False  # noqa: TRY300
        except Exception as e:  # noqa: BLE001
            log.warning("Error fingerprinting %s: %s", str(file), str(e))
            return None, False

    @callback
    def compare(  # noqa: C901, PLR0912
        self, test_files: Iterable[File], source_files: Iterable[File]
    ) -> Iterator[Detection]:
        test_files: list[File] = list(test_files)
        source_files: list[File] = list(source_files)

        with ExitStack() as stack:
            for repo in {f.repo for f in test_files} | {f.repo for f in source_files}:
                stack.enter_context(repo)

            tf: list[File] = []
            sf: list[File] = []

            # Apply file filtering based on lexer availability
            for lst, files in ((tf, test_files), (sf, source_files)):
                for file in files:
                    if get_lexer_for_filename(file.path.name) is None:
                        log.warning(
                            "Ignoring %s because we do not have a lexer for its filetype",
                            str(file),
                        )
                    else:
                        lst.append(file)

            test_files = tf
            source_files = sf

            self.status.update_num_comparisons(len(test_files) * len(source_files))

            explored_sources: set[Source] = set()
            detections: list[Detection] = []

            # Process files in batches to allow incremental result reporting
            for i in range(0, len(test_files), self.batch_size):
                batch_test_files = test_files[i : i + self.batch_size]

                for test_file in batch_test_files:
                    self.status.update_compare_progress(test_file)

                    fp1, _ = self._get_fingerprint(test_file)
                    if fp1 is None:
                        continue

                    for source_file in source_files:
                        self.status.update_compare_progress()

                        fp2, _ = self._get_fingerprint(source_file)
                        if fp2 is None:
                            continue

                        cmp = self.comparator.compare(fp1, fp2)  # type: ignore
                        d = Detection(test_file, source_file, cmp)
                        heappush(detections, d)

                # Process accumulated detections for this batch
                if self.incremental and detections:
                    # Process detections incrementally
                    processed_batch = []
                    while detections:
                        d = heappop(detections)
                        if d.test_source in explored_sources:
                            # Skip if already found with better similarity
                            continue

                        processed_batch.append(d)
                        explored_sources.add(d.test_source)

                    # Yield all detections from this batch
                    for d in processed_batch:
                        yield d

            # Process remaining detections if not in incremental mode
            if not self.incremental:
                while detections:
                    d = heappop(detections)
                    if d.test_source in explored_sources:
                        continue
                    yield d
                    explored_sources.add(d.test_source)

    def find_probable_copy(self, detection: Detection, max_depth: int | None = None) -> Detection:
        """Find the most probable point in the test repo and source repo when the given detection was vendored.

        Args:
            detection: The detection to find the probable copy for
            max_depth: Maximum depth to traverse in history (None uses class default)

        """
        max_depth = max_depth if max_depth is not None and max_depth >= 0 else self.max_history_depth
        log.info(
            "Finding the most probable commit in which %s was copied… (max depth: %s)",
            str(detection.test_source),
            str(max_depth) if max_depth is not None else "∞",
        )

        best: Detection = detection
        to_test: list[tuple[Repository, Repository, int]] = [(detection.test_repo, detection.source_repo, 0)]
        history: set[tuple[Repository | None, Repository | None]] = set()

        while to_test:
            test_repo, source_repo, depth = to_test.pop()

            # Stop if we've reached max depth
            if max_depth is not None and 0 <= max_depth <= depth:
                continue

            if history:
                new_detections = tuple(self.compare((detection.test,), (detection.source,)))
                if new_detections:
                    best = min(best, *new_detections)

            pv = test_repo.previous_version(detection.test.relative_path)
            spv = source_repo.previous_version(detection.source.relative_path)

            if (pv, spv) in history:
                continue

            history.add((pv, spv))
            if pv is not None and spv is not None:
                to_test.append((pv, spv, depth + 1))
            if pv is not None:
                to_test.append((pv, source_repo, depth + 1))
            if spv is not None:
                to_test.append((test_repo, spv, depth + 1))

        return best

    def detect(
        self,
        test_repo: Repository,
        source_repo: Repository,
        file_filter: Callable[[File], bool] = lambda _: True,
        max_history_depth: int | None = None,
    ) -> Iterator[Detection]:
        """Detect vendored code between repositories.

        Args:
            test_repo: Repository to test
            source_repo: Repository to check for vendored code
            file_filter: Function to filter files
            max_history_depth: Maximum depth to traverse in history

        Returns:
            Iterator of Detection objects

        """
        test_files = [f for f in test_repo.files() if file_filter(f)]
        source_files = [f for f in source_repo.files() if file_filter(f)]

        log.info("Analyzing %d test files against %d source files", len(test_files), len(source_files))

        # Get detections by comparing files
        for d in self.compare(test_files, source_files):
            # Find probable copy with potentially limited history depth
            probable = self.find_probable_copy(d, max_depth=max_history_depth)
            yield probable
