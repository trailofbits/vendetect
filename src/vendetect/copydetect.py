from pathlib import Path

from copydetect import CodeFingerprint, compare_files

from .comparison import Comparator, Comparison, Slice


class CopyDetectComparator(Comparator[CodeFingerprint]):
    def fingerprint(self, path: Path) -> CodeFingerprint:
        return CodeFingerprint(str(path), k=25, win_size=1)

    def compare(self, fp1: CodeFingerprint, fp2: CodeFingerprint) -> Comparison:
        overlap, (sim1, sim2), (slice1, slice2) = compare_files(fp1, fp2)
        return Comparison(
            overlap,
            sim1,
            sim2,
            tuple(Slice.from_ndarray(slice1)),
            tuple(Slice.from_ndarray(slice2)),
        )
