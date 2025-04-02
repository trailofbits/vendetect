from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from numpy import ndarray

F = TypeVar("F")


@dataclass(frozen=True, unsafe_hash=True)
class Comparison:
    """Number of overlapping tokens between the two files"""
    token_overlap: int

    """number of overlapping tokens divided by the total number of tokens in the first file"""
    similarity1: float

    """number of overlapping tokens divided by the total number of tokens in the second file"""
    similarity2: float

    """2xN int array: locations of copied code in the unfiltered text.
    Dimension 0 contains slice starts, dimension 1 contains slice ends.
    """
    slices1: ndarray

    """2xN int array: locations of copied code in the unfiltered text.
    Dimension 0 contains slice starts, dimension 1 contains slice ends.
    """
    slices2: ndarray

    def __lt__(self, other: "Comparison") -> bool:
        if self.token_overlap > other.token_overlap:
            return True
        elif self.token_overlap < other.token_overlap:
            return False
        oursim = self.similarity1 + self.similarity2
        theirsim = other.similarity1 + other.similarity2
        return oursim > theirsim


class Comparator(ABC, Generic[F]):
    @abstractmethod
    def fingerprint(self, path: Path) -> F:
        raise NotImplementedError()

    @abstractmethod
    def compare(self, fp1: F, fp2: F) -> Comparison:
        raise NotImplementedError()
