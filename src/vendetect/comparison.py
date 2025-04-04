from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from numpy import ndarray

C = TypeVar("C", bound="Slice")
F = TypeVar("F")


@dataclass(frozen=True, unsafe_hash=True, order=True)
class Slice:
    from_index: int
    to_index: int

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.from_index}, {self.to_index})"

    def __str__(self) -> str:
        return f"{self.from_index}–{self.to_index}"  # noqa: RUF001

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if index == 0:
            return self.from_index
        if index == 1:
            return self.to_index
        msg = f"Invalid index: {index}"
        raise IndexError(msg)

    def __iter__(self) -> Iterator[int]:
        yield self.from_index
        yield self.to_index

    @classmethod
    def from_ndarray(cls: type[C], array: ndarray) -> Iterable[C]:
        return (cls(from_pos, to_pos) for from_pos, to_pos in zip(*array, strict=False))


@dataclass(frozen=True, unsafe_hash=True)
class Comparison:
    """The result of a comparison between two files."""

    """Number of overlapping tokens between the two files"""
    token_overlap: int

    """number of overlapping tokens divided by the total number of tokens in the first file"""
    similarity1: float

    """number of overlapping tokens divided by the total number of tokens in the second file"""
    similarity2: float

    """tuple of Slices: locations of copied code in the unfiltered text.
    """
    slices1: tuple[Slice, ...]

    """tuple of Slices: locations of copied code in the unfiltered text.
    """
    slices2: tuple[Slice, ...]

    def __lt__(self, other: "Comparison") -> bool:
        if self.token_overlap > other.token_overlap:
            return True
        if self.token_overlap < other.token_overlap:
            return False
        oursim = self.similarity1 + self.similarity2
        theirsim = other.similarity1 + other.similarity2
        return oursim > theirsim


class Comparator(ABC, Generic[F]):
    @abstractmethod
    def fingerprint(self, path: Path) -> F:
        raise NotImplementedError

    @abstractmethod
    def compare(self, fp1: F, fp2: F) -> Comparison:
        raise NotImplementedError
