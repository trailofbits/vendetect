from difflib import ndiff
from enum import Enum
from typing import Iterable, Iterator, Self  # noqa: UP035

from .repo import File, Rounding


class DiffLineStatus(Enum):
    COPIED = "COPIED"
    DIFFERENT = "DIFFERENT"


class DiffLine:
    def __init__(self, left: str | None, status: DiffLineStatus, right: str | None, left_line: int, right_line: int):
        self.left: str | None = left
        self.status: DiffLineStatus = status
        self.right: str | None = right
        self.left_line: int = left_line
        self.right_line: int = right_line


class CollapsedDiffLine(DiffLine):
    def __init__(self, left_start_line: int, right_start_line: int, num_identical_lines: int):
        self.left_start_line: int = left_start_line
        self.right_start_line: int = right_start_line
        self.num_identical_lines: int = num_identical_lines
        self.left: str
        self.right: str
        super().__init__(
            f"<{self.num_identical_lines} identical lines starting on line {self.left_start_line}>",
            DiffLineStatus.COPIED,
            f"<{self.num_identical_lines} identical lines starting on line {self.right_start_line}>",
            left_line=left_start_line,
            right_line=right_start_line,
        )


class Document:
    def __init__(self, lines: Iterable[str], start_line: int = 1):
        self.lines: tuple[str, ...] = tuple(lines)
        self.start_line: int = start_line
        self._line_start_offsets: list[int] = [0]
        for line in self.lines[:-1]:
            self._line_start_offsets.append(self._line_start_offsets[-1] + len(line))

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[str]:
        return iter(self.lines)

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        return self.lines[index]

    def get_line(self, byte_offset: int, rounding: Rounding = Rounding.DOWN, min_line: int = 0) -> int:
        if byte_offset < 0:
            byte_offset = self._line_start_offsets[-1] + len(self.lines[-1]) + byte_offset
        if rounding == Rounding.DOWN:
            start = max(min_line, 0)
            while start + 1 < len(self._line_start_offsets) and self._line_start_offsets[start + 1] < byte_offset:
                start += 1
            return start
        # round up
        end = max(min_line, 0)
        while end < len(self._line_start_offsets) and self._line_start_offsets[end] < byte_offset:
            end += 1
        return end

    @classmethod
    def from_str(cls, text: str) -> Self:
        return cls(text.splitlines(keepends=True))

    @classmethod
    def from_file(cls, file: File) -> Self:
        with file.repo:
            return cls.from_str(file.path.read_text())


class DiffContext:
    def __init__(
        self,
        test_start_line: int = 0,
        test_end_line: int = -1,
        source_start_line: int = 0,
        source_end_line: int = -1,
        collapse_identical_lines_threshold: int = 10,
    ):
        self.test_start_line: int = test_start_line
        self.test_end_line: int = test_end_line
        self.source_start_line: int = source_start_line
        self.source_end_line: int = source_end_line
        self.collapse_identical_lines_threshold: int = collapse_identical_lines_threshold
        self.rows: list[DiffLine] = []
        self.same_lines: int = 0
        self.test_line: int = 0
        self.source_line: int = 0

    def add_test_row(self, code: str) -> None:
        insertion_point: DiffLine | None = None
        for line in reversed(self.rows):
            if line.left is not None:
                break
            insertion_point = line
        if insertion_point is None:
            self.rows.append(DiffLine(code, DiffLineStatus.DIFFERENT, None, self.test_line, -1))
        else:
            insertion_point.left = code
            insertion_point.left_line = self.test_line
        self.same_lines = 0
        self.test_line += 1

    def add_source_row(self, code: str) -> None:
        insertion_point: DiffLine | None = None
        for line in reversed(self.rows):
            if line.right is not None:
                break
            insertion_point = line
        if insertion_point is None:
            self.rows.append(DiffLine(None, DiffLineStatus.DIFFERENT, code, -1, self.source_line))
        else:
            insertion_point.right = code
            insertion_point.right_line = self.source_line
        self.same_lines = 0
        self.source_line += 1

    def add_identical_row(self, num_identical: int) -> Iterator[DiffLine]:
        for _ in range(num_identical):
            self.rows.pop()
        yield from self.rows
        self.rows.clear()
        yield CollapsedDiffLine(self.test_line - num_identical, self.source_line - num_identical, num_identical)


class Differ:
    def __init__(
        self,
        test: Document,
        source: Document,
    ):
        self.test: Document = test
        self.source: Document = source

    def diff(self, context: DiffContext | None = None) -> Iterator[DiffLine]:
        if context is None:
            context = DiffContext()
        test_slice_content = self.test[max(0, context.test_start_line) : context.test_end_line]
        source_slice_content = self.source[max(0, context.source_start_line) : context.source_end_line]

        context.test_line = max(1, context.test_start_line + 1)
        context.source_line = max(1, context.source_start_line + 1)
        context.rows = []
        context.same_lines = 0

        for diff_line in ndiff(test_slice_content, source_slice_content):
            if diff_line.startswith("  "):
                context.same_lines += 1
                copied_str = diff_line[2:].rstrip()
                context.rows.append(
                    DiffLine(
                        copied_str,
                        DiffLineStatus.COPIED,
                        copied_str,
                        context.test_line,
                        context.source_line,
                    )
                )
                context.test_line += 1
                context.source_line += 1
            else:
                if diff_line[:2] in ("- ", "+ ") and context.same_lines >= context.collapse_identical_lines_threshold:
                    yield from context.add_identical_row(context.same_lines)
                if diff_line.startswith("- "):
                    context.add_test_row(diff_line[2:].rstrip())
                elif diff_line.startswith("+ "):
                    context.add_source_row(diff_line[2:].rstrip())

        if context.same_lines >= context.collapse_identical_lines_threshold:
            yield from context.add_identical_row(context.same_lines)

        yield from context.rows

    def diff_from_offsets(
        self,
        test_start_offset: int = 0,
        test_end_offset: int = -1,
        source_start_offset: int = 0,
        source_end_offset: int = -1,
        collapse_identical_lines_threshold: int = 10,
    ) -> Iterator[DiffLine]:
        test_start = self.test.get_line(test_start_offset, rounding=Rounding.DOWN)
        test_end = self.test.get_line(test_end_offset, rounding=Rounding.UP, min_line=test_start + 1)
        source_start = self.source.get_line(source_start_offset, rounding=Rounding.DOWN)
        source_end = self.source.get_line(source_end_offset, rounding=Rounding.UP, min_line=source_start + 1)
        return self.diff(
            DiffContext(
                test_start_line=test_start,
                test_end_line=test_end,
                source_start_line=source_start,
                source_end_line=source_end,
                collapse_identical_lines_threshold=collapse_identical_lines_threshold,
            )
        )
