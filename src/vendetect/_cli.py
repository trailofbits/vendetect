import argparse
import csv
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from rich import traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .detector import Detection, Status, VenDetector, get_lexer_for_filename
from .diffing import CollapsedDiffLine, DiffContext, Differ, Document, DiffLineStatus
from .errors import VendetectError
from .repo import File, Repository

logger = logging.getLogger(__name__)


class RichStatus(Status):
    def __init__(self, console: Console):
        self.console: Console = console
        self.progress: Progress | None = None
        self.compare_tasks: list[TaskID] = []

    def __enter__(self):  # type: ignore
        self.progress = Progress(console=self.console, transient=True)
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.progress.__exit__(exc_type, exc_val, exc_tb)  # type: ignore

    def on_compare(self, test_files: Iterable[File], source_files: Iterable[File]) -> None:  # noqa: ARG002
        self.compare_tasks.append(
            self.progress.add_task(":magnifying_glass_tilted_right: comparing…")  # type: ignore
        )

    def compare_completed(self, test_files: Iterable[File], source_files: Iterable[File]) -> None:  # noqa: ARG002
        self.progress.remove_task(self.compare_tasks[-1])  # type: ignore
        self.compare_tasks.pop()

    def update_num_comparisons(self, num: int) -> None:
        self.progress.update(self.compare_tasks[-1], total=num)  # type: ignore

    def update_compare_progress(self, file: File | None = None) -> None:
        self.progress.update(self.compare_tasks[-1], advance=1)  # type: ignore
        if file is not None:
            self.progress.update(  # type: ignore
                self.compare_tasks[-1],
                description=f":magnifying_glass_tilted_right: {file.relative_path.name!s}",
            )


def output_csv(detections: Iterable[Detection], min_similarity: float = 0.5, output_file: TextIO | None = None) -> None:
    output = output_file if output_file else sys.stdout
    csv_writer = csv.writer(output)
    # Write header
    csv_writer.writerow(
        [
            "Test File",
            "Source File",
            "Test Slice Start",
            "Test Slice End",
            "Source Slice Start",
            "Source Slice End",
            "Similarity",
        ]
    )

    for d in detections:
        # Calculate overall similarity (average of both similarities)
        avg_similarity = (d.comparison.similarity1 + d.comparison.similarity2) / 2

        if avg_similarity < min_similarity:
            break

        # Get slices
        test_slices = d.comparison.slices1
        source_slices = d.comparison.slices2

        for (test_start, test_end), (source_start, source_end) in zip(test_slices, source_slices, strict=False):
            # Write one row per matched slice
            csv_writer.writerow(
                [
                    f"{d.test.relative_path!s}",
                    f"{d.source.relative_path!s}",
                    test_start,
                    test_end,
                    source_start,
                    source_end,
                    f"{avg_similarity:.4f}",
                ]
            )


def output_json(
    detections: Iterable[Detection], min_similarity: float = 0.5, output_file: TextIO | None = None
) -> None:
    results = []
    output = output_file if output_file else sys.stdout

    for d in detections:
        # Calculate overall similarity (average of both similarities)
        avg_similarity = (d.comparison.similarity1 + d.comparison.similarity2) / 2

        if avg_similarity < min_similarity:
            break

        # Get slices
        test_slices = d.comparison.slices1
        source_slices = d.comparison.slices2

        # Prepare slices data
        slices_data = []
        for (test_slice_start, test_slice_end), (source_slice_start, source_slice_end) in zip(
            test_slices, source_slices, strict=False
        ):
            slices_data.append(
                {
                    "test_slice": {"start": test_slice_start, "end": test_slice_end},
                    "source_slice": {"start": source_slice_start, "end": source_slice_end},
                }
            )

        # Create detection data
        detection_data = {
            "test_file": f"{d.test.relative_path!s}",
            "source_file": f"{d.source.relative_path!s}",
            "similarity": round(avg_similarity, 4),
            "similarity_test": round(d.comparison.similarity1, 4),
            "similarity_source": round(d.comparison.similarity2, 4),
            "slices": slices_data,
        }

        results.append(detection_data)

    # Output JSON
    json.dump(results, output, indent=2)


def output_rich(
    detections: Iterable[Detection],
    console: Console,
    min_similarity: float = 0.5,
    output_file: TextIO | None = None,
    collapse_identical_lines_threshold: int = 10,
) -> None:
    # If an output file is specified, create a new Console for it
    file_console = Console(file=output_file) if output_file else console

    for d in detections:
        # Create a table for the detection results
        table = Table(title="Vendoring Detection", expand=True)
        table.add_column("Test File", style="cyan")
        table.add_column("Source File", style="green")
        table.add_column("Similarity", justify="right", style="yellow")

        # Calculate overall similarity (average of both similarities)
        avg_similarity = (d.comparison.similarity1 + d.comparison.similarity2) / 2

        if avg_similarity < min_similarity:
            break

        similarity_str = f"{avg_similarity:.1%}"

        # Add the main row with test and source files
        table.add_row(f"{d.test.relative_path!s}", f"{d.source.relative_path!s}", similarity_str)

        # Read file content for both test and source files
        def read_file_content(file: File) -> str:
            with file.repo:
                return file.path.read_text()

        try:
            test_doc = Document.from_file(d.test)
            source_doc = Document.from_file(d.source)

            # Create a side-by-side view of the detected slices
            test_slices = d.comparison.slices1
            source_slices = d.comparison.slices2

            match_table = Table()
            match_table.add_column(f"{d.test.relative_path.name!s}", style="cyan")
            match_table.add_column(" ")
            match_table.add_column(f"{d.source.relative_path.name!s}", style="green")
            first = True

            test_lexer = get_lexer_for_filename(d.test.relative_path.name)
            source_lexer = get_lexer_for_filename(d.source.relative_path.name)

            for (test_start_offset, test_end_offset), (source_start_offset, source_end_offset) in zip(
                test_slices, source_slices, strict=False
            ):
                if first:
                    first = False
                else:
                    match_table.add_row(Text("  ⋮", style="dim"), Text("⋮", style="dim"), Text("  ⋮", style="dim"))

                for diff_line in Differ(test=test_doc, source=source_doc).diff(
                    context=DiffContext.from_offsets(
                        test=test_doc,
                        source=source_doc,
                        test_start_offset=test_start_offset,
                        test_end_offset=test_end_offset,
                        source_start_offset=source_start_offset,
                        source_end_offset=source_end_offset,
                        collapse_identical_lines_threshold=collapse_identical_lines_threshold,
                    )
                ):
                    if isinstance(diff_line, CollapsedDiffLine):
                        test_msg = diff_line.left
                        source_msg = diff_line.right
                        match_table.add_row(
                            Text(f"        {' ' * (len(test_msg) // 2)}⋮", "red reverse bold"),
                            Text("←", style="red reverse bold"),
                            Text(f"        {' ' * (len(source_msg) // 2)}⋮", "red reverse bold"),
                        )
                        match_table.add_row(
                            Text(f"        {test_msg}", style="red reverse bold"),
                            Text("←", style="red reverse bold"),
                            Text(f"        {source_msg}", style="red reverse bold"),
                        )
                        match_table.add_row(
                            Text(f"        {' ' * (len(test_msg) // 2)}⋮", "red reverse bold"),
                            Text("←", style="red reverse bold"),
                            Text(f"        {' ' * (len(source_msg) // 2)}⋮", "red reverse bold"),
                        )
                    else:
                        if diff_line.status == DiffLineStatus.COPIED:
                            status_col = Text("←", style="red reverse bold")
                        else:
                            status_col = Text("✓", style="green reverse")
                        if diff_line.left is None:
                            left = Text("")
                        else:
                            left = Syntax(
                                diff_line.left, lexer=test_lexer, line_numbers=True, start_line=diff_line.left_line
                            )
                        if diff_line.right is None:
                            right = Text("")
                        else:
                            right = Syntax(
                                diff_line.right, lexer=source_lexer, line_numbers=True, start_line=diff_line.right_line
                            )
                        match_table.add_row(left, status_col, right)

            if not first:
                context_panel = Panel.fit(
                    match_table,
                    title="Vendored Code",
                    border_style="red",
                    title_align="left",
                    padding=(1, 1),
                )

                file_console.print()
                file_console.print(context_panel)
                file_console.print()
            else:
                # Just print the table if there are no specific slices
                file_console.print(table)
                file_console.print()
        except Exception as e:  # noqa: BLE001
            # Fallback to basic output if there's an error reading or processing files
            file_console.print(table)
            file_console.print(f"[red]Error displaying code: {e}[/red]")
            file_console.print()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser(prog="vendetect")

    parser.add_argument("TEST_REPO", type=str, help="path or URL to the test repository")
    parser.add_argument("SOURCE_REPO", type=str, help="path or URL to the source repository")
    parser.add_argument(
        "--test-subdir",
        "-ts",
        type=str,
        default=None,
        help="relative path to the subdirectory in TEST_REPO in which to test (optional, for use when TEST_REPO is a "
        "remote git repository URL)",
    )
    parser.add_argument(
        "--source-subdir",
        "-ss",
        type=str,
        default=None,
        help="relative path to the subdirectory in SOURCE_REPO in which to test (optional, for use when SOURCE_REPO is "
        "a remote git repository URL)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["rich", "csv", "json"],
        default="rich",
        help="output format (default: rich)",
    )
    parser.add_argument("--output", type=str, help="output file path (default: stdout)")
    parser.add_argument("--force", action="store_true", help="force overwrite of existing output file")
    parser.add_argument(
        "--type",
        "-t",
        action="append",
        dest="file_types",
        help="file extension to consider (can be used multiple times, e.g. `-t py -t c`)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="the minimum similarity threshold to output a match (range: 0.0-1.0, default: 0.5)",
    )

    # Performance optimization options
    perf_section = parser.add_argument_group(title="performance optimizations")
    perf_section.add_argument(
        "--incremental",
        action="store_true",
        help="enable incremental result reporting (processes and outputs files in batches)",
    )
    perf_section.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="number of files to process in each batch when using incremental mode (default: 100)",
    )
    perf_section.add_argument(
        "--max-history-depth",
        type=int,
        default=-1,
        help="maximum depth to traverse in commit history (default: -1, -1 = entire history, 0 = no history traversal)",
    )

    log_section = parser.add_argument_group(title="logging")
    log_group = log_section.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=[logging.getLevelName(x) for x in range(1, 101) if not logging.getLevelName(x).startswith("Level")],
        help="sets the log level for deptective (default=INFO)",
    )
    log_group.add_argument("--debug", action="store_true", help="equivalent to `--log-level=DEBUG`")
    log_group.add_argument(
        "--quiet",
        action="store_true",
        help="equivalent to `--log-level=CRITICAL`",
    )

    args = parser.parse_args()

    if args.debug:
        numeric_log_level = logging.DEBUG
    elif args.quiet:
        numeric_log_level = logging.CRITICAL
    else:
        log_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(log_level, int):
            sys.stderr.write(f"Invalid log level: {args.log_level}\n")
            sys.exit(1)
        numeric_log_level = log_level

    console = Console(log_path=False, file=sys.stderr)

    logging.basicConfig(
        level=numeric_log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )

    traceback.install(show_locals=True)

    # Check for output file existence if --force is not specified
    output_file = None
    if args.output:
        if not args.force and Path(args.output).exists():
            sys.stderr.write(f"Error: Output file {args.output} already exists. Use --force to overwrite.\n")
            sys.exit(1)
        try:
            output_file = Path(args.output).open("w")  # noqa: SIM115
        except OSError as e:
            sys.stderr.write(f"Error: Could not open output file {args.output} for writing: {e}\n")
            sys.exit(1)

    try:
        with (
            Repository.load(args.TEST_REPO, args.test_subdir) as test_repo,
            Repository.load(args.SOURCE_REPO, args.source_subdir) as source_repo,
            RichStatus(console) as status,
        ):
            # Initialize detector with optimization options
            vend = VenDetector(
                status=status,
                incremental=args.incremental,
                batch_size=args.batch_size,
                max_history_depth=args.max_history_depth,
            )

            # Get detections
            if not args.file_types:

                def file_filter(file: File) -> bool:  # noqa: ARG001
                    return True
            else:

                def file_filter(file: File) -> bool:
                    suffix = file.relative_path.suffix
                    if suffix in args.file_types or (suffix.startswith(".") and suffix[1:] in args.file_types):
                        return True
                    suffixes = "".join(file.relative_path.suffixes)
                    return suffixes in args.file_types or (suffixes.startswith(".") and suffixes[1:] in args.file_types)

            detections = vend.detect(
                test_repo, source_repo, file_filter=file_filter, max_history_depth=args.max_history_depth
            )

            # Output based on format
            if args.format == "csv":
                output_csv(detections, args.min_similarity, output_file)
            elif args.format == "json":
                output_json(detections, args.min_similarity, output_file)
            else:  # rich format
                output_rich(detections, console, args.min_similarity, output_file)
    except VendetectError as e:
        logger.error(str(e))  # noqa: TRY400
    except KeyboardInterrupt:
        sys.exit(127)
    finally:
        # Close the output file if it was opened
        if output_file and output_file != sys.stdout:
            output_file.close()
