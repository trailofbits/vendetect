import argparse
import logging
import sys
from typing import Iterable

from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.syntax import Syntax
from rich.table import Table
from rich import traceback

from .detector import get_lexer_for_filename, Status, VenDetector
from .repo import File, Repository

logger = logging.getLogger(__name__)

class RichStatus(Status):
    def __init__(self, console: Console):
        self.console: Console = console
        self.progress: Progress | None = None
        self.compare_tasks: list[TaskID] = []

    def __enter__(self):
        self.progress = Progress(console=self.console, transient=True)
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def on_compare(self, test_files: Iterable[File], source_files: Iterable[File]):
        self.compare_tasks.append(self.progress.add_task(f":magnifying_glass_tilted_right: comparing…"))

    def compare_completed(self, test_files: Iterable[File], source_files: Iterable[File]):
        self.progress.remove_task(self.compare_tasks[-1])
        self.compare_tasks.pop()

    def update_num_comparisons(self, num: int):
        self.progress.update(self.compare_tasks[-1], total=num)

    def update_compare_progress(self, file: File | None = None):
        self.progress.update(self.compare_tasks[-1], advance=1)
        if file is not None:
            self.progress.update(self.compare_tasks[-1],
                                 description=f":magnifying_glass_tilted_right: {file.relative_path.name!s}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vendetect")

    parser.add_argument("TEST_REPO", type=str, help="path to the test repository")
    parser.add_argument("SOURCE_REPO", type=str, help="path to the source repository")

    log_section = parser.add_argument_group(title="logging")
    log_group = log_section.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=list(
            logging.getLevelName(x)
            for x in range(1, 101)
            if not logging.getLevelName(x).startswith("Level")
        ),
        help="sets the log level for deptective (default=INFO)",
    )
    log_group.add_argument(
        "--debug", action="store_true", help="equivalent to `--log-level=DEBUG`"
    )
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
            exit(1)
        numeric_log_level = log_level

    console = Console(log_path=False, file=sys.stderr)

    logging.basicConfig(
        level=numeric_log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console)],
    )

    # logging.getLogger("root").setLevel(logging.ERROR)

    traceback.install(show_locals=True)

    with Repository.load(args.TEST_REPO) as test_repo, Repository.load(args.SOURCE_REPO) as source_repo, \
            RichStatus(Console()) as status:
        vend = VenDetector(status=status)

        for d in vend.detect(test_repo, source_repo):
            # Create a table for the detection results
            table = Table(title=f"Vendoring Detection", expand=True)
            table.add_column("Test File", style="cyan")
            table.add_column("Source File", style="green")
            table.add_column("Similarity", justify="right", style="yellow")
            
            # Calculate overall similarity (average of both similarities)
            avg_similarity = (d.comparison.similarity1 + d.comparison.similarity2) / 2

            if avg_similarity < 0.5:
                break

            similarity_str = f"{avg_similarity:.1%}"
            
            # Add the main row with test and source files
            table.add_row(
                f"{d.test.relative_path!s}",
                f"{d.source.relative_path!s}",
                similarity_str
            )
            
            # Read file content for both test and source files
            def read_file_content(file):
                with file.repo:
                    return file.path.read_text()
            
            try:
                test_content = read_file_content(d.test)
                source_content = read_file_content(d.source)
                
                # Create a side-by-side view of the detected slices
                test_slices = d.comparison.slices1
                source_slices = d.comparison.slices2
                
                if len(test_slices[0]) > 0:
                    # Get the first detected slice (most significant one)
                    test_start: int = test_slices[0][0]
                    test_end: int = test_slices[1][0]
                    source_start: int = source_slices[0][0]
                    source_end: int = source_slices[1][0]

                    # Extract the content for the detected slices
                    test_lines = test_content.splitlines()
                    source_lines = source_content.splitlines()
                    
                    # Convert character positions to line numbers (approximate)
                    test_slice_content = "\n".join(test_lines[max(0, test_start-10):test_end+10])
                    source_slice_content = "\n".join(source_lines[max(0, source_start-10):source_end+10])
                    
                    # Create syntax-highlighted code panels
                    test_syntax = Syntax(
                        test_slice_content,
                        lexer=get_lexer_for_filename(d.test.relative_path.name),
                        line_numbers=True,
                        start_line=max(1, test_start-10),
                        highlight_lines=set(range(max(1, test_start), test_end+1))
                    )
                    
                    source_syntax = Syntax(
                        source_slice_content, 
                        lexer=get_lexer_for_filename(d.source.relative_path.name),
                        line_numbers=True,
                        start_line=max(1, source_start-10),
                        highlight_lines=set(range(max(1, source_start), source_end+1))
                    )
                    
                    # Create side-by-side panels
                    test_panel = Panel(test_syntax, title="Test Code", border_style="cyan")
                    source_panel = Panel(source_syntax, title="Source Code", border_style="green")
                    
                    # Print the result
                    console.print(table)
                    console.print()

                    context_panel = Panel.fit(
                        Columns([test_panel, source_panel]),
                        title="Vendored Code",
                        border_style="red",
                        title_align="left",
                        padding=(1, 1),
                    )

                    console.print(context_panel)
                    console.print()
                else:
                    # Just print the table if there are no specific slices
                    console.print(table)
                    console.print()
            except Exception as e:
                # Fallback to basic output if there's an error reading or processing files
                console.print(table)
                console.print(f"[red]Error displaying code: {e}[/red]")
                console.print()
