import argparse
import logging
from pathlib import Path
import sys
from typing import Iterable

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TaskID
from rich import traceback

from .detector import Status, VenDetector
from .repo import Repository

logger = logging.getLogger(__name__)

class RichStatus(Status):
    def __init__(self, console: Console):
        self.console: Console = console
        self.progress: Progress | None = None
        self.compare_tasks: list[tuple[TaskID, TaskID, Repository]] = []

    def __enter__(self):
        self.progress = Progress(console=self.console, transient=True)
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def on_compare(self, test_repo: Repository, source_repo: Repository, test_paths: Iterable[Path] = (),
                source_paths: Iterable[Path] = ()):
        if not test_paths or not source_paths:
            self.compare_tasks.append((
                self.progress.add_task(f":magnifying_glass_tilted_right: {test_repo!s}", start=False),
                TaskID(-1),
                source_repo,
            ))

    def compare_completed(self, test_repo: Repository, source_repo: Repository, test_paths: Iterable[Path] = (),
                          source_paths: Iterable[Path] = ()):
        if not test_paths or not source_paths:
            self.compare_tasks.pop()

    def update_num_test_paths(self, num: int):
        self.progress.update(self.compare_tasks[-1][0], total=num)

    def update_num_source_paths(self, num: int):
        if self.compare_tasks[-1][1] < 0:
            self.compare_tasks[-1] = (
                self.compare_tasks[-1][0],
                self.progress.add_task(f":paw_print: {self.compare_tasks[-1][2]!s}", start=True),
                self.compare_tasks[-1][2]
            )
        self.progress.update(self.compare_tasks[-1][1], total=num)

    def update_test_progress(self, path: Path):
        self.progress.update(self.compare_tasks[-1][0], advance=1, start=True, description=f"::magnifying_glass_tilted_right:: {path.name!s}")

    def update_source_progress(self, path: Path):
        self.progress.update(self.compare_tasks[-1][1], advance=1, description=f":paw_print: {path.name!s}")


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
            print(f"{d.test_repo!s}/{d.test.relative_path!s} <-- {d.source_repo!s}/{d.source.relative_path!s}")
