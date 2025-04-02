from pathlib import Path
import shutil
import subprocess
from typing import Iterator


GIT_PATH: Path | None = shutil.which("git")


class Repository:
    def __init__(self, root_path: Path):
        self.root_path: Path = root_path
        if self.is_git:
            self.rev: str = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        else:
            self.rev = ""

    def __hash__(self) -> int:
        return hash(self.rev)

    @property
    def is_git(self) -> bool:
        return self.root_path.is_dir() and (self.root_path / ".git").is_dir()

    def git_files(self) -> Iterator[Path]:
        if GIT_PATH is None:
            raise RuntimeError("`git` binary could not be found")
        for line in subprocess.check_output([GIT_PATH, "ls-files"], cwd=self.root_path).splitlines():
            line = line.strip()
            if line:
                path = Path(line.decode("utf-8"))
                if not path.is_absolute():
                    path = self.root_path / path
                yield path

    def files(self) -> Iterator[Path]:
        if GIT_PATH is None or not self.is_git:
            stack: list[Path] = [self.root_path]
        else:
            stack = list(reversed(list(self.git_files())))
        history = set()
        while stack:
            path = stack.pop()
            if path.is_symlink():
                path = path.readlink()
            if path in history:
                continue
            history.add(path)
            if path.is_dir():
                stack.extend(reversed(list(path.iterdir())))
            elif path.is_file():
                yield path

    def previous_version

    def __iter__(self) -> Iterator[Path]:
        yield from self.files()
