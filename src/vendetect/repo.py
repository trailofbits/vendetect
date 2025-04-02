from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterator, Optional


GIT_PATH: Path | None = shutil.which("git")


class Repository:
    def __init__(self, root_path: Path):
        self.root_path: Path = root_path
        if self.is_git:
            self.rev: str = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root_path
            ).strip().decode("utf-8")
        else:
            self.rev = ""

    def __hash__(self) -> int:
        return hash(self.rev)

    def previous_version(self, path: Path) -> Optional["RepositoryCommit"]:
        if not self.is_git or GIT_PATH is None:
            return None
        if path.is_absolute():
            path = path.relative_to(self.root_path)
        prev_version = subprocess.check_output([
            GIT_PATH, "log", "-n", "1", "--skip", "1", "--pretty=format:\"%H\"", "--follow", "--", str(path)
        ], cwd=self.root_path).decode("utf-8").strip()
        if prev_version.startswith('"') and prev_version.endswith('"'):
            prev_version = prev_version[1:-1]
        if prev_version:
            return RepositoryCommit(self, prev_version)
        else:
            return None

    @property
    def is_git(self) -> bool:
        return self.root_path.is_dir() and (self.root_path / ".git").is_dir()

    def git_files(self) -> Iterator[Path]:
        if GIT_PATH is None:
            raise RuntimeError("`git` binary could not be found")
        for line in subprocess.check_output([str(GIT_PATH), "ls-files"], cwd=self.root_path).splitlines():
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

    def __iter__(self) -> Iterator[Path]:
        yield from self.files()


class RepositoryCommit(Repository):
    def __init__(self, repo: Repository, commit: str):
        super().__init__(repo.root_path)
        self.repo: Repository = repo
        self.rev = commit
        self._entries: int = 0
        self._tempdir: TemporaryDirectory | None = None

    def __enter__(self) -> "RepositoryCommit":
        if self._entries == 0:
            self._tempdir = TemporaryDirectory()
            self.root_path = Path(self._tempdir.__enter__())
            subprocess.check_call([GIT_PATH, "clone", str(self.repo.root_path), "."], cwd=self.root_path,
                                  stderr=subprocess.DEVNULL)
            subprocess.check_call([GIT_PATH, "checkout", self.rev], cwd=self.root_path, stderr=subprocess.DEVNULL)
        self._entries += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._entries -= 1
        if self._entries == 0:
            self._tempdir.__exit__(exc_type, exc_val, exc_tb)
            self._tempdir = None
