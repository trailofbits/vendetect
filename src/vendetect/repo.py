from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterable, Iterator, Optional, Self


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
        if self.rev:
            return hash(self.rev)
        else:
            return hash(self.root_path)

    def __enter__(self) -> "Repository":
        # print(f"Entering {self!r}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f"Exiting {self!r}")
        pass

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


class _ClonedRepository(Repository):
    def __init__(self, clone_uri: str):
        super().__init__(Path(""))
        self._clone_uri: str = clone_uri
        self._entries: int = 0
        self._tempdir: TemporaryDirectory | None = None

    def __enter__(self) -> Self:
        if self._entries == 0:
            self._tempdir = TemporaryDirectory()
            self.root_path = Path(self._tempdir.__enter__())
            subprocess.check_call([GIT_PATH, "clone", str(self._clone_uri), "."], cwd=self.root_path,
                                  stderr=subprocess.DEVNULL)
        self._entries += 1
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._entries -= 1
        if self._entries == 0:
            self._tempdir.__exit__(exc_type, exc_val, exc_tb)
            self._tempdir = None
        super().__exit__(exc_type, exc_val, exc_tb)

class RepositoryCommit(_ClonedRepository):
    def __init__(self, repo: Repository, commit: str):
        super().__init__(str(repo.root_path))
        self.repo: Repository = repo
        self.rev = commit

    def _ancestors(self) -> list[Repository]:
        stack = [self]
        while isinstance(stack[-1], RepositoryCommit):
            stack.append(stack[-1].repo)
        return stack

    def _enter(self) -> Self:
        ret = super().__enter__()
        if self._entries == 1:
            subprocess.check_call([GIT_PATH, "checkout", self.rev], cwd=self.root_path, stderr=subprocess.DEVNULL)
        return ret

    def __enter__(self) -> Self:
        for a in reversed(self._ancestors()):
            if isinstance(a, RepositoryCommit):
                a._enter()
            else:
                a.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for a in self._ancestors():
            if a is self:
                super().__exit__(exc_type, exc_val, exc_tb)
            else:
                a.__exit__(exc_type, exc_val, exc_tb)


class RemoteGitRepository(_ClonedRepository):
    def __init__(self, url: str):
        super().__init__(url)
        self.url: str = url
