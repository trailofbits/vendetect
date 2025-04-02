from functools import wraps
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Self


GIT_PATH: Path | None = shutil.which("git")


def with_self(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self:
            return func(self, *args, **kwargs)

    return wrapper


class Repository:
    def __init__(self, root_path: Path):
        self.root_path: Path = root_path
        with self:
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, Repository):
            return False
        if self.rev and other.rev:
            return self.rev == other.rev
        elif self.rev or other.rev:
            return False
        else:
            return self.root_path == other.root_path

    def __enter__(self) -> "Repository":
        # print(f"Entering {self!r}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f"Exiting {self!r}")
        pass

    @with_self
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
    @with_self
    def is_git(self) -> bool:
        return self.root_path.is_dir() and (self.root_path / ".git").is_dir()

    @with_self
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

    @with_self
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.root_path!r})"

    def __str__(self):
        if self.rev:
            return f"{self.root_path!s}@{self.rev}"
        else:
            return f"{self.root_path!s}"

    @classmethod
    def load(cls, repo_uri: str) -> "Repository":
        # first see if it is a local repo
        repo_uri_path = Path(repo_uri).absolute()
        if repo_uri_path.exists() and repo_uri_path.is_dir():
            return Repository(repo_uri_path)
        else:
            return RemoteGitRepository(repo_uri)


class _ClonedRepository(Repository):
    def __init__(self, clone_uri: str):
        self._clone_uri: str = clone_uri
        self._entries: int = 0
        self._tempdir: TemporaryDirectory | None = None
        super().__init__(Path(""))

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
        self.repo: Repository = repo
        self.rev = commit
        super().__init__(str(repo.root_path))

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.repo!r}, {self.rev!r})"

    @property
    def root(self) -> Repository:
        r = self
        while isinstance(r, RepositoryCommit):
            r = r.repo
        return r

    def __str__(self):
        return f"{self.root.root_path!s}@{self.rev}"


class RemoteGitRepository(_ClonedRepository):
    def __init__(self, url: str):
        self.url: str = url
        super().__init__(url)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.url!r})"

    def __str__(self):
        if self.rev:
            return f"{self.url!s}@{self.rev}"
        else:
            return f"{self.url!s}"
