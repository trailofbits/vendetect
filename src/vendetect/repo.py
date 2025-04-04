import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from functools import wraps
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional, ParamSpec, Self, TypeVar, cast
from urllib.parse import urlparse

GIT_PATH: str | None = shutil.which("git")

log = getLogger(__name__)

T = TypeVar('T')
P = ParamSpec('P')

def with_self(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with args[0]:  # type: ignore
            return func(*args, **kwargs)

    return wrapper


class Repository:
    def __init__(self, root_path: Path, rev: str | None = None):
        self.root_path: Path = root_path
        self.rev: str = ""
        if rev is None:
            with self:
                if self.is_git:
                    self.rev = (
                        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root_path)
                        .strip()
                        .decode("utf-8")
                    )
        else:
            self.rev = rev

    def __hash__(self) -> int:
        if self.rev:
            return hash(self.rev)
        return hash(self.root_path)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Repository):
            return False
        if self.rev and other.rev:
            return self.rev == other.rev
        if self.rev or other.rev:
            return False
        return self.root_path == other.root_path

    def __enter__(self) -> Self:
        # print(f"Entering {self!r}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        # print(f"Exiting {self!r}")
        pass

    @with_self
    def previous_version(self, path: Path) -> Optional["RepositoryCommit"]:
        if not self.is_git or GIT_PATH is None:
            return None
        if path.is_absolute():
            path = path.relative_to(self.root_path)
        prev_version = (
            subprocess.check_output(
                [
                    GIT_PATH,
                    "log",
                    "-n",
                    "1",
                    "--skip",
                    "1",
                    '--pretty=format:"%H"',
                    "--follow",
                    "--",
                    str(path),
                ],
                cwd=self.root_path,
            )
            .decode("utf-8")
            .strip()
        )
        if prev_version.startswith('"') and prev_version.endswith('"'):
            prev_version = prev_version[1:-1]
        if prev_version:
            return RepositoryCommit(self, prev_version)
        return None

    @property
    @with_self
    def is_git(self) -> bool:
        return self.root_path.is_dir() and (self.root_path / ".git").is_dir()

    @with_self
    def git_files(self) -> Iterator["File"]:
        if GIT_PATH is None:
            raise RuntimeError("`git` binary could not be found")
        for line in subprocess.check_output(
            [GIT_PATH, "ls-files"], cwd=self.root_path
        ).splitlines():
            line = line.strip()
            if line:
                path = Path(line.decode("utf-8"))
                yield File(path, self)

    @with_self
    def files(self) -> Iterator["File"]:
        if GIT_PATH is None or not self.is_git:
            stack: list[File] = [File(self.root_path, self)]
        else:
            stack = list(reversed(list(self.git_files())))
        history = set()
        while stack:
            file = stack.pop().resolve()
            if file in history:
                continue
            history.add(file)
            if file.path.is_dir():
                stack.extend((File(p, self) for p in reversed(list(file.path.iterdir()))))
            elif file.path.is_file():
                yield file

    def __iter__(self) -> Iterator["File"]:
        yield from self.files()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root_path!r})"

    def __str__(self) -> str:
        if self.rev:
            return f"{self.root_path!s}@{self.rev}"
        return f"{self.root_path!s}"

    @classmethod
    def load(cls, repo_uri: str) -> "Repository":
        # first see if it is a local repo
        repo_uri_path = Path(repo_uri).absolute()
        if repo_uri_path.exists() and repo_uri_path.is_dir():
            return Repository(repo_uri_path)
        return RemoteGitRepository(repo_uri)


class _ClonedRepository(Repository):
    def __init__(self, clone_uri: str, rev: str | None = None):
        self._clone_uri: str = clone_uri
        self._entries: int = 0
        self._tempdir: TemporaryDirectory | None = None
        super().__init__(Path(), rev=rev)

    def __enter__(self) -> Self:
        self._entries += 1
        if self._entries == 1:
            self._tempdir = TemporaryDirectory()
            self.root_path = Path(self._tempdir.__enter__())
            subprocess.check_call(
                [GIT_PATH, "clone", str(self._clone_uri), "."],  # type: ignore
                cwd=self.root_path,
                stderr=subprocess.DEVNULL,
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self._entries -= 1
        if self._entries == 0:
            self._tempdir.__exit__(exc_type, exc_val, exc_tb)  # type: ignore
            self._tempdir = None
        super().__exit__(exc_type, exc_val, exc_tb)


@dataclass(frozen=True, unsafe_hash=True, init=False)
class File:
    relative_path: Path
    repo: Repository

    def __init__(self, path: Path, repo: Repository):
        if path.is_absolute():
            path = path.relative_to(repo.root_path)
        object.__setattr__(self, "relative_path", path)
        object.__setattr__(self, "repo", repo)

    @property
    def path(self) -> Path:
        with self.repo:
            return self.repo.root_path / self.relative_path

    def resolve(self) -> Self:
        with self.repo:
            if self.path.is_symlink():
                return self.__class__(self.path.readlink(), self.repo)
            return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.relative_path!r}, {self.repo!r})"

    def __str__(self) -> str:
        return f"{self.repo!s}/{self.relative_path!s}"


class RepositoryCommit(_ClonedRepository):
    def __init__(self, repo: Repository, commit: str):
        if isinstance(repo, RepositoryCommit):
            repo = repo.root
        self.repo: Repository = repo
        self.rev = commit
        super().__init__(str(repo.root_path), rev=commit)

    def _ancestors(self) -> list[Repository]:
        stack: list[Repository] = [self]
        while isinstance(stack[-1], RepositoryCommit):
            stack.append(stack[-1].repo)
        return stack

    def _enter(self) -> Self:
        ret = super().__enter__()
        if self._entries == 1:
            subprocess.check_call(
                [GIT_PATH, "checkout", self.rev], cwd=self.root_path, stderr=subprocess.DEVNULL  # type: ignore
            )
        return ret

    def __enter__(self) -> Self:
        for a in reversed(self._ancestors()):
            if isinstance(a, RepositoryCommit):
                a._enter()
            else:
                a.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        for a in self._ancestors():
            if a is self:
                super().__exit__(exc_type, exc_val, exc_tb)
            else:
                a.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.repo!r}, {self.rev!r})"

    @property
    def root(self) -> Repository:
        r: Repository = self
        while isinstance(r, RepositoryCommit):
            r = r.repo
        return r

    def __str__(self) -> str:
        root = self.root
        if isinstance(root, RemoteGitRepository):
            return root.format_url()
        return f"{self.root.root_path!s}@{self.rev[:8]}"


class RemoteGitRepository(_ClonedRepository):
    def __init__(self, url: str):
        self.url: str = url
        super().__init__(url, rev="")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url!r})"

    @property
    def is_git(self) -> bool:
        return True

    def __enter__(self) -> Self:
        if self._entries == 0:
            log.info(f"⎘ cloning {self!s}…")
        return super().__enter__()

    def format_url(self, for_file: File | None = None, for_rev: str | None = None) -> str:
        if for_rev is None:
            if self.rev:
                for_rev = self.rev
            elif for_file:
                # get the latest commit from the remote repo
                for_rev = (
                    subprocess.check_output(["git", "ls-remote", self.url, "HEAD"])
                    .split()[0]
                    .decode("utf-8")
                )
        result = urlparse(self.url)
        if result.netloc == "github.com" and for_rev:
            path = result.path
            if path.endswith("/"):
                path = path[:-1]
            if path.endswith(".git"):
                path = path[:-4]
            if for_file is not None:
                return result._replace(
                    path=f"{path}/blob/{for_rev}/{for_file.relative_path!s}"
                ).geturl()
            return result._replace(path=f"{path}/commit/{for_rev}").geturl()
        if for_file is not None:
            url = self.url
            if url.endswith("/"):
                url = url[:-1]
            if for_rev:
                url = f"{url}@{for_rev}"
            return f"{url}/{for_file.relative_path!s}"
        if for_rev:
            return f"{self.url}@{for_rev}"
        return self.url

    __str__ = format_url
