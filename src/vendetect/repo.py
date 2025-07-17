import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import wraps
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, ParamSpec, Self, TypeVar
from urllib.parse import urlparse

from .errors import VendetectRuntimeError

GIT_PATH: str | None = shutil.which("git")

log = getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


class RepositoryError(VendetectRuntimeError):
    pass


def with_self(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with args[0]:  # type: ignore
            return func(*args, **kwargs)

    return wrapper


class Repository:
    def __init__(self, root_path: Path, rev: str | None = None, subdir: Path | None = None):
        self.root_path: Path = root_path
        if subdir is not None and subdir.is_absolute():
            subdir = subdir.relative_to(root_path)
        self.subdir: Path | None = subdir
        self.rev: str = ""
        if rev is None:
            with self:
                if self.is_git:
                    git_path: str = GIT_PATH  # type: ignore
                    self.rev = (
                        subprocess.check_output([git_path, "rev-parse", "HEAD"], cwd=self.path)  #  noqa: S603
                        .strip()
                        .decode("utf-8")
                    )
        else:
            self.rev = rev

    @property
    @with_self
    def path(self) -> Path:
        if self.subdir is None:
            return self.root_path
        return self.root_path / self.subdir

    def __hash__(self) -> int:
        if self.rev:
            return hash(self.rev)
        return hash(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Repository):
            return False
        if self.rev and other.rev:
            return self.rev == other.rev
        if self.rev or other.rev:
            return False
        return self.path == other.path

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        pass

    @with_self
    def previous_version(self, path: Path) -> Optional["RepositoryCommit"]:
        if not self.is_git:
            log.debug(
                "Cannot get previous version of %s because %s does not appear to be a git repository",
                str(path),
                str(self),
            )
            return None
        if GIT_PATH is None:
            log.warning("Cannot get previous version of %s because `git` is not installed", str(File(path, self)))
            return None
        if self.is_shallow_clone:
            msg = (
                f"{self!s} appears to be a shallow clone; please fetch the entire git history, e.g., "
                f"with `git fetch --unshallow` or by cloning the entire repository"
            )
            raise RepositoryError(msg)
        if path.is_absolute():
            path = path.relative_to(self.path)
        prev_version = (
            subprocess.check_output(  # noqa: S603
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
                cwd=self.path,
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
    def is_shallow_clone(self) -> bool:
        """Test whether the root repository is a shallow cone."""
        try:
            return (
                subprocess.check_output(  # noqa: S603
                    [GIT_PATH, "rev-parse", "--is-shallow-repository"],  # type: ignore
                    cwd=self.path,
                    stderr=subprocess.DEVNULL,
                ).strip()
                != b"false"
            )
        except subprocess.CalledProcessError:
            return False

    @property
    @with_self
    def git_root(self) -> Path | None:
        if GIT_PATH is None:
            return None
        try:
            return Path(
                subprocess.check_output(  # noqa: S603
                    [GIT_PATH, "-C", str(self.path), "rev-parse", "--show-toplevel"],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .decode("utf-8")
            )
        except subprocess.CalledProcessError:
            return None

    @with_self
    def is_inside_git_work_tree(self) -> bool:
        if GIT_PATH is None:
            return False
        try:
            return (
                subprocess.check_output(  # noqa: S603
                    [GIT_PATH, "-C", str(self.path), "rev-parse", "--is-inside-work-tree"],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .lower()
                == b"true"
            )
        except subprocess.CalledProcessError:
            return False

    @property
    @with_self
    def is_git(self) -> bool:
        return self.root_path.is_dir() and ((self.root_path / ".git").is_dir() or self.is_inside_git_work_tree())

    @with_self
    def git_files(self) -> Iterator["File"]:
        if GIT_PATH is None:
            msg = "`git` binary could not be found"
            raise RepositoryError(msg)
        for line in subprocess.check_output(  # noqa: S603
            [GIT_PATH, "ls-files", "--cached", "--others", "--exclude-standard"], cwd=self.path
        ).splitlines():
            line = line.strip()  # noqa: PLW2901
            if line:
                path = Path(line.decode("utf-8"))
                if self.subdir is not None:
                    path = self.subdir / path
                yield File(path, self)

    @with_self
    def files(self) -> Iterator["File"]:
        if GIT_PATH is None or not self.is_git:
            stack: list[File] = [File(self.path, self)]
        else:
            stack = list(reversed(list(self.git_files())))
        history = set()
        while stack:
            file = stack.pop().resolve()
            if file in history:
                continue
            history.add(file)
            if file.path.is_dir():
                stack.extend(File(p, self) for p in reversed(list(file.path.iterdir())))
            elif file.path.is_file():
                yield file

    def __iter__(self) -> Iterator["File"]:
        yield from self.files()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.root_path!r}, rev={self.rev!r}, subdir={self.subdir!r})"

    def __str__(self) -> str:
        if self.rev:
            return f"{self.path!s}@{self.rev}"
        return f"{self.path!s}"

    @classmethod
    def load(cls, repo_uri: str, subdir: Path | str | None = None) -> "Repository":
        # first see if it is a local repo
        repo_uri_path = Path(repo_uri).absolute()
        if subdir is not None and not isinstance(subdir, Path):
            subdir = Path(subdir)
        if repo_uri_path.exists() and repo_uri_path.is_dir():
            return Repository(repo_uri_path, subdir=subdir)
        return RemoteGitRepository(repo_uri, subdir=subdir)


class _ClonedRepository(Repository):
    def __init__(self, clone_uri: str, rev: str | None = None, subdir: Path | None = None):
        self._clone_uri: str = clone_uri
        self._entries: int = 0
        self._tempdir: TemporaryDirectory | None = None
        if subdir is not None and subdir.is_absolute():
            msg = f"Invalid subdirectory {subdir!s}: the path must be relative, not absolute"
            raise ValueError(msg)
        if GIT_PATH is None:
            msg = (
                f"Error cloning {self._clone_uri}: `git` binary could not be found;please make sure it is in your PATH"
            )
            raise RepositoryError(msg)
        super().__init__(Path(), rev=rev, subdir=subdir)

    def __enter__(self) -> Self:
        self._entries += 1
        if self._entries == 1:
            self._tempdir = TemporaryDirectory()
            self.root_path = Path(self._tempdir.__enter__())
            try:
                subprocess.check_call(  # noqa: S603
                    [GIT_PATH, "clone", str(self._clone_uri), "."],  # type: ignore
                    cwd=self.root_path,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as e:
                msg = f"Failed to clone `{self._clone_uri}`: {e!s}"
                raise RepositoryError(msg) from None
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

    def commit_exists(self, rev: str | None = None) -> bool:
        if rev is None:
            rev = self.rev
        try:
            return (
                subprocess.check_output(  # noqa: S603
                    [GIT_PATH, "cat-file", "-t", rev],  # type: ignore
                    cwd=self.root_path,
                    stderr=subprocess.DEVNULL,
                ).strip()
                == b"commit"
            )
        except subprocess.CalledProcessError:
            return False

    def _enter(self) -> Self:
        ret = super().__enter__()
        if self._entries == 1:
            if not self.commit_exists():
                # is the source repo a shallow cone?
                if self.is_shallow_clone:
                    msg = (
                        f"{self.root!s} appears to be a shallow clone; please fetch the entire git history, e.g., "
                        f"with `git fetch --unshallow` or by cloning the entire repository"
                    )
                else:
                    msg = f"{self.root!s} does not have commit {self.rev} fetched"
                raise RepositoryError(msg)
            subprocess.check_call(  # noqa: S603
                [GIT_PATH, "checkout", self.rev],  # type: ignore
                cwd=self.root_path,
                stderr=subprocess.DEVNULL,
            )
        return ret

    def __enter__(self) -> Self:
        for a in reversed(self._ancestors()):
            if isinstance(a, RepositoryCommit):
                a._enter()  # noqa: SLF001
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
    def __init__(self, url: str, subdir: Path | None = None):
        self.url: str = url
        super().__init__(url, rev="", subdir=subdir)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url!r}, subdir={self.subdir!r})"

    @property
    def is_git(self) -> bool:
        return True

    def __enter__(self) -> Self:
        if self._entries == 0:
            log.info("⎘ cloning %s…", str(self))
        return super().__enter__()

    def format_url(self, for_file: File | None = None, for_rev: str | None = None) -> str:
        if for_rev is None:
            if self.rev:
                for_rev = self.rev
            elif for_file:
                # get the latest commit from the remote repo
                git_path: str = GIT_PATH  # type: ignore
                raw_head = subprocess.check_output([git_path, "ls-remote", self.url, "HEAD"])  # noqa: S603
                for_rev = raw_head.split()[0].decode("utf-8")
        result = urlparse(self.url)
        if result.netloc == "github.com" and for_rev:
            path = result.path
            path = path.removesuffix("/")
            path = path.removesuffix(".git")
            if for_file is not None:
                return result._replace(path=f"{path}/blob/{for_rev}/{for_file.relative_path!s}").geturl()
            return result._replace(path=f"{path}/commit/{for_rev}").geturl()
        if for_file is not None:
            url = self.url
            url = url.removesuffix("/")
            if for_rev:
                url = f"{url}@{for_rev}"
            return f"{url}/{for_file.relative_path!s}"
        if for_rev:
            return f"{self.url}@{for_rev}"
        return self.url

    __str__ = format_url
