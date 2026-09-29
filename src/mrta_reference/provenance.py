"""Fail-closed source provenance for formal scientific runs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import subprocess
from urllib.parse import urlparse


REPOSITORY_ID = "luckyfishanddog/DRL"
SOURCE_PROVENANCE_POLICY_V1 = "SOURCE_PROVENANCE_POLICY_V1"


class SourceProvenanceError(RuntimeError):
    """Raised when a run cannot establish the requested source identity."""


@dataclass(frozen=True)
class SourceProvenance:
    repository_id: str
    source_commit: str
    source_tree_hash: str
    worktree_dirty: bool
    commit_verified: bool

    @property
    def development_only(self) -> bool:
        return not self.commit_verified or self.worktree_dirty

    def require_formal_result(self) -> None:
        if not self.commit_verified:
            raise SourceProvenanceError(
                "formal result requires a commit verified against luckyfishanddog/DRL"
            )
        if self.worktree_dirty:
            raise SourceProvenanceError("formal result requires a clean DRL worktree")


def _source_files(repository_root: Path) -> tuple[Path, ...]:
    files = []
    pyproject = repository_root / "pyproject.toml"
    if pyproject.is_file():
        files.append(pyproject)
    source_root = repository_root / "src"
    if source_root.is_dir():
        files.extend(path for path in source_root.rglob("*.py") if path.is_file())
    return tuple(sorted(files, key=lambda path: path.relative_to(repository_root).as_posix()))


def compute_source_tree_hash(repository_root: str | Path) -> str:
    """Hash sorted relative paths and exact bytes for scientific source/config."""
    root = Path(repository_root).resolve()
    files = _source_files(root)
    if not files:
        raise SourceProvenanceError(f"no scientific source files found under {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(struct.pack(">Q", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", len(content)))
        digest.update(content)
    return digest.hexdigest()


def canonicalize_repository_url(remote_url: str) -> str | None:
    value = remote_url.strip().replace("\\", "/")
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
    else:
        parsed = urlparse(value if "://" in value else f"https://{value}")
        host = (parsed.hostname or "").lower()
        path = parsed.path.lstrip("/")
    path = path.rstrip("/")
    if path.lower().endswith(".git"):
        path = path[:-4]
    if host.lower() != "github.com" or not path:
        return None
    return f"github.com/{path}".lower()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=root, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise SourceProvenanceError(detail)
    return completed.stdout.strip()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _remote_urls(root: Path) -> tuple[str, ...]:
    names = tuple(name for name in _git(root, "remote").splitlines() if name)
    urls = []
    for name in names:
        urls.extend(
            line for line in _git(root, "remote", "get-url", "--all", name).splitlines() if line
        )
    return tuple(urls)


def resolve_source_provenance(
    repository_root: str | Path,
    *,
    source_commit: str | None = None,
    allow_unverified_source: bool = False,
) -> SourceProvenance:
    """Resolve DRL provenance, refusing to borrow an enclosing repository HEAD."""
    root = Path(repository_root).resolve()
    tree_hash = compute_source_tree_hash(root)
    git_root = None
    git_error = None
    try:
        git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    except SourceProvenanceError as error:
        git_error = str(error)

    root_matches = git_root is not None and _same_path(root, git_root)
    remote_matches = False
    if root_matches:
        target = f"github.com/{REPOSITORY_ID}".lower()
        remote_matches = any(canonicalize_repository_url(url) == target for url in _remote_urls(root))

    if root_matches and remote_matches:
        resolved_commit = _git(root, "rev-parse", "HEAD")
        if source_commit is not None and source_commit != resolved_commit:
            raise SourceProvenanceError(
                "explicit source_commit does not match the verified DRL HEAD"
            )
        dirty = bool(_git(root, "status", "--porcelain"))
        return SourceProvenance(REPOSITORY_ID, resolved_commit, tree_hash, dirty, True)

    if not allow_unverified_source:
        reason = (
            "git root is not the supplied DRL root"
            if git_root is not None and not root_matches
            else "no remote matches github.com/luckyfishanddog/DRL"
            if root_matches
            else git_error or "DRL git root is unavailable"
        )
        raise SourceProvenanceError(f"unverified DRL source provenance: {reason}")
    if not source_commit:
        raise SourceProvenanceError(
            "allow_unverified_source requires an explicit source_commit label"
        )
    dirty = True
    if git_root is not None:
        dirty = bool(_git(root, "status", "--porcelain", "--", "."))
    return SourceProvenance(REPOSITORY_ID, source_commit, tree_hash, dirty, False)
