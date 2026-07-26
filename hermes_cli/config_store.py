"""Fail-closed filesystem primitives for Hermes configuration writers.

The public config helpers use this module for inter-process coordination.  A
lock is keyed by the captured physical target, not by the spelling used by a
caller, so a symlink and its destination contend for the same lock.
"""
from __future__ import annotations

import base64
import contextvars
import errno
import hashlib
import json
import logging
import ntpath
import os
import re
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from utils import _fsync_directory, atomic_replace

try:  # pragma: no cover - exercised on POSIX in the supported runtime
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
try:  # pragma: no cover - imported only on native Windows
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]


_IS_WINDOWS = os.name == "nt"


_log = logging.getLogger(__name__)
_SAFE_SURFACE_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")


def _safe_surface_name(surface: str) -> str:
    """Return a bounded code-owned label, never caller data."""
    value = str(surface or "")
    return value if _SAFE_SURFACE_RE.fullmatch(value) else "invalid"


def _content_sha256(content: bytes | None) -> str | None:
    return hashlib.sha256(content).hexdigest() if content is not None else None


def _emit_write_provenance(
    *,
    surface: str,
    targets: Sequence[Path],
    before_contents: Sequence[bytes | None],
    result: str,
) -> None:
    """Emit hashes only; never log paths, values, or exception strings."""
    target_events: list[dict[str, str | None]] = []
    for index, target in enumerate(targets):
        before = before_contents[index] if index < len(before_contents) else None
        try:
            after = snapshot_file(target).content
        except BaseException:
            after = None
        target_events.append(
            {
                "before_sha256": _content_sha256(before),
                "after_sha256": _content_sha256(after),
            }
        )
    event = {
        "pid": os.getpid(),
        "surface": _safe_surface_name(surface),
        "targets": target_events,
        "result": result,
    }
    _log.info("config_write_provenance %s", json.dumps(event, sort_keys=True))


class ConfigStoreError(RuntimeError):
    """Base class for actionable configuration-store failures."""


class UnsafeConfigPathError(ConfigStoreError):
    """A path crossed a symlink/non-directory or changed while locked."""


class ConfigLockTimeoutError(ConfigStoreError):
    """A physical target could not be locked before the deadline."""


class ConfigTransactionError(ConfigStoreError):
    """A multi-file publication failed and was rolled back."""


class _AfterPublishError(ConfigTransactionError):
    """Internal carrier for a callback error after successful rollback."""

    def __init__(self, cause: BaseException):
        super().__init__("post-publish validation failed")
        self.cause = cause


@dataclass(frozen=True)
class TargetCapture:
    """The spelling and physical identity captured before a lock is taken."""

    target: Path
    physical: Path
    kind: str
    link_text: str | None
    identity: tuple[int, int] | None

    def verify_unchanged(self, *, verify_physical: bool = True) -> None:
        """Reject retargeting/replacement instead of writing a stale target."""
        try:
            current = os.lstat(self.target)
        except FileNotFoundError as exc:
            if self.kind == "absent":
                return
            raise UnsafeConfigPathError(
                f"Configuration target disappeared while locked: {self.target}"
            ) from exc
        except OSError as exc:
            raise UnsafeConfigPathError(
                f"Cannot validate configuration target {self.target}: {exc}"
            ) from exc

        if self.kind == "absent":
            raise UnsafeConfigPathError(
                f"Configuration target appeared while locked: {self.target}"
            )
        if self.kind == "symlink":
            if not stat.S_ISLNK(current.st_mode):
                raise UnsafeConfigPathError(
                    f"Configuration target changed from symlink: {self.target}"
                )
            try:
                link_text = os.readlink(self.target)
                physical = Path(os.path.realpath(self.target))
            except OSError as exc:
                raise UnsafeConfigPathError(
                    f"Cannot validate symlink target {self.target}: {exc}"
                ) from exc
            if link_text != self.link_text or physical != self.physical:
                raise UnsafeConfigPathError(
                    f"Configuration symlink was retargeted while locked: {self.target}"
                )
            if verify_physical:
                try:
                    physical_stat = os.lstat(self.physical)
                    physical_identity = (physical_stat.st_dev, physical_stat.st_ino)
                    physical_nlink = physical_stat.st_nlink
                except FileNotFoundError:
                    physical_identity = None
                    physical_nlink = None
                except OSError as exc:
                    raise UnsafeConfigPathError(
                        f"Cannot validate symlink destination {self.target}: {exc}"
                    ) from exc
                if physical_nlink is not None and physical_nlink > 1:
                    raise UnsafeConfigPathError(
                        f"Configuration symlink destination has hardlink aliases: {self.target}"
                    )
                if physical_identity != self.identity:
                    raise UnsafeConfigPathError(
                        f"Configuration symlink destination changed while locked: {self.target}"
                    )
            return

        if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
            raise UnsafeConfigPathError(
                f"Configuration target changed type while locked: {self.target}"
            )
        if current.st_nlink > 1:
            raise UnsafeConfigPathError(
                f"Configuration target has hardlink aliases: {self.target}"
            )
        # Once the path lock has recaptured the current inode, any further
        # replacement before publication is an uncoordinated race.
        if self.identity is not None and (current.st_dev, current.st_ino) != self.identity:
            raise UnsafeConfigPathError(
                f"Configuration target was replaced while locked: {self.target}"
            )


@dataclass(frozen=True)
class FileSnapshot:
    """Exact pre-publication state needed for a byte-for-byte rollback."""

    target: Path
    physical: Path
    exists: bool
    kind: str
    content: bytes | None
    mode: int | None
    owner: tuple[int, int] | None
    link_text: str | None


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def physical_target(path: str | os.PathLike[str] | Path) -> Path:
    """Return the physical destination, preserving a dangling final target."""
    return Path(os.path.realpath(os.fspath(path)))


def capture_target(path: str | os.PathLike[str] | Path) -> TargetCapture:
    """Capture a target before locking; aliases resolve to the same physical path."""
    target = _absolute_without_resolving(Path(path))
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        return TargetCapture(target, physical_target(target), "absent", None, None)
    except OSError as exc:
        raise UnsafeConfigPathError(f"Cannot inspect configuration target {target}: {exc}") from exc

    if stat.S_ISLNK(st.st_mode):
        try:
            link_text = os.readlink(target)
            physical = physical_target(target)
            physical_st = os.stat(physical)
            if physical_st.st_nlink > 1:
                raise UnsafeConfigPathError(
                    f"Configuration symlink destination has hardlink aliases: {target}"
                )
            identity = (physical_st.st_dev, physical_st.st_ino)
        except FileNotFoundError:
            link_text = os.readlink(target)
            physical = physical_target(target)
            identity = None
        except OSError as exc:
            raise UnsafeConfigPathError(f"Cannot read configuration symlink {target}: {exc}") from exc
        return TargetCapture(target, physical, "symlink", link_text, identity)
    if not stat.S_ISREG(st.st_mode):
        raise UnsafeConfigPathError(f"Configuration target is not a regular file: {target}")
    if st.st_nlink > 1:
        raise UnsafeConfigPathError(
            f"Configuration target has hardlink aliases: {target}"
        )
    return TargetCapture(target, target, "regular", None, (st.st_dev, st.st_ino))


def snapshot_file(path: str | os.PathLike[str] | Path) -> FileSnapshot:
    """Capture absence, bytes, mode, owner, and symlink spelling exactly."""
    target = _absolute_without_resolving(Path(path))
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        return FileSnapshot(target, physical_target(target), False, "absent", None, None, None, None)
    if stat.S_ISLNK(st.st_mode):
        link_text = os.readlink(target)
        physical = physical_target(target)
        try:
            physical_st = os.stat(physical)
            content = Path(physical).read_bytes()
            mode = stat.S_IMODE(physical_st.st_mode)
            owner = (physical_st.st_uid, physical_st.st_gid)
        except FileNotFoundError:
            content = None
            mode = None
            owner = None
        return FileSnapshot(target, physical, True, "symlink", content, mode, owner, link_text)
    if not stat.S_ISREG(st.st_mode):
        raise UnsafeConfigPathError(f"Configuration target is not a regular file: {target}")
    return FileSnapshot(
        target,
        target,
        True,
        "regular",
        target.read_bytes(),
        stat.S_IMODE(st.st_mode),
        (st.st_uid, st.st_gid),
        None,
    )


def _validate_existing_ancestors(
    path: Path, *, root: Path | None = None, include_leaf: bool = False
) -> None:
    """Validate every existing component before any lock-root mkdir/open."""
    absolute = _absolute_without_resolving(path)
    if not include_leaf:
        absolute = absolute.parent
    root_abs = _absolute_without_resolving(root) if root is not None else None
    components: list[Path] = []
    current = absolute
    while True:
        components.append(current)
        if current.parent == current:
            break
        current = current.parent
    for component in reversed(components):
        if root_abs is not None:
            try:
                component.relative_to(root_abs)
            except ValueError:
                continue
        try:
            st = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise UnsafeConfigPathError(f"Cannot inspect lock ancestor {component}: {exc}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise UnsafeConfigPathError(f"Lock path crosses symlink ancestor: {component}")
        if not stat.S_ISDIR(st.st_mode):
            raise UnsafeConfigPathError(f"Lock path crosses non-directory ancestor: {component}")


def _ensure_private_lock_root(home: Path) -> Path:
    home = _absolute_without_resolving(home)
    # All named profiles share the global ~/.hermes namespace.  Otherwise a
    # profile write-through to global auth.json could race a root-profile
    # writer while both held different lock files for the same physical path.
    if home.parent.name == "profiles":
        home = home.parent.parent
    _validate_existing_ancestors(home, root=None, include_leaf=True)
    try:
        home_st = os.lstat(home)
    except FileNotFoundError as exc:
        raise UnsafeConfigPathError(f"HERMES_HOME must exist before locking: {home}") from exc
    if stat.S_ISLNK(home_st.st_mode) or not stat.S_ISDIR(home_st.st_mode):
        raise UnsafeConfigPathError(f"HERMES_HOME is not a real directory: {home}")
    if not _IS_WINDOWS and hasattr(os, "geteuid") and home_st.st_uid != os.geteuid():
        raise UnsafeConfigPathError(f"Lock root is not owned by the current user: {home}")
    if not _IS_WINDOWS and stat.S_IMODE(home_st.st_mode) & 0o022:
        raise UnsafeConfigPathError(f"HERMES_HOME is group/world writable: {home}")

    lock_root = home / ".config-locks"
    _validate_existing_ancestors(lock_root, root=home, include_leaf=True)
    try:
        os.mkdir(lock_root, 0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        raise UnsafeConfigPathError(f"Cannot create private lock root {lock_root}: {exc}") from exc
    try:
        st = os.lstat(lock_root)
    except OSError as exc:
        raise UnsafeConfigPathError(f"Cannot validate private lock root {lock_root}: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UnsafeConfigPathError(f"Private lock root is a symlink: {lock_root}")
    if not stat.S_ISDIR(st.st_mode):
        raise UnsafeConfigPathError(f"Private lock root is not a directory: {lock_root}")
    if not _IS_WINDOWS and hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        raise UnsafeConfigPathError(f"Private lock root is not current-user owned: {lock_root}")
    if not _IS_WINDOWS and stat.S_IMODE(st.st_mode) & 0o077:
        raise UnsafeConfigPathError(f"Private lock root is not private: {lock_root}")
    return lock_root


def _canonical_physical_path(capture: TargetCapture) -> str:
    physical = str(capture.physical)
    return ntpath.normcase(physical) if _IS_WINDOWS else physical


def _lock_name(capture: TargetCapture) -> str:
    digest = hashlib.sha256(os.fsencode(_canonical_physical_path(capture))).hexdigest()
    return f"{digest}.lock"


@contextmanager
def _interprocess_lock_capture(
    capture: TargetCapture,
    *,
    lock_root: Path,
    timeout: float | None,
) -> Iterator[TargetCapture]:
    if fcntl is None and msvcrt is None:  # pragma: no cover
        raise ConfigStoreError(
            "Inter-process configuration locking is unavailable on this platform"
        )
    lock_path = lock_root / _lock_name(capture)
    _validate_existing_ancestors(lock_path, root=lock_root)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise UnsafeConfigPathError(
            f"Cannot open configuration lock {lock_path}: {exc}"
        ) from exc
    acquired = False
    try:
        if not _IS_WINDOWS and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        if fcntl is None:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    os.lseek(fd, 0, os.SEEK_SET)
                    getattr(msvcrt, "locking")(
                        fd,
                        getattr(msvcrt, "LK_NBLCK"),
                        1,
                    )
                acquired = True
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise ConfigLockTimeoutError(
                        f"Timed out waiting for configuration lock on {capture.physical}"
                    )
                time.sleep(0.01)
            except OSError as exc:
                if fcntl is not None:
                    raise
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise ConfigLockTimeoutError(
                        f"Timed out waiting for configuration lock on {capture.physical}"
                    )
                time.sleep(0.01)
        if capture.kind == "symlink":
            capture.verify_unchanged()
            locked_capture = capture
        elif capture.kind == "absent":
            locked_capture = capture_target(capture.target)
            if locked_capture.kind != "absent":
                raise UnsafeConfigPathError(
                    f"Configuration target appeared while locking: {capture.target}"
                )
            if locked_capture.physical != capture.physical:
                raise UnsafeConfigPathError(
                    f"Configuration physical target changed while locking: {capture.target}"
                )
            locked_capture.verify_unchanged()
        else:
            # A preceding writer may have atomically replaced this same physical
            # pathname while we waited. Recapture exactly once after acquiring
            # the path lock; subsequent replacement is rejected.
            locked_capture = capture_target(capture.target)
            if locked_capture.kind == "symlink":
                raise UnsafeConfigPathError(
                    f"Configuration target changed to symlink while locking: {capture.target}"
                )
            if locked_capture.physical != capture.physical:
                raise UnsafeConfigPathError(
                    f"Configuration physical target changed while locking: {capture.target}"
                )
            locked_capture.verify_unchanged()
        yield locked_capture
    finally:
        if acquired:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    os.lseek(fd, 0, os.SEEK_SET)
                    getattr(msvcrt, "locking")(
                        fd,
                        getattr(msvcrt, "LK_UNLCK"),
                        1,
                    )
            except OSError as exc:
                _log.warning(
                    "Failed to release configuration lock (%s)",
                    type(exc).__name__,
                )
        try:
            os.close(fd)
        except OSError as exc:
            _log.warning(
                "Failed to close configuration lock (%s)",
                type(exc).__name__,
            )


@contextmanager
def interprocess_lock(
    target: str | os.PathLike[str] | Path,
    *,
    home: str | os.PathLike[str] | Path,
    timeout: float | None = None,
) -> Iterator[TargetCapture]:
    """Lock one captured physical target using a private current-user root."""
    capture = capture_target(target)
    lock_root = _ensure_private_lock_root(Path(home))
    with _interprocess_lock_capture(
        capture,
        lock_root=lock_root,
        timeout=timeout,
    ) as locked:
        yield locked


@contextmanager
def interprocess_locks(
    targets: Sequence[str | os.PathLike[str] | Path],
    *,
    home: str | os.PathLike[str] | Path,
    timeout: float | None = None,
) -> Iterator[tuple[TargetCapture, ...]]:
    """Acquire several physical-target locks in a deterministic global order."""
    captures = [capture_target(target) for target in targets]
    ordered = sorted(enumerate(captures), key=lambda item: str(item[1].physical))
    lock_root = _ensure_private_lock_root(Path(home))
    acquired: list[tuple[int, TargetCapture, object]] = []
    try:
        for index, capture in ordered:
            manager = _interprocess_lock_capture(
                capture,
                lock_root=lock_root,
                timeout=timeout,
            )
            locked = manager.__enter__()
            acquired.append((index, locked, manager))
        result = [None] * len(captures)
        for index, locked, _manager in acquired:
            result[index] = locked
        yield tuple(result)  # type: ignore[arg-type]
    finally:
        for _index, _locked, manager in reversed(acquired):
            manager.__exit__(None, None, None)


def _physical_alias_key(capture: TargetCapture) -> tuple[str, tuple[int, int] | None]:
    return _canonical_physical_path(capture), capture.identity


def _write_temp(
    physical: Path,
    content: bytes,
    snapshot: FileSnapshot,
    *,
    mode: int | None = None,
) -> Path:
    physical.parent.mkdir(parents=False, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(physical.parent), prefix=f".{physical.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        publication_mode = mode if mode is not None else snapshot.mode
        if publication_mode is not None:
            os.chmod(temp_name, publication_mode)
        if snapshot.owner is not None and hasattr(os, "chown"):
            try:
                os.chown(temp_name, *snapshot.owner)
            except PermissionError:
                if snapshot.owner[0] != getattr(os, "geteuid", lambda: -1)():
                    raise
        return Path(temp_name)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _restore_snapshot(snapshot: FileSnapshot) -> None:
    path = snapshot.physical if snapshot.kind == "symlink" else snapshot.target
    if not snapshot.exists:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        _fsync_directory(path.parent)
        return
    if snapshot.kind == "symlink":
        if snapshot.content is None:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            _fsync_directory(path.parent)
        else:
            rollback = _write_temp(path, snapshot.content, snapshot)
            try:
                os.replace(rollback, path)
                rollback = None  # type: ignore[assignment]
                if snapshot.owner is not None and hasattr(os, "chown"):
                    try:
                        os.chown(path, *snapshot.owner)
                    except PermissionError:
                        pass
                if snapshot.mode is not None:
                    os.chmod(path, snapshot.mode)
                _fsync_directory(path.parent)
            finally:
                if rollback is not None:
                    try:
                        os.unlink(rollback)
                    except OSError:
                        pass

        assert snapshot.link_text is not None
        link_is_original = False
        try:
            link_is_original = (
                stat.S_ISLNK(os.lstat(snapshot.target).st_mode)
                and os.readlink(snapshot.target) == snapshot.link_text
            )
        except FileNotFoundError:
            pass
        if not link_is_original:
            try:
                os.unlink(snapshot.target)
            except FileNotFoundError:
                pass
            os.symlink(snapshot.link_text, snapshot.target)
            _fsync_directory(snapshot.target.parent)
        return
    assert snapshot.content is not None
    rollback = _write_temp(path, snapshot.content, snapshot)
    try:
        os.replace(rollback, path)
        rollback = None  # type: ignore[assignment]
        if snapshot.owner is not None and hasattr(os, "chown"):
            try:
                os.chown(path, *snapshot.owner)
            except PermissionError:
                pass
        if snapshot.mode is not None:
            os.chmod(path, snapshot.mode)
        _fsync_directory(path.parent)
    finally:
        if rollback is not None:
            try:
                os.unlink(rollback)
            except OSError:
                pass


_JOURNAL_PREFIX = "transaction-"
_JOURNAL_SUFFIX = ".json"
_JOURNAL_VERSION = 1
_ACTIVE_TRANSACTION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "config_store_active_transaction_depth",
    default=0,
)


def _snapshot_to_journal(snapshot: FileSnapshot) -> dict[str, object]:
    return {
        "target": str(snapshot.target),
        "physical": str(snapshot.physical),
        "exists": snapshot.exists,
        "kind": snapshot.kind,
        "content": (
            base64.b64encode(snapshot.content).decode("ascii")
            if snapshot.content is not None
            else None
        ),
        "mode": snapshot.mode,
        "owner": list(snapshot.owner) if snapshot.owner is not None else None,
        "link_text": snapshot.link_text,
    }


def _snapshot_from_journal(raw: object) -> FileSnapshot:
    if not isinstance(raw, dict):
        raise ConfigTransactionError("invalid configuration transaction journal entry")
    try:
        target = Path(raw["target"])
        physical = Path(raw["physical"])
        exists = raw["exists"]
        kind = raw["kind"]
        encoded = raw["content"]
        mode = raw["mode"]
        owner_raw = raw["owner"]
        link_text = raw["link_text"]
    except KeyError as exc:
        raise ConfigTransactionError(
            "incomplete configuration transaction journal entry"
        ) from exc
    if not target.is_absolute() or not physical.is_absolute():
        raise ConfigTransactionError("configuration transaction journal uses relative paths")
    if not isinstance(exists, bool) or kind not in {"absent", "regular", "symlink"}:
        raise ConfigTransactionError("invalid configuration transaction journal state")
    if encoded is None:
        content = None
    elif isinstance(encoded, str):
        try:
            content = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ConfigTransactionError(
                "invalid configuration transaction journal content"
            ) from exc
    else:
        raise ConfigTransactionError("invalid configuration transaction journal content")
    if mode is not None and (not isinstance(mode, int) or mode < 0 or mode > 0o7777):
        raise ConfigTransactionError("invalid configuration transaction journal mode")
    owner: tuple[int, int] | None
    if owner_raw is None:
        owner = None
    elif (
        isinstance(owner_raw, list)
        and len(owner_raw) == 2
        and all(isinstance(value, int) and value >= 0 for value in owner_raw)
    ):
        owner = (owner_raw[0], owner_raw[1])
    else:
        raise ConfigTransactionError("invalid configuration transaction journal owner")
    if link_text is not None and not isinstance(link_text, str):
        raise ConfigTransactionError("invalid configuration transaction journal symlink")
    if exists != (kind != "absent"):
        raise ConfigTransactionError("inconsistent configuration transaction journal state")
    if kind == "absent" and any(
        value is not None for value in (content, mode, owner, link_text)
    ):
        raise ConfigTransactionError("invalid absent transaction journal snapshot")
    if kind == "regular" and (content is None or link_text is not None):
        raise ConfigTransactionError("invalid regular transaction journal snapshot")
    if kind == "symlink" and link_text is None:
        raise ConfigTransactionError("invalid symlink transaction journal snapshot")
    return FileSnapshot(
        target=target,
        physical=physical,
        exists=exists,
        kind=kind,
        content=content,
        mode=mode,
        owner=owner,
        link_text=link_text,
    )


def _write_transaction_journal(
    journal_root: Path,
    snapshots: Sequence[FileSnapshot],
) -> Path:
    journal_root = _absolute_without_resolving(journal_root)
    name = f"{_JOURNAL_PREFIX}{os.getpid()}-{uuid.uuid4().hex}{_JOURNAL_SUFFIX}"
    journal = journal_root / name
    temp = journal.with_suffix(f"{journal.suffix}.tmp")
    payload = json.dumps(
        {
            "version": _JOURNAL_VERSION,
            "snapshots": [_snapshot_to_journal(snapshot) for snapshot in snapshots],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    try:
        if not _IS_WINDOWS and hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, journal)
        _fsync_directory(journal_root)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return journal


def _remove_transaction_journal(journal: Path) -> None:
    journal.unlink()
    _fsync_directory(journal.parent)


def _read_transaction_journal(journal: Path) -> list[FileSnapshot]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(journal, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ConfigTransactionError("configuration transaction journal is not regular")
        if not _IS_WINDOWS and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ConfigTransactionError("configuration transaction journal is not private")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigTransactionError("cannot read configuration transaction journal") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if not isinstance(payload, dict) or payload.get("version") != _JOURNAL_VERSION:
        raise ConfigTransactionError("unsupported configuration transaction journal")
    raw_snapshots = payload.get("snapshots")
    if not isinstance(raw_snapshots, list) or len(raw_snapshots) < 2:
        raise ConfigTransactionError("invalid configuration transaction journal snapshot list")
    snapshots = [_snapshot_from_journal(raw) for raw in raw_snapshots]
    targets = [snapshot.target for snapshot in snapshots]
    if len(set(targets)) != len(targets):
        raise ConfigTransactionError("configuration transaction journal has duplicate targets")
    return snapshots


def recover_incomplete_transactions(
    home: str | os.PathLike[str] | Path,
) -> None:
    """Roll back durable multi-file journals before config state is consumed."""
    if _ACTIVE_TRANSACTION_DEPTH.get() > 0:
        return
    lock_home = _absolute_without_resolving(Path(home))
    if lock_home.parent.name == "profiles":
        lock_home = lock_home.parent.parent
    candidate_root = lock_home / ".config-locks"
    try:
        os.lstat(candidate_root)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UnsafeConfigPathError(
            f"Cannot inspect private lock root {candidate_root}: {exc}"
        ) from exc
    journal_root = _ensure_private_lock_root(Path(home))
    journals = sorted(
        journal_root.glob(f"{_JOURNAL_PREFIX}*{_JOURNAL_SUFFIX}"),
        key=lambda path: path.name,
    )
    for journal in journals:
        try:
            snapshots = _read_transaction_journal(journal)
        except FileNotFoundError:
            continue
        targets = [snapshot.target for snapshot in snapshots]
        with interprocess_locks(targets, home=home):
            try:
                snapshots = _read_transaction_journal(journal)
            except FileNotFoundError:
                continue
            for snapshot in reversed(snapshots):
                _restore_snapshot(snapshot)
            _remove_transaction_journal(journal)


def _normalize_targets(
    targets: Sequence[str | os.PathLike[str] | Path],
    *,
    home: str | os.PathLike[str] | Path,
) -> list[Path]:
    normalized = [_absolute_without_resolving(Path(path)) for path in targets]
    if len(set(normalized)) != len(normalized):
        raise ConfigTransactionError("configuration transaction contains duplicate targets")
    captures = [capture_target(path) for path in normalized]
    keys = [_physical_alias_key(capture) for capture in captures]
    if len(set(keys)) != len(keys):
        raise ConfigTransactionError("configuration transaction contains physical aliases")
    for target in normalized:
        _validate_existing_ancestors(target.parent, root=Path(home), include_leaf=True)
        if not target.parent.is_dir():
            raise UnsafeConfigPathError(
                f"Configuration parent is not a directory: {target.parent}"
            )
    return normalized


def _locked_snapshots(captures: Sequence[TargetCapture]) -> list[FileSnapshot]:
    locked_keys = [_physical_alias_key(capture) for capture in captures]
    if len(set(locked_keys)) != len(locked_keys):
        raise ConfigTransactionError("configuration transaction contains physical aliases")
    snapshots: list[FileSnapshot] = []
    for capture in captures:
        capture.verify_unchanged()
        snapshot = snapshot_file(capture.target)
        if snapshot.physical != capture.physical:
            raise UnsafeConfigPathError(
                f"Configuration target changed while locked: {capture.target}"
            )
        snapshots.append(snapshot)
    return snapshots


def _publish_locked(
    targets: Sequence[Path],
    captures: Sequence[TargetCapture],
    snapshots: Sequence[FileSnapshot],
    updates: Mapping[Path, bytes],
    after_publish: Callable[[], None] | None = None,
    modes: Mapping[Path, int] | None = None,
    journal_root: Path | None = None,
) -> None:
    temps: list[Path] = []
    selected: list[tuple[Path, TargetCapture, FileSnapshot, Path]] = []
    published = 0
    callback_failed = False
    journal: Path | None = None
    active_token: contextvars.Token | None = None
    try:
        for target, capture, snapshot in zip(targets, captures, snapshots):
            if target not in updates:
                continue
            mode = modes.get(target) if modes is not None else None
            temp = _write_temp(
                snapshot.physical,
                bytes(updates[target]),
                snapshot,
                mode=mode,
            )
            temps.append(temp)
            selected.append((target, capture, snapshot, temp))
        if len(selected) > 1:
            if journal_root is None:
                raise ConfigTransactionError(
                    "multi-file configuration publication requires a journal root"
                )
            journal = _write_transaction_journal(
                journal_root,
                [snapshot for _target, _capture, snapshot, _temp in selected],
            )
            active_token = _ACTIVE_TRANSACTION_DEPTH.set(
                _ACTIVE_TRANSACTION_DEPTH.get() + 1
            )
        for target, capture, snapshot, temp in selected:
            capture.verify_unchanged()
            mode = modes.get(target) if modes is not None else None
            if mode is None:
                atomic_replace(temp, snapshot.physical, follow_symlinks=False)
            else:
                atomic_replace(
                    temp,
                    snapshot.physical,
                    mode=mode,
                    follow_symlinks=False,
                )
            published += 1
            if capture.kind == "symlink":
                capture.verify_unchanged(verify_physical=False)
        for _target, _capture, snapshot, _temp in selected:
            _fsync_directory(snapshot.physical.parent)
        if after_publish is not None:
            try:
                after_publish()
            except BaseException:
                callback_failed = True
                raise
        if journal is not None:
            _remove_transaction_journal(journal)
            journal = None
    except BaseException as exc:
        rollback_errors: list[BaseException] = []
        if published:
            for _target, _capture, snapshot, _temp in reversed(
                selected[:published]
            ):
                try:
                    _restore_snapshot(snapshot)
                except BaseException as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if rollback_errors:
                raise ConfigTransactionError(
                    f"configuration publish failed and rollback failed: {rollback_errors[0]}"
                ) from exc
        if journal is not None:
            try:
                _remove_transaction_journal(journal)
                journal = None
            except BaseException as cleanup_exc:
                raise ConfigTransactionError(
                    "configuration publish failed and journal cleanup failed"
                ) from cleanup_exc
        if callback_failed:
            raise _AfterPublishError(exc) from exc
        if isinstance(exc, ConfigTransactionError):
            raise
        raise ConfigTransactionError(f"configuration publish failed: {exc}") from exc
    finally:
        if active_token is not None:
            _ACTIVE_TRANSACTION_DEPTH.reset(active_token)
        for temp in temps:
            try:
                temp.unlink()
            except OSError:
                pass


def update_transaction(
    targets: Sequence[str | os.PathLike[str] | Path],
    updater: Callable[
        [dict[Path, bytes | None]],
        dict[Path, bytes],
    ],
    *,
    home: str | os.PathLike[str] | Path,
    surface: str = "config_store.update_transaction",
    after_publish: Callable[[], None] | None = None,
    modes: Mapping[str | os.PathLike[str] | Path, int] | None = None,
) -> None:
    """Read, transform, publish, then validate several files under one lock."""
    recover_incomplete_transactions(home)
    normalized_targets = _normalize_targets(targets, home=home)
    normalized_modes = {
        _absolute_without_resolving(Path(path)): int(mode)
        for path, mode in (modes or {}).items()
    }
    unknown_modes = set(normalized_modes) - set(normalized_targets)
    if unknown_modes:
        raise ConfigTransactionError("transaction modes contain unlocked targets")
    if not normalized_targets:
        return
    before_contents: list[bytes | None] = []
    try:
        with interprocess_locks(normalized_targets, home=home) as captures:
            snapshots = _locked_snapshots(captures)
            before_contents = [snapshot.content for snapshot in snapshots]
            current = {
                target: snapshot.content
                for target, snapshot in zip(normalized_targets, snapshots)
            }
            raw_updates = updater(current)
            updates = {
                _absolute_without_resolving(Path(path)): bytes(content)
                for path, content in raw_updates.items()
            }
            unknown = set(updates) - set(normalized_targets)
            if unknown:
                raise ConfigTransactionError(
                    "transaction updater returned unlocked targets: "
                    + ", ".join(sorted(map(str, unknown)))
                )
            _publish_locked(
                normalized_targets,
                captures,
                snapshots,
                updates,
                after_publish=after_publish,
                modes=normalized_modes,
                journal_root=_ensure_private_lock_root(Path(home)),
            )
    except BaseException as exc:
        _emit_write_provenance(
            surface=surface,
            targets=normalized_targets,
            before_contents=before_contents,
            result="failed",
        )
        if isinstance(exc, _AfterPublishError):
            raise exc.cause
        if isinstance(exc, ConfigTransactionError):
            raise
        raise ConfigTransactionError(f"configuration transaction failed: {exc}") from exc
    _emit_write_provenance(
        surface=surface,
        targets=normalized_targets,
        before_contents=before_contents,
        result="success",
    )


def publish_locked_capture(
    capture: TargetCapture,
    content: bytes,
    *,
    mode: int | None = None,
) -> TargetCapture:
    """Publish one target and return a refreshed capture for reentrant writes."""
    snapshots = _locked_snapshots([capture])
    _publish_locked(
        [capture.target],
        [capture],
        snapshots,
        {capture.target: bytes(content)},
        modes={capture.target: mode} if mode is not None else None,
    )

    refreshed = capture_target(capture.target)
    if refreshed.physical != capture.physical:
        raise UnsafeConfigPathError(
            f"Configuration target changed after publication: {capture.target}"
        )
    if capture.kind == "symlink":
        if refreshed.kind != "symlink" or refreshed.link_text != capture.link_text:
            raise UnsafeConfigPathError(
                f"Configuration symlink changed after publication: {capture.target}"
            )
    elif refreshed.kind != "regular":
        raise UnsafeConfigPathError(
            f"Configuration target changed type after publication: {capture.target}"
        )
    return refreshed


def publish_transaction(
    updates: Mapping[str | os.PathLike[str] | Path, bytes],
    *,
    home: str | os.PathLike[str] | Path,
    surface: str = "config_store.publish_transaction",
    modes: Mapping[str | os.PathLike[str] | Path, int] | None = None,
) -> None:
    """Publish precomputed files all-or-nothing with exact rollback."""
    if not updates:
        return
    normalized_updates = {
        _absolute_without_resolving(Path(path)): bytes(content)
        for path, content in updates.items()
    }
    update_transaction(
        list(normalized_updates),
        lambda _current: normalized_updates,
        home=home,
        surface=surface,
        modes=modes,
    )


class ConfigTransaction:
    """Object-oriented spelling for :func:`publish_transaction`."""

    def __init__(
        self,
        updates: Mapping[str | os.PathLike[str] | Path, bytes],
        *,
        home: str | os.PathLike[str] | Path,
        surface: str = "config_store.ConfigTransaction",
    ):
        self.updates = updates
        self.home = home
        self.surface = surface

    def commit(self) -> None:
        publish_transaction(self.updates, home=self.home, surface=self.surface)
