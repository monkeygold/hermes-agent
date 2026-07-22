"""R11 permanent allowlist inter-process transaction tests."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
from pathlib import Path

import pytest
import yaml


def _approve_worker(home: str, pattern: str, start, results) -> None:
    os.environ["HERMES_HOME"] = home
    try:
        start.wait(timeout=10)
        from tools.approval import approve_always

        approve_always({pattern}, session_key=f"worker:{pattern}")
        results.put((pattern, None))
    except BaseException as exc:  # pragma: no cover - surfaced in parent
        results.put((pattern, f"{type(exc).__name__}: {exc}"))


def test_concurrent_always_merges_every_pattern(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text(
        "model:\n  default: test/model\n",
        encoding="utf-8",
    )
    patterns = {f"terminal:bounded-{index}" for index in range(6)}
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_approve_worker,
            args=(str(home), pattern, start, results),
        )
        for pattern in sorted(patterns)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert not process.is_alive(), "allowlist worker deadlocked"
        assert process.exitcode == 0

    outcomes = [results.get(timeout=2) for _ in processes]
    assert all(error is None for _pattern, error in outcomes), outcomes
    persisted = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert set(persisted["command_allowlist"]) == patterns


def test_persistence_uses_normalized_physical_key_for_symlink(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    physical = home / "physical-config.yaml"
    physical.write_text("command_allowlist:\n  - terminal:existing\n", encoding="utf-8")
    config = home / "config.yaml"
    try:
        config.symlink_to(physical)
    except OSError:
        pytest.skip("symlinks are not available")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.approval as approval

    persisted = approval._persist_permanent_allowlist(
        {"terminal:new"},
        merge=True,
    )

    assert persisted == {"terminal:existing", "terminal:new"}
    assert config.is_symlink()
    parsed = yaml.safe_load(physical.read_text(encoding="utf-8"))
    assert set(parsed["command_allowlist"]) == persisted


def test_hot_process_revokes_same_inode_size_and_mtime(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config = home / "config.yaml"
    before = "command_allowlist:\n  - terminal:alpha\n"
    after = "command_allowlist:\n  - terminal:bravo\n"
    assert len(before) == len(after)
    config.write_text(before, encoding="utf-8")
    original_stat = config.stat()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.approval as approval

    approval.load_permanent_allowlist()
    assert approval.is_approved("session", "terminal:alpha") is True

    config.write_text(after, encoding="utf-8")
    os.utime(
        config,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    current_stat = config.stat()
    assert current_stat.st_ino == original_stat.st_ino
    assert current_stat.st_size == original_stat.st_size
    assert current_stat.st_mtime_ns == original_stat.st_mtime_ns

    assert approval.is_approved("session", "terminal:alpha") is False
    assert approval.is_approved("session", "terminal:bravo") is True


def test_permanent_allowlist_persistence_error_logs_only_exception_class(
    monkeypatch, caplog,
) -> None:
    import tools.approval as approval

    secret_detail = "token=must-not-enter-logs"

    def fail_persistence(*_args, **_kwargs):
        raise RuntimeError(secret_detail)

    monkeypatch.setattr(approval, "_persist_permanent_allowlist", fail_persistence)

    with pytest.raises(approval.AllowlistPersistenceError):
        approval.approve_always({"terminal:bounded"})

    rendered = "\n".join(caplog.handler.format(record) for record in caplog.records)
    assert secret_detail not in rendered
    assert "RuntimeError" in rendered


def test_hot_reload_never_tags_stale_allowlist_with_new_file_signature(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_path = home / "config.yaml"
    config_path.write_text(
        "command_allowlist:\n  - terminal:old-value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module
    import tools.approval as approval

    real_load_config = config_module.load_config
    raced = False

    def racing_load_config():
        nonlocal raced
        raced = True
        config_path.write_text(
            "command_allowlist:\n  - terminal:new-value\n",
            encoding="utf-8",
        )
        return {"command_allowlist": ["terminal:old-value"]}

    monkeypatch.setattr(config_module, "load_config", racing_load_config)
    approval.load_permanent_allowlist()
    if not raced:
        config_path.write_text(
            "command_allowlist:\n  - terminal:new-value\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(config_module, "load_config", real_load_config)

    approval._refresh_permanent_allowlist_if_changed()

    assert approval.is_approved("session", "terminal:old-value") is False
    assert approval.is_approved("session", "terminal:new-value") is True


def test_permanent_allowlist_is_isolated_between_concurrent_home_contexts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_constants
    import tools.approval as approval

    approved_home = tmp_path / "approved"
    denied_home = tmp_path / "denied"
    approved_home.mkdir()
    denied_home.mkdir()
    pattern = "terminal:profile-a-only"
    (approved_home / "config.yaml").write_text(
        f"command_allowlist:\n  - {pattern}\n",
        encoding="utf-8",
    )
    (denied_home / "config.yaml").write_text(
        "command_allowlist: []\n",
        encoding="utf-8",
    )

    read_barrier = threading.Barrier(2)
    publish_barrier = threading.Barrier(2)
    real_read = approval._read_allowlist_config_snapshot
    real_load = approval.load_permanent

    def synchronized_read():
        result = real_read()
        read_barrier.wait(timeout=10)
        return result

    def synchronized_load(patterns):
        real_load(patterns)
        publish_barrier.wait(timeout=10)

    monkeypatch.setattr(approval, "_read_allowlist_config_snapshot", synchronized_read)
    monkeypatch.setattr(approval, "load_permanent", synchronized_load)
    monkeypatch.setattr(approval, "_last_allowlist_config_signature", object())
    results: dict[str, bool] = {}
    errors: list[BaseException] = []

    def check(label: str, home: Path) -> None:
        token = hermes_constants.set_hermes_home_override(home)
        try:
            results[label] = approval.is_approved(label, pattern)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            hermes_constants.reset_hermes_home_override(token)

    threads = [
        threading.Thread(target=check, args=("approved", approved_home)),
        threading.Thread(target=check, args=("denied", denied_home)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert results == {"approved": True, "denied": False}
