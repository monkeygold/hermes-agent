from __future__ import annotations

import errno
import hashlib
import json
import logging
import multiprocessing
import os
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest


def _crash_after_first_transaction_publish(home: str) -> None:
    import hermes_cli.config_store as store

    config = Path(home) / "config.yaml"
    env = Path(home) / ".env"
    real_atomic_replace = store.atomic_replace
    calls = 0

    def replace_then_crash(source, target, **kwargs):
        nonlocal calls
        result = real_atomic_replace(source, target, **kwargs)
        calls += 1
        if calls == 1:
            os._exit(86)
        return result

    store.atomic_replace = replace_then_crash
    store.publish_transaction(
        {
            config: b"model:\n  default: after/model\n",
            env: b"TOKEN=after\n",
        },
        home=home,
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _hold_lock(path: str, home: str, ready, release) -> None:
    from hermes_cli.config_store import interprocess_lock

    with interprocess_lock(Path(path), home=Path(home)):
        ready.set()
        release.wait(10)


def _probe_lock(path: str, home: str, acquired) -> None:
    from hermes_cli.config_store import interprocess_lock

    with interprocess_lock(Path(path), home=Path(home), timeout=5):
        acquired.set()


def _hold_ordered_locks(
    paths: tuple[str, str], home: str, ready, done,
) -> None:
    from hermes_cli.config_store import interprocess_locks

    with interprocess_locks(
        [Path(paths[0]), Path(paths[1])],
        home=Path(home),
        timeout=5,
    ):
        ready.set()
        time.sleep(0.15)
    done.set()


def test_interprocess_lock_uses_windows_backend_without_posix_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    home.chmod(0o777)
    target = home / "config.yaml"
    target.write_text("x\n", encoding="utf-8")
    calls: list[tuple[int, int]] = []
    fake_msvcrt = SimpleNamespace(
        LK_NBLCK=3,
        LK_UNLCK=2,
        locking=lambda _fd, mode, length: calls.append((mode, length)),
    )
    monkeypatch.setattr(store, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(store, "fcntl", None)
    monkeypatch.setattr(store, "msvcrt", fake_msvcrt, raising=False)

    with store.interprocess_lock(target, home=home):
        pass

    assert calls == [(fake_msvcrt.LK_NBLCK, 1), (fake_msvcrt.LK_UNLCK, 1)]


def test_windows_case_aliases_share_one_lock_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import hermes_cli.config_store as store

    monkeypatch.setattr(store, "_IS_WINDOWS", True)
    upper = store.TargetCapture(
        Path("C:/Users/Alice/.hermes/config.yaml"),
        Path("C:/Users/Alice/.hermes/config.yaml"),
        "absent",
        None,
        None,
    )
    lower = store.TargetCapture(
        Path("c:/users/alice/.HERMES/CONFIG.YAML"),
        Path("c:/users/alice/.HERMES/CONFIG.YAML"),
        "absent",
        None,
        None,
    )

    assert store._lock_name(upper) == store._lock_name(lower)


def test_interprocess_lock_is_shared_by_symlink_aliases(tmp_path: Path) -> None:
    from hermes_cli.config_store import interprocess_lock

    home = tmp_path / "home"
    home.mkdir()
    physical = home / "real.yaml"
    physical.write_text("old\n", encoding="utf-8")
    alias = home / "config.yaml"
    alias.symlink_to(physical)

    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(
        target=_hold_lock,
        args=(str(alias), str(home), ready, release),
    )
    probe_acquired = multiprocessing.Event()
    probe = multiprocessing.Process(
        target=_probe_lock,
        args=(str(physical), str(home), probe_acquired),
    )
    process.start()
    try:
        assert ready.wait(5), "child did not acquire the physical-target lock"
        probe.start()
        assert not probe_acquired.wait(0.2), "physical aliases must contend"
    finally:
        release.set()
        process.join(10)
        probe.join(10)
        if process.is_alive():
            process.kill()
        if probe.is_alive():
            probe.kill()
    assert process.exitcode == 0
    assert probe.exitcode == 0
    assert probe_acquired.is_set()


def test_interprocess_lock_rejects_symlink_lock_root_before_open(tmp_path: Path) -> None:
    from hermes_cli.config_store import UnsafeConfigPathError, interprocess_lock

    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (home / ".config-locks").symlink_to(outside, target_is_directory=True)
    target = home / "config.yaml"
    target.write_text("x\n", encoding="utf-8")

    with pytest.raises(UnsafeConfigPathError, match="symlink"):
        with interprocess_lock(target, home=home):
            pass
    assert not list(outside.iterdir()), "a symlinked lock root must never receive files"


def test_interprocess_lock_rejects_symlinked_home_ancestor_before_open(
    tmp_path: Path,
) -> None:
    from hermes_cli.config_store import UnsafeConfigPathError, interprocess_lock

    physical_parent = tmp_path / "physical-parent"
    physical_parent.mkdir(mode=0o700)
    logical_parent = tmp_path / "logical-parent"
    logical_parent.symlink_to(physical_parent, target_is_directory=True)
    home = logical_parent / "home"
    home.mkdir(mode=0o700)
    target = home / "config.yaml"
    target.write_text("x\n", encoding="utf-8")

    with pytest.raises(UnsafeConfigPathError, match="symlink ancestor"):
        with interprocess_lock(target, home=home):
            pass
    assert not (physical_parent / "home" / ".config-locks").exists()


def test_interprocess_locks_have_global_physical_order(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    left = home / "left"
    right = home / "right"
    left.write_text("l\n", encoding="utf-8")
    right.write_text("r\n", encoding="utf-8")

    context = multiprocessing.get_context("spawn")
    ready_a = context.Event()
    ready_b = context.Event()
    done_a = context.Event()
    done_b = context.Event()
    a = context.Process(
        target=_hold_ordered_locks,
        args=((str(left), str(right)), str(home), ready_a, done_a),
    )
    b = context.Process(
        target=_hold_ordered_locks,
        args=((str(right), str(left)), str(home), ready_b, done_b),
    )
    a.start()
    b.start()
    try:
        assert ready_a.wait(5) or ready_b.wait(5)
        assert done_a.wait(10)
        assert done_b.wait(10)
    finally:
        for process in (a, b):
            process.join(10)
            if process.is_alive():
                process.kill()
    assert a.exitcode == 0
    assert b.exitcode == 0


def test_atomic_replace_ebusy_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from utils import atomic_replace

    target = tmp_path / "config.yaml"
    source = tmp_path / ".source.tmp"
    target.write_bytes(b"old")
    source.write_bytes(b"new")

    def busy(*_args, **_kwargs):
        raise OSError(errno.EBUSY, "busy")

    monkeypatch.setattr("utils.os.replace", busy)
    with pytest.raises(OSError) as exc_info:
        atomic_replace(source, target)
    assert exc_info.value.errno == errno.EBUSY
    assert target.read_bytes() == b"old"
    assert source.read_bytes() == b"new"


def test_atomic_replace_exdev_uses_sibling_replace_not_in_place_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from utils import atomic_replace

    target = tmp_path / "config.yaml"
    source = tmp_path / ".source.tmp"
    target.write_bytes(b"old")
    source.write_bytes(b"new")
    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def replace_with_one_exdev(src, dst, *args, **kwargs):
        calls.append((str(src), str(dst)))
        if len(calls) == 1:
            raise OSError(errno.EXDEV, "cross-device")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr("utils.os.replace", replace_with_one_exdev)
    atomic_replace(source, target)
    assert target.read_bytes() == b"new"
    assert not source.exists()
    assert len(calls) == 2
    assert calls[0][0] == str(source)
    assert calls[1][1] == str(target)
    assert Path(calls[1][0]).parent == target.parent
    assert Path(calls[1][0]) != target


def test_set_env_bytes_replaces_exported_assignment_without_duplicate() -> None:
    from hermes_cli.config import _set_env_bytes

    updated, clean = _set_env_bytes(
        b"export TEST_R11_KEY=old\nOTHER=value\n",
        "TEST_R11_KEY",
        "new",
    )

    assert clean == "new"
    assert updated == b"export TEST_R11_KEY=new\nOTHER=value\n"
    assert updated.count(b"TEST_R11_KEY=") == 1


def test_set_env_bytes_quotes_values_containing_tabs() -> None:
    from hermes_cli.config import _set_env_bytes

    updated, clean = _set_env_bytes(None, "TEST_R11_KEY", "left\tright")

    assert clean == "left\tright"
    assert updated == b'TEST_R11_KEY="left\tright"\n'


def test_remove_env_bytes_removes_exported_assignment() -> None:
    from hermes_cli.config import _remove_env_bytes

    updated, removed = _remove_env_bytes(
        b"export TEST_R11_KEY=old\nOTHER=value\n",
        "TEST_R11_KEY",
    )

    assert removed is True
    assert updated == b"OTHER=value\n"


def test_publish_transaction_rejects_physical_aliases_before_mutation(tmp_path: Path) -> None:
    from hermes_cli.config_store import ConfigTransactionError, publish_transaction

    home = tmp_path / "home"
    home.mkdir()
    physical = home / "physical"
    physical.write_bytes(b"original")
    config = home / "config.yaml"
    env = home / ".env"
    config.symlink_to(physical)
    env.symlink_to(physical)

    with pytest.raises(ConfigTransactionError, match="alias"):
        publish_transaction({config: b"config", env: b"env"}, home=home)
    assert physical.read_bytes() == b"original"
    assert config.is_symlink()
    assert env.is_symlink()


def test_publish_transaction_rejects_hardlinked_targets_before_mutation(
    tmp_path: Path,
) -> None:
    from hermes_cli.config_store import ConfigStoreError, publish_transaction

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    env = home / ".env"
    config.write_bytes(b"shared-before")
    os.link(config, env)

    with pytest.raises(ConfigStoreError, match="hardlink|alias"):
        publish_transaction({config: b"config-after", env: b"env-after"}, home=home)

    assert config.read_bytes() == b"shared-before"
    assert env.read_bytes() == b"shared-before"


def test_publish_transaction_rolls_back_exact_snapshot_on_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.config_store import ConfigTransactionError, publish_transaction
    from utils import atomic_replace as real_atomic_replace

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    env = home / ".env"
    config.write_bytes(b"config-before\x00\xff")
    os.chmod(config, 0o640)
    before = (config.read_bytes(), config.stat().st_mode & 0o777)
    calls = 0

    def fail_second(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EBUSY, "busy")
        return real_atomic_replace(source, target, **kwargs)

    monkeypatch.setattr("hermes_cli.config_store.atomic_replace", fail_second)
    with pytest.raises(ConfigTransactionError, match="publish"):
        publish_transaction({config: b"config-after", env: b"env-after"}, home=home)
    assert config.read_bytes() == before[0]
    assert config.stat().st_mode & 0o777 == before[1]
    assert not env.exists()
    assert not list(home.glob(".*.tmp"))


def test_config_read_recovers_journal_after_process_crash_mid_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    env = home / ".env"
    config_before = b"model:\n  default: before/model\n"
    env_before = b"TOKEN=before\n"
    config.write_bytes(config_before)
    env.write_bytes(env_before)

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_crash_after_first_transaction_publish,
        args=(str(home),),
    )
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 86
    assert config.read_bytes() == b"model:\n  default: after/model\n"
    assert env.read_bytes() == env_before
    journals = list((home / ".config-locks").glob("transaction-*.json"))
    assert len(journals) == 1
    assert journals[0].stat().st_mode & 0o777 == 0o600

    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import read_raw_config

    loaded = read_raw_config()

    assert loaded["model"]["default"] == "before/model"
    assert config.read_bytes() == config_before
    assert env.read_bytes() == env_before
    assert not list((home / ".config-locks").glob("transaction-*.json"))


def test_config_read_does_not_deadlock_with_transaction_after_publish_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config as config_module
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    env = home / ".env"
    config.write_bytes(b"model:\n  default: before/model\n")
    env.write_bytes(b"TOKEN=before\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    config_module.invalidate_config_caches(config)

    callback_entered = threading.Event()
    reader_entered_recovery = threading.Event()
    errors: list[str] = []
    real_recover = store.recover_incomplete_transactions

    def tracked_recover(recovery_home):
        if threading.current_thread().name == "r11-config-reader":
            reader_entered_recovery.set()
        return real_recover(recovery_home)

    monkeypatch.setattr(store, "recover_incomplete_transactions", tracked_recover)

    def after_publish() -> None:
        callback_entered.set()
        if not reader_entered_recovery.wait(timeout=5):
            raise AssertionError("reader did not enter recovery")
        config_module.read_raw_config()

    def callback_writer() -> None:
        try:
            store.update_transaction(
                [config, env],
                lambda _current: {
                    config: b"model:\n  default: callback/model\n",
                    env: b"TOKEN=callback\n",
                },
                home=home,
                after_publish=after_publish,
            )
        except BaseException as exc:
            errors.append(f"writer: {exc!r}\n{traceback.format_exc()}")

    def reader() -> None:
        try:
            if not callback_entered.wait(timeout=5):
                raise AssertionError("transaction callback did not start")
            config_module.read_raw_config()
        except BaseException as exc:
            errors.append(f"reader: {exc!r}\n{traceback.format_exc()}")

    # Keep the threads daemonized so a RED deadlock cannot hang the pytest process.
    writer_thread = threading.Thread(
        target=callback_writer,
        name="r11-config-writer",
        daemon=True,
    )
    reader_thread = threading.Thread(
        target=reader,
        name="r11-config-reader",
        daemon=True,
    )
    writer_thread.start()
    reader_thread.start()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not writer_thread.is_alive(), "writer deadlocked acquiring _CONFIG_LOCK"
    assert not reader_thread.is_alive(), "reader deadlocked acquiring transaction locks"
    assert not errors, "\n".join(errors)


def test_publish_transaction_preserves_symlink_and_fsyncs_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.config_store import publish_transaction

    home = tmp_path / "home"
    home.mkdir()
    physical_config = home / "physical-config.yaml"
    physical_env = home / "physical.env"
    physical_config.write_bytes(b"old-config")
    physical_env.write_bytes(b"old-env")
    config = home / "config.yaml"
    env = home / ".env"
    config.symlink_to(physical_config)
    env.symlink_to(physical_env)
    fsynced: list[Path] = []
    monkeypatch.setattr(
        "hermes_cli.config_store._fsync_directory",
        lambda path: fsynced.append(Path(path)),
    )

    publish_transaction({config: b"new-config", env: b"new-env"}, home=home)
    assert config.is_symlink()
    assert env.is_symlink()
    assert physical_config.read_bytes() == b"new-config"
    assert physical_env.read_bytes() == b"new-env"
    assert fsynced


def test_publish_transaction_rolls_back_physical_symlink_target_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.config_store import ConfigTransactionError, publish_transaction
    from utils import atomic_replace as real_atomic_replace

    home = tmp_path / "home"
    home.mkdir()
    physical_config = home / "physical-config.yaml"
    physical_env = home / "physical.env"
    physical_config.write_bytes(b"config-before\x00\xff")
    physical_env.write_bytes(b"env-before")
    os.chmod(physical_config, 0o640)
    config = home / "config.yaml"
    env = home / ".env"
    config.symlink_to(physical_config)
    env.symlink_to(physical_env)
    calls = 0

    def fail_second(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EBUSY, "busy")
        return real_atomic_replace(source, target, **kwargs)

    monkeypatch.setattr("hermes_cli.config_store.atomic_replace", fail_second)
    with pytest.raises(ConfigTransactionError, match="publish"):
        publish_transaction({config: b"config-after", env: b"env-after"}, home=home)

    assert config.is_symlink()
    assert os.readlink(config) == str(physical_config)
    assert physical_config.read_bytes() == b"config-before\x00\xff"
    assert physical_config.stat().st_mode & 0o777 == 0o640
    assert physical_env.read_bytes() == b"env-before"


def test_publish_transaction_rejects_retarget_between_lock_and_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli.config_store import ConfigTransactionError, publish_transaction
    from utils import atomic_replace as real_atomic_replace

    home = tmp_path / "home"
    home.mkdir()
    captured = home / "captured.yaml"
    wrong = home / "wrong.yaml"
    captured.write_bytes(b"captured-before")
    wrong.write_bytes(b"wrong-before")
    config = home / "config.yaml"
    config.symlink_to(captured)

    def retarget_then_publish(source, target, **kwargs):
        config.unlink()
        config.symlink_to(wrong)
        return real_atomic_replace(source, target, **kwargs)

    monkeypatch.setattr(
        "hermes_cli.config_store.atomic_replace", retarget_then_publish
    )
    with pytest.raises(ConfigTransactionError, match="retarget|changed"):
        publish_transaction({config: b"after"}, home=home)

    assert captured.read_bytes() == b"captured-before"
    assert wrong.read_bytes() == b"wrong-before"


def test_publish_transaction_rejects_regular_target_swapped_to_symlink_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    victim = home / "victim.yaml"
    config.write_bytes(b"config-before")
    victim.write_bytes(b"victim-before")
    real_atomic_replace = store.atomic_replace
    swapped = False

    def swap_regular_for_symlink_then_publish(source, target, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            config.unlink()
            config.symlink_to(victim)
        return real_atomic_replace(source, target, **kwargs)

    monkeypatch.setattr(store, "atomic_replace", swap_regular_for_symlink_then_publish)

    with pytest.raises(store.ConfigTransactionError, match="changed|publish"):
        store.publish_transaction({config: b"attacker-selected-write"}, home=home)

    assert victim.read_bytes() == b"victim-before"
    assert config.is_symlink()
    assert config.resolve() == victim.resolve()


def test_transaction_provenance_contains_only_pid_surface_hashes_and_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from hermes_cli.config_store import publish_transaction

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config = home / "config.yaml"
    before = b"api_key: before-secret-value\n"
    after = b"api_key: after-secret-value\n"
    config.write_bytes(before)

    with caplog.at_level(logging.INFO, logger="hermes_cli.config_store"):
        publish_transaction(
            {config: after},
            home=home,
            surface="tests.f01.config-write",
        )

    messages = [
        record.getMessage().removeprefix("config_write_provenance ")
        for record in caplog.records
        if record.getMessage().startswith("config_write_provenance ")
    ]
    assert len(messages) == 1
    event = json.loads(messages[0])
    assert event == {
        "pid": os.getpid(),
        "surface": "tests.f01.config-write",
        "targets": [
            {
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
            }
        ],
        "result": "success",
    }
    serialized = json.dumps(event, sort_keys=True)
    assert "before-secret-value" not in serialized
    assert "after-secret-value" not in serialized
    assert str(config) not in serialized


def test_invalidate_config_caches_removes_three_physical_aliases(
    tmp_path: Path,
) -> None:
    import hermes_cli.config as config_module

    target = tmp_path / "physical-config.yaml"
    target.write_text("model: old\n", encoding="utf-8")
    aliases = [tmp_path / "config.yaml", tmp_path / "second.yaml", tmp_path / "third.yaml"]
    for alias in aliases:
        alias.symlink_to(target)
    unrelated = tmp_path / "unrelated.yaml"
    unrelated.write_text("model: keep\n", encoding="utf-8")

    alias_keys = [str(alias) for alias in aliases]
    unrelated_key = str(unrelated)
    try:
        for key in alias_keys:
            config_module._RAW_CONFIG_CACHE[key] = (1, 1, {"marker": key})
            config_module._LOAD_CONFIG_CACHE[key] = (
                1,
                1,
                1,
                1,
                {"marker": key},
                {},
            )
            config_module._LAST_EXPANDED_CONFIG_BY_PATH[key] = {"marker": key}
        config_module._RAW_CONFIG_CACHE[unrelated_key] = (1, 1, {"keep": True})
        config_module._LOAD_CONFIG_CACHE[unrelated_key] = (
            1,
            1,
            1,
            1,
            {"keep": True},
            {},
        )
        config_module._LAST_EXPANDED_CONFIG_BY_PATH[unrelated_key] = {"keep": True}

        config_module.invalidate_config_caches(aliases[0])

        for key in alias_keys:
            assert key not in config_module._RAW_CONFIG_CACHE
            assert key not in config_module._LOAD_CONFIG_CACHE
            assert key not in config_module._LAST_EXPANDED_CONFIG_BY_PATH
        assert unrelated_key in config_module._RAW_CONFIG_CACHE
        assert unrelated_key in config_module._LOAD_CONFIG_CACHE
        assert unrelated_key in config_module._LAST_EXPANDED_CONFIG_BY_PATH
    finally:
        for key in [*alias_keys, unrelated_key]:
            config_module._RAW_CONFIG_CACHE.pop(key, None)
            config_module._LOAD_CONFIG_CACHE.pop(key, None)
            config_module._LAST_EXPANDED_CONFIG_BY_PATH.pop(key, None)


def test_absent_target_appearing_after_capture_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    target = home / "config.yaml"
    capture = store.capture_target(target)
    assert capture.kind == "absent"
    real_capture_target = store.capture_target

    def create_then_capture(path):
        target.write_bytes(b"uncoordinated-writer")
        return real_capture_target(path)

    monkeypatch.setattr(store, "capture_target", create_then_capture)

    lock_root = store._ensure_private_lock_root(home)
    with pytest.raises(store.UnsafeConfigPathError, match="appeared|changed"):
        with store._interprocess_lock_capture(
            capture, lock_root=lock_root, timeout=1
        ):
            pass

    assert target.read_bytes() == b"uncoordinated-writer"


def test_dangling_symlink_destination_appearing_after_capture_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    physical = home / "physical.yaml"
    target = home / "config.yaml"
    try:
        target.symlink_to(physical)
    except OSError:
        pytest.skip("symlinks are not available")
    capture = store.capture_target(target)
    assert capture.kind == "symlink"
    assert capture.identity is None
    real_verify = store.TargetCapture.verify_unchanged

    def create_then_verify(self, *args, **kwargs):
        if self.target == target and not physical.exists():
            physical.write_bytes(b"uncoordinated-writer")
        return real_verify(self, *args, **kwargs)

    monkeypatch.setattr(store.TargetCapture, "verify_unchanged", create_then_verify)

    lock_root = store._ensure_private_lock_root(home)
    with pytest.raises(store.UnsafeConfigPathError, match="destination|changed"):
        with store._interprocess_lock_capture(
            capture, lock_root=lock_root, timeout=1
        ):
            pass

    assert target.is_symlink()
    assert physical.read_bytes() == b"uncoordinated-writer"


def test_windows_lock_permanent_error_is_not_retried_as_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    target = home / "config.yaml"
    target.write_text("x\n", encoding="utf-8")

    def fail_lock(_fd, _mode, _length):
        raise OSError(errno.EINVAL, "invalid lock operation")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=3, LK_UNLCK=2, locking=fail_lock)
    monkeypatch.setattr(store, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(store, "fcntl", None)
    monkeypatch.setattr(store, "msvcrt", fake_msvcrt, raising=False)

    with pytest.raises(OSError) as exc_info:
        with store.interprocess_lock(target, home=home, timeout=0.01):
            pass
    assert exc_info.value.errno == errno.EINVAL


def test_windows_unlock_error_does_not_turn_committed_body_into_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.config_store as store

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    target = home / "config.yaml"
    target.write_text("before\n", encoding="utf-8")

    def locking(_fd, mode, _length):
        if mode == 2:
            raise OSError(errno.EIO, "unlock failed")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=3, LK_UNLCK=2, locking=locking)
    monkeypatch.setattr(store, "_IS_WINDOWS", True, raising=False)
    monkeypatch.setattr(store, "fcntl", None)
    monkeypatch.setattr(store, "msvcrt", fake_msvcrt, raising=False)

    with store.interprocess_lock(target, home=home, timeout=0.01):
        target.write_text("committed\n", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "committed\n"
