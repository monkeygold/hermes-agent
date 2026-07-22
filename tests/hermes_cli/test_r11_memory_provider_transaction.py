"""R11 memory-provider multi-file transaction regressions."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest
import yaml


def test_memory_provider_transaction_takes_config_lock_before_file_locks(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.config as config_module
    import hermes_cli.config_store as config_store
    import hermes_cli.web_server as web_server

    class Provider:
        def get_config_schema(self):
            return [{"key": "mode", "default": "local"}]

    observed: list[bool] = []

    def probe_config_lock(*_args, **_kwargs):
        def probe() -> None:
            acquired = config_module._CONFIG_LOCK.acquire(blocking=False)
            observed.append(acquired)
            if acquired:
                config_module._CONFIG_LOCK.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()

    monkeypatch.setattr(config_store, "update_transaction", probe_config_lock)

    web_server._write_memory_provider_config_values(
        "test-provider",
        Provider(),
        {"mode": "local"},
    )

    assert observed == [False]


def test_declared_memory_provider_rolls_back_native_file_when_env_write_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    provider_dir = home / "hindsight"
    provider_dir.mkdir(parents=True)
    native = provider_dir / "config.json"
    native_before = b'{"mode":"local_external","extra":"keep"}\n'
    native.write_bytes(native_before)
    os.chmod(native, 0o640)

    env_file = home / ".env"
    env_before = b"KEEP=old\n"
    env_file.write_bytes(env_before)
    os.chmod(env_file, 0o600)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.web_server as web_server
    import hermes_cli.config_store as config_store
    from hermes_cli.memory_providers import HINDSIGHT

    real_replace = config_store.atomic_replace
    calls = 0

    def fail_env_write(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected env publish failure")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", fail_env_write)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
    from hermes_cli.config_store import ConfigTransactionError

    with pytest.raises(ConfigTransactionError, match="injected env publish failure"):
        web_server._update_declared_provider_config(
            HINDSIGHT,
            {"mode": "cloud", "api_key": "test-secret"},
        )

    assert native.read_bytes() == native_before
    assert stat.S_IMODE(native.stat().st_mode) == 0o640
    assert env_file.read_bytes() == env_before
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert "HINDSIGHT_API_KEY" not in os.environ


def test_declared_memory_provider_publishes_env_with_secret_mode(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    env_file = home / ".env"
    env_file.write_bytes(b"KEEP=old\n")
    os.chmod(env_file, 0o644)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)

    import hermes_cli.web_server as web_server
    from hermes_cli.memory_providers import HINDSIGHT

    web_server._update_declared_provider_config(
        HINDSIGHT,
        {"mode": "cloud", "api_key": "test-secret"},
    )

    assert b"HINDSIGHT_API_KEY=test-secret" in env_file.read_bytes()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_declared_memory_provider_removes_new_parent_when_transaction_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    provider_dir = home / "hindsight"
    assert not provider_dir.exists()

    env_file = home / ".env"
    env_before = b"KEEP=old\n"
    env_file.write_bytes(env_before)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)

    import hermes_cli.config_store as config_store
    import hermes_cli.web_server as web_server
    from hermes_cli.memory_providers import HINDSIGHT

    real_replace = config_store.atomic_replace
    calls = 0

    def fail_second_publish(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected declared-provider publish failure")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", fail_second_publish)
    from hermes_cli.config_store import ConfigTransactionError

    with pytest.raises(
        ConfigTransactionError,
        match="declared-provider publish failure",
    ):
        web_server._update_declared_provider_config(
            HINDSIGHT,
            {"mode": "cloud", "api_key": "test-secret"},
        )

    assert env_file.read_bytes() == env_before
    assert not provider_dir.exists()
    assert "HINDSIGHT_API_KEY" not in os.environ


def test_undeclared_memory_provider_managed_preflight_precedes_all_side_effects(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED", "nix")

    native_dir = home / "generic-provider"

    class Provider:
        saved = False

        def get_config_schema(self):
            return [
                {
                    "key": "api_key",
                    "secret": True,
                    "env_var": "GENERIC_PROVIDER_API_KEY",
                }
            ]

        def save_config(self, _values, _home):
            self.saved = True
            native_dir.mkdir()
            (native_dir / "config.json").write_text("should-not-exist", encoding="utf-8")

    provider = Provider()
    import hermes_cli.web_server as web_server
    from hermes_cli.config import ManagedConfigWriteError

    with pytest.raises(ManagedConfigWriteError, match="managed"):
        web_server._write_memory_provider_config_values(
            "generic-provider",
            provider,
            {"api_key": "test-secret"},
        )

    assert provider.saved is False
    assert not native_dir.exists()
    assert not (home / ".env").exists()
    assert not (home / ".config-locks").exists()
    assert "GENERIC_PROVIDER_API_KEY" not in os.environ


def test_declared_memory_provider_managed_env_preflight_precedes_all_side_effects(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text(
        "HINDSIGHT_API_KEY=administrator-value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)

    import hermes_cli.managed_scope as managed_scope
    import hermes_cli.web_server as web_server
    from hermes_cli.config import ManagedConfigWriteError
    from hermes_cli.memory_providers import HINDSIGHT

    managed_scope.invalidate_managed_cache()
    with pytest.raises(ManagedConfigWriteError, match="administrator"):
        web_server._update_declared_provider_config(
            HINDSIGHT,
            {"mode": "cloud", "api_key": "user-value"},
        )

    assert not (home / "hindsight").exists()
    assert not (home / ".env").exists()
    assert not (home / ".config-locks").exists()
    assert "HINDSIGHT_API_KEY" not in os.environ


def test_declared_memory_provider_rejects_nul_before_all_side_effects(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)

    import hermes_cli.web_server as web_server
    from hermes_cli.memory_providers import HINDSIGHT

    with pytest.raises(ValueError, match="environment value"):
        web_server._update_declared_provider_config(
            HINDSIGHT,
            {"mode": "cloud", "api_key": "before\x00after"},
        )

    assert not (home / "hindsight").exists()
    assert not (home / ".env").exists()
    assert not (home / ".config-locks").exists()
    assert "HINDSIGHT_API_KEY" not in os.environ


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [1, 2, 3], ids=["native", "env", "yaml"])
async def test_memory_provider_activation_rolls_back_native_env_yaml_and_runtime(
    tmp_path: Path, monkeypatch, fail_at: int,
) -> None:
    """Every physical publish failure restores all three stores and runtime state."""

    home = tmp_path / "home"
    native_dir = home / "hindsight"
    native_dir.mkdir(parents=True)
    native = native_dir / "config.json"
    native_before = b'{"setting":"before","keep":"native"}\n'
    native.write_bytes(native_before)
    os.chmod(native, 0o640)

    env_file = home / ".env"
    env_before = b"HINDSIGHT_API_KEY=before-secret\nKEEP=env\n"
    env_file.write_bytes(env_before)
    os.chmod(env_file, 0o600)

    config_file = home / "config.yaml"
    config_before = b"memory:\n  provider: before-provider\nkeep: yaml\n"
    config_file.write_bytes(config_before)
    os.chmod(config_file, 0o644)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HINDSIGHT_API_KEY", "before-runtime")

    import hermes_cli.config_store as config_store
    import hermes_cli.web_server as web_server

    class Provider:
        def get_config_schema(self):
            return [
                {"key": "setting", "default": "before"},
                {
                    "key": "api_key",
                    "secret": True,
                    "env_var": "HINDSIGHT_API_KEY",
                },
            ]

        def save_config(self, values, hermes_home):
            target = Path(hermes_home) / "hindsight" / "config.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                '{"setting":"' + str(values["setting"]) + '","keep":"native"}\n',
                encoding="utf-8",
            )
            os.chmod(target, 0o600)

    provider = Provider()
    monkeypatch.setattr(web_server, "_load_memory_provider", lambda _name: provider)

    real_replace = config_store.atomic_replace
    calls = 0

    def fail_selected_publish(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError(f"injected publish failure {fail_at}")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", fail_selected_publish)

    with pytest.raises(web_server.HTTPException) as raised:
        await web_server.update_memory_provider_config(
            "hindsight",
            web_server.MemoryProviderConfigUpdate(
                values={"setting": "after", "api_key": "after-secret"}
            ),
        )

    assert raised.value.status_code == 500
    assert calls >= fail_at
    assert native.read_bytes() == native_before
    assert stat.S_IMODE(native.stat().st_mode) == 0o640
    assert env_file.read_bytes() == env_before
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert config_file.read_bytes() == config_before
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o644
    assert os.environ["HINDSIGHT_API_KEY"] == "before-runtime"
    assert yaml.safe_load(config_file.read_text(encoding="utf-8"))["memory"]["provider"] == "before-provider"


@pytest.mark.asyncio
async def test_memory_provider_activation_rollback_restores_absent_native_and_env(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config_file = home / "config.yaml"
    config_before = b"memory:\n  provider: before-provider\nkeep: yaml\n"
    config_file.write_bytes(config_before)
    os.chmod(config_file, 0o640)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)

    import hermes_cli.config_store as config_store
    import hermes_cli.web_server as web_server

    class Provider:
        def get_config_schema(self):
            return [
                {"key": "setting", "default": "before"},
                {
                    "key": "api_key",
                    "secret": True,
                    "env_var": "HINDSIGHT_API_KEY",
                },
            ]

        def save_config(self, values, hermes_home):
            target = Path(hermes_home) / "hindsight" / "config.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                '{"setting":"' + str(values["setting"]) + '"}\n',
                encoding="utf-8",
            )

    provider = Provider()
    monkeypatch.setattr(web_server, "_load_memory_provider", lambda _name: provider)

    real_replace = config_store.atomic_replace
    calls = 0

    def fail_yaml_publish(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected yaml publish failure")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", fail_yaml_publish)

    with pytest.raises(web_server.HTTPException) as raised:
        await web_server.update_memory_provider_config(
            "hindsight",
            web_server.MemoryProviderConfigUpdate(
                values={"setting": "after", "api_key": "after-secret"}
            ),
        )

    assert raised.value.status_code == 500
    assert calls == 3
    assert not (home / "hindsight").exists()
    assert not (home / ".env").exists()
    assert config_file.read_bytes() == config_before
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o640
    assert "HINDSIGHT_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_memory_provider_readiness_failure_rolls_back_all_committed_state(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    native_dir = home / "hindsight"
    native_dir.mkdir(parents=True)
    native = native_dir / "config.json"
    native_before = b'{"setting":"before"}\n'
    native.write_bytes(native_before)
    os.chmod(native, 0o640)

    env_file = home / ".env"
    env_before = b"HINDSIGHT_API_KEY=before-disk\n"
    env_file.write_bytes(env_before)
    os.chmod(env_file, 0o600)

    config_file = home / "config.yaml"
    config_before = b"memory:\n  provider: before-provider\nkeep: yaml\n"
    config_file.write_bytes(config_before)
    os.chmod(config_file, 0o640)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HINDSIGHT_API_KEY", "before-runtime")

    import hermes_cli.web_server as web_server

    class Provider:
        def get_config_schema(self):
            return [
                {"key": "setting", "default": "before"},
                {
                    "key": "api_key",
                    "secret": True,
                    "env_var": "HINDSIGHT_API_KEY",
                },
            ]

        def save_config(self, values, hermes_home):
            target = Path(hermes_home) / "hindsight" / "config.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                '{"setting":"' + str(values["setting"]) + '"}\n',
                encoding="utf-8",
            )

    provider = Provider()
    monkeypatch.setattr(web_server, "_load_memory_provider", lambda _name: provider)

    def reject_readiness(_name):
        raise web_server.HTTPException(status_code=400, detail="provider not ready")

    monkeypatch.setattr(web_server, "_require_memory_provider_ready", reject_readiness)

    with pytest.raises(web_server.HTTPException) as raised:
        await web_server.update_memory_provider_config(
            "hindsight",
            web_server.MemoryProviderConfigUpdate(
                values={"setting": "after", "api_key": "after-secret"}
            ),
        )

    assert raised.value.status_code == 400
    assert native.read_bytes() == native_before
    assert stat.S_IMODE(native.stat().st_mode) == 0o640
    assert env_file.read_bytes() == env_before
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert config_file.read_bytes() == config_before
    assert stat.S_IMODE(config_file.stat().st_mode) == 0o640
    assert os.environ["HINDSIGHT_API_KEY"] == "before-runtime"


@pytest.mark.asyncio
async def test_memory_provider_http_error_is_generic_and_internal_log_is_redacted(
    monkeypatch, caplog,
) -> None:
    import hermes_cli.web_server as web_server

    secret_error = "internal-path=/secret/memory token=memory-secret"
    monkeypatch.setattr(web_server, "_load_memory_provider", lambda _name: object())

    def fail_write(*_args, **_kwargs):
        raise OSError(secret_error)

    monkeypatch.setattr(web_server, "_write_memory_provider_config_values", fail_write)

    with pytest.raises(web_server.HTTPException) as raised:
        await web_server.update_memory_provider_config(
            "generic-provider",
            web_server.MemoryProviderConfigUpdate(values={"api_key": "submitted-secret"}),
        )

    assert raised.value.status_code == 500
    assert raised.value.detail == "Internal server error"
    rendered_logs = "\n".join(caplog.handler.format(record) for record in caplog.records)
    assert secret_error not in rendered_logs
    assert "memory-secret" not in rendered_logs
    assert "OSError" in rendered_logs
    assert "[REDACTED]" in rendered_logs
