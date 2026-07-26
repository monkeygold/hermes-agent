"""R11 auth/config transaction rollback regressions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import pytest


def test_provider_update_restores_absent_auth_after_config_publish_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config_path = home / "config.yaml"
    config_before = b"model:\n  provider: existing\n  default: existing/model\n"
    config_path.write_bytes(config_before)
    auth_path = home / "auth.json"
    assert not auth_path.exists()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.auth as auth
    import hermes_cli.config_store as config_store

    real_replace = config_store.atomic_replace
    calls = 0

    def fail_config_publish(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected config publish failure")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", fail_config_publish)

    from hermes_cli.config_store import ConfigTransactionError

    with pytest.raises(ConfigTransactionError, match="injected config publish failure"):
        auth._update_config_for_provider(
            "openrouter",
            "https://openrouter.ai/api/v1",
            default_model="test/model",
        )

    assert not auth_path.exists()
    assert config_path.read_bytes() == config_before


def test_logout_restores_auth_and_config_when_config_publish_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_path = home / "config.yaml"
    config_before = b"model:\n  provider: nous\n  default: test/model\n"
    config_path.write_bytes(config_before)
    auth_path = home / "auth.json"
    auth_before = (
        json.dumps(
            {
                "version": 1,
                "active_provider": "nous",
                "providers": {"nous": {"access_token": "test-token"}},
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    auth_path.write_bytes(auth_before)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    import hermes_cli.auth as auth
    import hermes_cli.config_store as config_store

    real_replace = config_store.atomic_replace
    calls = 0

    def fail_config_publish(source, target, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected logout config publish failure")
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", fail_config_publish)

    from hermes_cli.config_store import ConfigTransactionError

    with pytest.raises(ConfigTransactionError, match="logout config publish failure"):
        auth.logout_command(SimpleNamespace(provider="nous"))

    assert auth_path.read_bytes() == auth_before
    assert config_path.read_bytes() == config_before


def test_logout_managed_preflight_precedes_lock_and_mutation(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_path = home / "config.yaml"
    config_before = b"model:\n  provider: nous\n  default: test/model\n"
    config_path.write_bytes(config_before)
    auth_path = home / "auth.json"
    auth_before = (
        json.dumps(
            {
                "version": 1,
                "active_provider": "nous",
                "providers": {"nous": {"access_token": "test-token"}},
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    auth_path.write_bytes(auth_before)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")

    import hermes_cli.auth as auth
    from hermes_cli.config import ManagedConfigWriteError

    with pytest.raises(ManagedConfigWriteError):
        auth.logout_command(SimpleNamespace(provider="nous"))

    assert auth_path.read_bytes() == auth_before
    assert config_path.read_bytes() == config_before
    assert not (home / ".config-locks").exists()
