"""R11 auth persistence regressions built from fresh origin/main."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest


def _store(marker: str) -> dict[str, object]:
    return {
        "version": 1,
        "providers": {},
        "active_provider": marker,
    }


def test_auth_store_lock_freezes_symlink_target_until_publication(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    old_target = tmp_path / "old-auth.json"
    new_target = tmp_path / "new-auth.json"
    old_target.write_text(json.dumps(_store("old-before")) + "\n", encoding="utf-8")
    new_target.write_text(json.dumps(_store("new-before")) + "\n", encoding="utf-8")
    logical = home / "auth.json"
    logical.symlink_to(old_target)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.auth as auth

    from hermes_cli.config_store import UnsafeConfigPathError

    with auth._auth_store_lock():
        logical.unlink()
        logical.symlink_to(new_target)
        with pytest.raises(UnsafeConfigPathError, match="retargeted while locked"):
            auth._save_auth_store(_store("written-under-lock"))

    old_data = json.loads(old_target.read_text(encoding="utf-8"))
    new_data = json.loads(new_target.read_text(encoding="utf-8"))
    assert old_data["active_provider"] == "old-before"
    assert new_data["active_provider"] == "new-before"
    assert logical.resolve() == new_target.resolve()


def test_auth_store_lock_refreshes_capture_after_own_atomic_publication(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    auth_path = home / "auth.json"
    auth_path.write_text(json.dumps(_store("before")) + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.auth as auth

    with auth._auth_store_lock():
        auth._save_auth_store(_store("first"))
        auth._save_auth_store(_store("second"))

    final = json.loads(auth_path.read_text(encoding="utf-8"))
    assert final["active_provider"] == "second"


def test_auth_store_is_private_before_atomic_publication(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    auth_path = home / "auth.json"
    auth_path.write_text(json.dumps(_store("before")) + "\n", encoding="utf-8")
    os.chmod(auth_path, 0o644)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.auth as auth
    import hermes_cli.config_store as config_store

    real_replace = config_store.atomic_replace
    observed_modes: list[int] = []

    def inspect_replace(source, target, **kwargs):
        observed_modes.append(stat.S_IMODE(Path(source).stat().st_mode))
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", inspect_replace)

    auth._save_auth_store(_store("after"))

    assert observed_modes == [0o600]
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600


def test_provider_activation_restricts_existing_auth_store_to_0o600(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    auth_path = home / "auth.json"
    config_path = home / "config.yaml"
    auth_path.write_text(json.dumps(_store("before")) + "\n", encoding="utf-8")
    config_path.write_text("model:\n  default: old/model\n", encoding="utf-8")
    os.chmod(auth_path, 0o644)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.auth as auth
    import hermes_cli.config_store as config_store

    real_replace = config_store.atomic_replace
    observed_auth_temp_modes: list[int] = []

    def inspect_replace(source, target, **kwargs):
        if Path(target) == auth_path:
            observed_auth_temp_modes.append(stat.S_IMODE(Path(source).stat().st_mode))
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(config_store, "atomic_replace", inspect_replace)

    auth._update_config_for_provider(
        "nous",
        "https://inference.example.com/v1",
        default_model="nous/test-model",
    )

    assert observed_auth_temp_modes == [0o600]
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600


def test_oauth_token_saves_can_preserve_the_active_provider(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    auth_path = home / "auth.json"
    auth_path.write_text(
        json.dumps(_store("existing-provider")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_cli.auth as auth

    auth._save_codex_tokens(
        {"access_token": "codex-access", "refresh_token": "codex-refresh"},
        set_active=False,
    )
    after_codex = json.loads(auth_path.read_text(encoding="utf-8"))
    assert after_codex["active_provider"] == "existing-provider"
    assert "openai-codex" in after_codex["providers"]

    auth._save_xai_oauth_tokens(
        {"access_token": "xai-access", "refresh_token": "xai-refresh"},
        set_active=False,
    )
    after_xai = json.loads(auth_path.read_text(encoding="utf-8"))
    assert after_xai["active_provider"] == "existing-provider"
    assert "xai-oauth" in after_xai["providers"]


def test_provider_state_save_can_preserve_the_active_provider() -> None:
    import hermes_cli.auth as auth

    store = _store("existing-provider")
    auth._save_provider_state(
        store,
        "nous",
        {"access_token": "nous-access"},
        set_active=False,
    )

    assert store["active_provider"] == "existing-provider"
    assert isinstance(store["providers"], dict)
    assert "nous" in store["providers"]
