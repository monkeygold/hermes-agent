from __future__ import annotations

from pathlib import Path

import yaml


def _write_config(path: Path, providers: list[dict[str, str]], marker: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {"custom_providers": providers, "concurrent_marker": marker},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_remove_custom_provider_rebases_by_stable_identity_after_concurrent_reorder(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_path = home / "config.yaml"
    provider_a = {"name": "A", "base_url": "https://a.example/v1"}
    provider_b = {"name": "B", "base_url": "https://b.example/v1"}
    concurrent = {"name": "Concurrent", "base_url": "https://new.example/v1"}
    _write_config(config_path, [provider_a, provider_b], "before")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    import hermes_cli.config as config_module
    import hermes_cli.curses_ui as curses_ui
    import hermes_cli.main as main

    config_module.invalidate_config_caches(config_path)

    def choose_b_after_concurrent_reorder(*_args, **_kwargs) -> int:
        _write_config(
            config_path,
            [provider_b, concurrent, provider_a],
            "after",
        )
        return 1  # B was index 1 in the snapshot shown to the operator.

    monkeypatch.setattr(curses_ui, "curses_radiolist", choose_b_after_concurrent_reorder)

    main._remove_custom_provider({})

    final = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert final["custom_providers"] == [concurrent, provider_a]
    assert final["concurrent_marker"] == "after"


def test_remove_custom_provider_preserves_concurrent_same_url_replacement(
    tmp_path: Path, monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_path = home / "config.yaml"
    provider_a = {"name": "A", "base_url": "https://a.example/v1"}
    provider_b = {
        "name": "B",
        "base_url": "https://shared.example/v1",
        "api_key": "${OLD_B_KEY}",
    }
    replacement_b = {
        "name": "B replacement",
        "base_url": "https://shared.example/v1",
        "api_key": "${NEW_B_KEY}",
    }
    _write_config(config_path, [provider_a, provider_b], "before")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    import hermes_cli.config as config_module
    import hermes_cli.curses_ui as curses_ui
    import hermes_cli.main as main

    config_module.invalidate_config_caches(config_path)

    def choose_b_after_concurrent_replacement(*_args, **_kwargs) -> int:
        _write_config(config_path, [replacement_b, provider_a], "after")
        return 1  # B was index 1 in the snapshot shown to the operator.

    monkeypatch.setattr(
        curses_ui,
        "curses_radiolist",
        choose_b_after_concurrent_replacement,
    )

    main._remove_custom_provider({})

    final = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert final["custom_providers"] == [replacement_b, provider_a]
    assert final["concurrent_marker"] == "after"


def test_remove_custom_provider_rejects_non_positive_fallback_choice(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    config_path = home / "config.yaml"
    providers = [
        {"name": "A", "base_url": "https://a.example/v1"},
        {"name": "B", "base_url": "https://b.example/v1"},
    ]
    _write_config(config_path, providers, "before")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    import hermes_cli.config as config_module
    import hermes_cli.curses_ui as curses_ui
    import hermes_cli.main as main

    config_module.invalidate_config_caches(config_path)
    monkeypatch.setattr(
        curses_ui,
        "curses_radiolist",
        lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError()),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "0")

    main._remove_custom_provider({})

    final = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert final["custom_providers"] == providers
    assert final["concurrent_marker"] == "before"
    assert "No change." in capsys.readouterr().out
