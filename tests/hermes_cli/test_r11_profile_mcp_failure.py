"""R11 profile builder persistence boundary regressions."""


def test_profile_create_fails_closed_when_requested_mcp_persistence_fails(
    monkeypatch, _isolate_hermes_home, caplog,
) -> None:
    from starlette.testclient import TestClient

    import hermes_cli.profiles as profiles
    import hermes_cli.web_server as web_server

    wrapper_called = False

    def record_wrapper(_name):
        nonlocal wrapper_called
        wrapper_called = True
        return None

    monkeypatch.setattr(profiles, "create_wrapper_script", record_wrapper)
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda *_args, **_kwargs: {})

    secret_error = "internal-path=/secret/config.yaml token=super-secret"

    def fail_mcp_write(*_args, **_kwargs):
        raise OSError(secret_error)

    monkeypatch.setattr(web_server, "_write_profile_mcp_servers", fail_mcp_write)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.post(
        "/api/profiles",
        json={
            "name": "mcp-failure",
            "no_skills": True,
            "mcp_servers": [
                {"name": "ctx", "url": "https://example.invalid/mcp"},
            ],
        },
    )

    assert response.status_code >= 500
    assert secret_error not in response.text
    assert "super-secret" not in response.text
    rendered_logs = "\n".join(caplog.handler.format(record) for record in caplog.records)
    assert secret_error not in rendered_logs
    assert "super-secret" not in rendered_logs
    assert "OSError" in rendered_logs
    assert "[REDACTED]" in rendered_logs
    assert not profiles.get_profile_dir("mcp-failure").exists()
    assert wrapper_called is False


def test_profile_create_unexpected_failure_does_not_expose_internal_exception(
    monkeypatch, _isolate_hermes_home, caplog,
) -> None:
    from starlette.testclient import TestClient

    import hermes_cli.profiles as profiles
    import hermes_cli.web_server as web_server

    secret_error = "internal-path=/secret/profiles token=profile-secret"

    def fail_profile_create(*_args, **_kwargs):
        raise OSError(secret_error)

    monkeypatch.setattr(profiles, "create_profile", fail_profile_create)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.post("/api/profiles", json={"name": "profile-failure"})

    assert response.status_code == 500
    assert secret_error not in response.text
    assert "profile-secret" not in response.text
    rendered_logs = "\n".join(caplog.handler.format(record) for record in caplog.records)
    assert secret_error not in rendered_logs
    assert "profile-secret" not in rendered_logs
    assert "OSError" in rendered_logs
    assert "[REDACTED]" in rendered_logs


def test_profile_create_rolls_back_when_alias_check_raises(
    monkeypatch, _isolate_hermes_home,
) -> None:
    from starlette.testclient import TestClient

    import hermes_cli.profiles as profiles
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(profiles, "seed_profile_skills", lambda *_args, **_kwargs: {})

    def fail_alias_check(_name):
        raise OSError("alias-check-failed")

    monkeypatch.setattr(profiles, "check_alias_collision", fail_alias_check)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.post(
        "/api/profiles",
        json={"name": "alias-check-failure", "no_skills": True},
    )

    assert response.status_code == 500
    assert not profiles.get_profile_dir("alias-check-failure").exists()


def test_profile_create_rolls_back_when_wrapper_creation_returns_none(
    monkeypatch, _isolate_hermes_home,
) -> None:
    from starlette.testclient import TestClient

    import hermes_cli.profiles as profiles
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(profiles, "seed_profile_skills", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(profiles, "check_alias_collision", lambda _name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda _name: None)

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.post(
        "/api/profiles",
        json={"name": "wrapper-failure", "no_skills": True},
    )

    assert response.status_code == 500
    assert not profiles.get_profile_dir("wrapper-failure").exists()
