from providers import get_provider_profile


def test_qoder_acp_profile_is_registered():
    profile = get_provider_profile("qoder-acp")

    assert profile is not None
    assert profile.auth_type == "external_process"
    assert profile.base_url == "acp://qoder"
    assert "qoder" in profile.aliases
