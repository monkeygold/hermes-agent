"""Qoder CLI provider profile for its local ACP subprocess."""

from providers import register_provider
from providers.base import ProviderProfile


class QoderACPProfile(ProviderProfile):
    """Qoder CLI ACP — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Qoder manages its selected model inside the ACP subprocess."""
        return None


qoder_acp = QoderACPProfile(
    name="qoder-acp",
    aliases=("qoder", "qoder-cli"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://qoder",
    auth_type="external_process",
)

register_provider(qoder_acp)
