"""Client Manager for handling multiple API providers per model tier."""

from typing import Dict, Union, TYPE_CHECKING
from src.core.client import OpenAIClient
from src.core.config import config

# Lazy import for Google GenAI to avoid requiring it when not used
if TYPE_CHECKING:
    from src.core.google_client import GoogleGenAIClient


class ClientManager:
    """Manages multiple API clients (OpenAI/Google), one per model tier (big/middle/small)."""

    def __init__(self):
        """Initialize clients for each model tier based on provider type."""
        self.clients: Dict[str, Union[OpenAIClient, "GoogleGenAIClient"]] = {}

        # Create client for BIG model
        self.clients['big'] = self._create_client(
            tier='BIG',
            provider=config.big_model_provider,
            api_key=config.big_model_api_key,
            base_url=config.big_model_base_url,
            azure_api_version=config.big_model_azure_api_version,
        )

        # Create client for MIDDLE model
        self.clients['middle'] = self._create_client(
            tier='MIDDLE',
            provider=config.middle_model_provider,
            api_key=config.middle_model_api_key,
            base_url=config.middle_model_base_url,
            azure_api_version=config.middle_model_azure_api_version,
        )

        # Create client for SMALL model
        self.clients['small'] = self._create_client(
            tier='SMALL',
            provider=config.small_model_provider,
            api_key=config.small_model_api_key,
            base_url=config.small_model_base_url,
            azure_api_version=config.small_model_azure_api_version,
        )

    def _create_client(
        self,
        tier: str,
        provider: str,
        api_key: str,
        base_url: str,
        azure_api_version: str = None,
    ) -> Union[OpenAIClient, "GoogleGenAIClient"]:
        """Create appropriate client based on provider type.

        Args:
            tier: Model tier name (BIG/MIDDLE/SMALL)
            provider: Provider type ('openai' or 'google')
            api_key: API key
            base_url: Base URL (only for OpenAI)
            azure_api_version: Azure API version (only for OpenAI)

        Returns:
            OpenAIClient or GoogleGenAIClient instance
        """
        provider = provider.lower()

        if provider == "google":
            # Lazy import Google client only when needed
            try:
                from src.core.google_client import GoogleGenAIClient
                print(f"   {tier} model using Google Generative AI")
                return GoogleGenAIClient(
                    api_key=api_key,
                    timeout=config.request_timeout,
                )
            except ImportError as e:
                print(f"❌ Error: Google Generative AI package not installed")
                print(f"   Install with: pip install google-generativeai")
                print(f"   Or run: uv pip install google-generativeai")
                raise ImportError(
                    f"google-generativeai package is required for provider='google'. "
                    f"Install with: pip install google-generativeai"
                ) from e
        else:  # Default to OpenAI
            return OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                timeout=config.request_timeout,
                api_version=azure_api_version,
                custom_headers=config.get_custom_headers(tier),
            )

    def get_client_for_model(self, model_name: str) -> Union[OpenAIClient, "GoogleGenAIClient"]:
        """Get the appropriate client for a given model name.

        Args:
            model_name: The OpenAI model name (already mapped from Claude model)

        Returns:
            The appropriate OpenAIClient instance
        """
        # Determine which tier this model belongs to
        if model_name == config.big_model:
            return self.clients['big']
        elif model_name == config.middle_model:
            return self.clients['middle']
        elif model_name == config.small_model:
            return self.clients['small']
        else:
            # Default to big model client for unknown models
            return self.clients['big']


# Global client manager instance
client_manager = ClientManager()
