"""Client Manager for handling multiple OpenAI providers per model tier."""

from typing import Dict
from src.core.client import OpenAIClient
from src.core.config import config


class ClientManager:
    """Manages multiple OpenAI clients, one per model tier (big/middle/small)."""

    def __init__(self):
        """Initialize clients for each model tier."""
        self.clients: Dict[str, OpenAIClient] = {}

        # Create client for BIG model
        self.clients['big'] = OpenAIClient(
            api_key=config.big_model_api_key,
            base_url=config.big_model_base_url,
            timeout=config.request_timeout,
            api_version=config.big_model_azure_api_version,
            custom_headers=config.get_custom_headers('BIG'),
        )

        # Create client for MIDDLE model
        self.clients['middle'] = OpenAIClient(
            api_key=config.middle_model_api_key,
            base_url=config.middle_model_base_url,
            timeout=config.request_timeout,
            api_version=config.middle_model_azure_api_version,
            custom_headers=config.get_custom_headers('MIDDLE'),
        )

        # Create client for SMALL model
        self.clients['small'] = OpenAIClient(
            api_key=config.small_model_api_key,
            base_url=config.small_model_base_url,
            timeout=config.request_timeout,
            api_version=config.small_model_azure_api_version,
            custom_headers=config.get_custom_headers('SMALL'),
        )

    def get_client_for_model(self, model_name: str) -> OpenAIClient:
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
