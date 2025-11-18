import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Find .env file in project root (parent of parent of this file)
    config_dir = Path(__file__).parent  # src/core
    src_dir = config_dir.parent  # src
    project_root = src_dir.parent  # project root
    env_file = project_root / '.env'

    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from .env file")
    else:
        print(f"⚠️  No .env file found at {env_file}")
        print(f"   Please copy .env.example to .env and configure it")
except ImportError:
    print("⚠️  python-dotenv not installed")

# Configuration
class Config:
    def __init__(self):
        # Global/fallback provider settings
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Add Anthropic API key for client validation
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            print("Warning: ANTHROPIC_API_KEY not set. Client API key validation will be disabled.")

        self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.azure_api_version = os.environ.get("AZURE_API_VERSION")  # For Azure OpenAI
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "4096"))
        self.min_tokens_limit = int(os.environ.get("MIN_TOKENS_LIMIT", "100"))

        # Connection settings
        self.request_timeout = int(os.environ.get("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Model settings - BIG, MIDDLE, and SMALL models
        self.big_model = os.environ.get("BIG_MODEL", "gpt-4o")
        self.middle_model = os.environ.get("MIDDLE_MODEL", self.big_model)
        self.small_model = os.environ.get("SMALL_MODEL", "gpt-4o-mini")

        # Per-model provider settings (with fallback to global settings)
        # BIG model provider settings
        self.big_model_api_key = os.environ.get("BIG_MODEL_API_KEY", self.openai_api_key)
        self.big_model_base_url = os.environ.get("BIG_MODEL_BASE_URL", self.openai_base_url)
        self.big_model_azure_api_version = os.environ.get("BIG_MODEL_AZURE_API_VERSION", self.azure_api_version)

        # MIDDLE model provider settings
        self.middle_model_api_key = os.environ.get("MIDDLE_MODEL_API_KEY", self.openai_api_key)
        self.middle_model_base_url = os.environ.get("MIDDLE_MODEL_BASE_URL", self.openai_base_url)
        self.middle_model_azure_api_version = os.environ.get("MIDDLE_MODEL_AZURE_API_VERSION", self.azure_api_version)

        # SMALL model provider settings
        self.small_model_api_key = os.environ.get("SMALL_MODEL_API_KEY", self.openai_api_key)
        self.small_model_base_url = os.environ.get("SMALL_MODEL_BASE_URL", self.openai_base_url)
        self.small_model_azure_api_version = os.environ.get("SMALL_MODEL_AZURE_API_VERSION", self.azure_api_version)
        
    def validate_api_key(self):
        """Basic API key validation"""
        if not self.openai_api_key:
            return False
        # Basic format check for OpenAI API keys
        if not self.openai_api_key.startswith('sk-'):
            return False
        return True
        
    def validate_client_api_key(self, client_api_key):
        """Validate client's Anthropic API key"""
        # If no ANTHROPIC_API_KEY is set in environment, skip validation
        if not self.anthropic_api_key:
            return True
            
        # Check if the client's API key matches the expected value
        return client_api_key == self.anthropic_api_key
    
    def get_custom_headers(self, model_tier=None):
        """Get custom headers from environment variables

        Args:
            model_tier: Optional model tier ('BIG', 'MIDDLE', 'SMALL') to get tier-specific headers

        Returns:
            Dictionary of custom headers
        """
        custom_headers = {}

        # Get all environment variables
        env_vars = dict(os.environ)

        # Find CUSTOM_HEADER_* environment variables (global headers)
        for env_key, env_value in env_vars.items():
            if env_key.startswith('CUSTOM_HEADER_'):
                # Skip tier-specific headers for now
                if any(tier in env_key for tier in ['_BIG_MODEL_', '_MIDDLE_MODEL_', '_SMALL_MODEL_']):
                    continue

                # Convert CUSTOM_HEADER_KEY to Header-Key
                # Remove 'CUSTOM_HEADER_' prefix and convert to header format
                header_name = env_key[14:]  # Remove 'CUSTOM_HEADER_' prefix

                if header_name:  # Make sure it's not empty
                    # Convert underscores to hyphens for HTTP header format
                    header_name = header_name.replace('_', '-')
                    custom_headers[header_name] = env_value

        # If model_tier is specified, add/override with tier-specific headers
        if model_tier:
            prefix = f'CUSTOM_HEADER_{model_tier}_MODEL_'
            for env_key, env_value in env_vars.items():
                if env_key.startswith(prefix):
                    # Remove prefix and convert to header format
                    header_name = env_key[len(prefix):]

                    if header_name:
                        header_name = header_name.replace('_', '-')
                        custom_headers[header_name] = env_value

        return custom_headers

try:
    config = Config()
    print(f" Configuration loaded: API_KEY={'*' * 20}..., BASE_URL='{config.openai_base_url}'")
except Exception as e:
    print(f"=4 Configuration Error: {e}")
    sys.exit(1)
