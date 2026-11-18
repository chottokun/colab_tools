"""Configuration and token management for Colab API, MCP, and CLI.
"""

import os
from typing import Optional, Tuple


class ColabConfig:
    """Configuration loader for Google Colab API and MCP."""

    DEFAULT_MCP_ENDPOINT = "https://colaboratory.googleapis.com/mcp"
    DEFAULT_REST_ENDPOINT = "https://colaboratory.googleapis.com/v1beta"

    @classmethod
    def load_env(cls, env_path: Optional[str] = None):
        """Loads environment variables from .env or .env.colab if present."""
        candidates = [
            env_path,
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.getcwd(), ".env.colab"),
            os.path.expanduser("~/.env.colab"),
            "/home/antigravity/.env.colab"
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                break

    @classmethod
    def get_token(cls) -> Optional[str]:
        """Resolves access token from environment variables."""
        cls.load_env()
        return (
            os.environ.get("COLAB_API_TOKEN")
            or os.environ.get("GOOGLE_OAUTH_TOKEN")
            or os.environ.get("GCLOUD_AUTH_TOKEN")
        )

    @classmethod
    def verify(cls) -> Tuple[bool, str]:
        """Checks if authentication token is configured."""
        token = cls.get_token()
        if not token or "YOUR_GOOGLE" in token:
            return False, "COLAB_API_TOKEN is missing or contains placeholder."
        return True, "Token configured."
