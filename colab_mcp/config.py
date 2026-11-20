"""Configuration and token management for Colab API, MCP, and CLI."""

import os
import shutil
import subprocess  # nosec B404
from typing import Optional, Tuple


class ColabConfig:
    """Configuration loader and token resolver for Google Colab API and MCP."""

    DEFAULT_MCP_ENDPOINT = "https://colaboratory.googleapis.com/mcp"
    DEFAULT_REST_ENDPOINT = "https://colaboratory.googleapis.com/v1beta"
    DEFAULT_HTTP_TIMEOUT = 120

    REQUIRED_OAUTH_SCOPES = (
        "openid",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/colaboratory",
    )

    @classmethod
    def load_env(cls, env_path: Optional[str] = None) -> Optional[str]:
        """Loads environment variables from .env or .env.colab if present.

        Returns the path of the loaded file, or None if no file was found.
        """
        candidates = [
            env_path,
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.getcwd(), ".env.colab"),
            os.path.expanduser("~/.env.colab"),
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(
                                k.strip(), v.strip().strip('"').strip("'")
                            )
                return path
        return None

    @classmethod
    def _get_gcloud_adc_token(cls, timeout: int = 5) -> Optional[str]:
        """Attempts to mint an access token via gcloud ADC if available."""
        gcloud_bin = shutil.which("gcloud")
        if not gcloud_bin:
            return None
        try:
            proc = subprocess.run(  # nosec B603
                [gcloud_bin, "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode == 0:
                token = proc.stdout.strip()
                if token:
                    return token
        except (OSError, subprocess.SubprocessError):
            return None
        return None

    @classmethod
    def get_token_with_source(
        cls, use_gcloud_fallback: bool = True
    ) -> Tuple[Optional[str], str]:
        """Resolves access token and returns (token, source_description)."""
        cls.load_env()
        for env_key in ("COLAB_API_TOKEN", "GOOGLE_OAUTH_TOKEN", "GCLOUD_AUTH_TOKEN"):
            val = os.environ.get(env_key)
            if val and "YOUR_GOOGLE" not in val:
                return val, f"env:{env_key}"

        if use_gcloud_fallback:
            adc_token = cls._get_gcloud_adc_token()
            if adc_token:
                return adc_token, "gcloud:adc"

        return None, "none"

    @classmethod
    def get_token(cls, use_gcloud_fallback: bool = True) -> Optional[str]:
        """Resolves access token from environment variables or gcloud ADC."""
        token, _ = cls.get_token_with_source(use_gcloud_fallback=use_gcloud_fallback)
        return token

    @classmethod
    def gcloud_login_command(cls) -> str:
        """Returns the recommended gcloud ADC login command with Colab scopes."""
        scopes = ",".join(cls.REQUIRED_OAUTH_SCOPES)
        return f"gcloud auth application-default login --scopes={scopes}"

    @classmethod
    def verify(cls, use_gcloud_fallback: bool = True) -> Tuple[bool, str]:
        """Checks if an authentication token is configured and usable."""
        token, source = cls.get_token_with_source(
            use_gcloud_fallback=use_gcloud_fallback
        )
        if not token or "YOUR_GOOGLE" in token:
            return (
                False,
                "COLAB_API_TOKEN is missing or contains placeholder. "
                f"Run `{cls.gcloud_login_command()}` or export COLAB_API_TOKEN.",
            )
        return True, f"Token configured (source: {source})."
