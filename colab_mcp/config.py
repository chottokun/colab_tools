"""Configuration and token management for Colab API, MCP, and CLI."""

import json
import os
import shutil
import subprocess  # nosec B404
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional, Set, Tuple


class ColabConfig:
    """Configuration loader and token resolver for Google Colab API and MCP."""

    DEFAULT_MCP_ENDPOINT = "https://colaboratory.googleapis.com/mcp"
    DEFAULT_REST_ENDPOINT = "https://colaboratory.googleapis.com/v1beta"
    DEFAULT_HTTP_TIMEOUT = 120

    # セキュリティ保護・マスク・キャッシュ用の定数
    REDACTED_VALUE = "***REDACTED***"
    SENSITIVE_KEYS = frozenset(
        {"token", "access_token", "refresh_token", "client_secret", "authorization"}
    )
    LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
    TOKEN_EXPIRY_MARGIN_SECONDS = 120

    # ADC 自動リフレッシュトークン用インメモリキャッシュ
    _cached_token: Optional[str] = None
    _cached_token_expiry: float = 0.0
    _cached_auth_source: str = "none"
    # 401 で失効が確認された環境変数トークン値を除外するためのセット
    _invalidated_env_values: Set[str] = set()

    REQUIRED_OAUTH_SCOPES = (
        "openid",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/colaboratory",
    )

    @classmethod
    def clear_token_cache(cls, clear_invalidated_env: bool = True) -> None:
        """インメモリに保持している認証トークンキャッシュを初期化する。"""
        cls._cached_token = None
        cls._cached_token_expiry = 0.0
        cls._cached_auth_source = "none"
        if clear_invalidated_env:
            cls._invalidated_env_values.clear()

    @classmethod
    def validate_url(cls, url: str) -> None:
        """URL スキームを検証し、HTTPS または localhost 上の HTTP のみを許可する。"""
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "https" and host:
            return
        if parsed.scheme == "http" and host in cls.LOCALHOST_HOSTS:
            return
        raise ValueError(f"安全でないまたは不正な URL スキーム/ホストです: {url}")

    @classmethod
    def redact_secrets(cls, obj: Any) -> Any:
        """辞書・リスト内の機密キー（token 等）を再帰的にマスクした新しい構造を返す。"""
        if isinstance(obj, dict):
            redacted = {}
            for k, v in obj.items():
                if (
                    isinstance(k, str)
                    and k.lower() in cls.SENSITIVE_KEYS
                    and isinstance(v, str)
                    and v != ""
                ):
                    redacted[k] = cls.REDACTED_VALUE
                else:
                    redacted[k] = cls.redact_secrets(v)
            return redacted
        if isinstance(obj, list):
            return [cls.redact_secrets(item) for item in obj]
        return obj

    @classmethod
    def load_env(
        cls,
        env_path: Optional[str] = None,
        allow_cwd_dotenv: Optional[bool] = None,
    ) -> Optional[str]:
        """.env または .env.colab から環境変数を読み込む（見つからない場合は None）。"""
        if allow_cwd_dotenv is None:
            allow_cwd_dotenv = os.environ.get(
                "COLAB_IGNORE_CWD_DOTENV", ""
            ).strip().lower() not in ("1", "true", "yes")

        candidates = [env_path]
        if allow_cwd_dotenv:
            candidates.extend(
                [
                    os.path.join(os.getcwd(), ".env"),
                    os.path.join(os.getcwd(), ".env.colab"),
                ]
            )
        candidates.append(os.path.expanduser("~/.env.colab"))

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
    def _get_gcloud_adc_token(cls, timeout: int = 15) -> Optional[str]:
        """gcloud CLI の ADC からアクセストークンを取得する。"""
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
    def _get_adc_json_path(cls) -> Optional[str]:
        """Application Default Credentials (ADC) JSON ファイルのパスを返す。"""
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if gac and os.path.isfile(gac):
            return gac
        default_adc = os.path.expanduser(
            "~/.config/gcloud/application_default_credentials.json"
        )
        if os.path.isfile(default_adc):
            return default_adc
        return None

    @classmethod
    def get_quota_project(cls) -> Optional[str]:
        """クオータプロジェクト ID（x-goog-user-project 用）を解決して返す。"""
        cls.load_env()
        for env_key in (
            "COLAB_QUOTA_PROJECT",
            "GOOGLE_CLOUD_QUOTA_PROJECT",
            "GOOGLE_CLOUD_PROJECT",
            "GCLOUD_PROJECT",
        ):
            val = (os.environ.get(env_key) or "").strip()
            if val and "YOUR_" not in val:
                return val

        adc_path = cls._get_adc_json_path()
        if adc_path:
            try:
                with open(adc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    qp = (data.get("quota_project_id") or "").strip()
                    if qp:
                        return qp
            except (OSError, ValueError, TypeError):
                return None
        return None

    @classmethod
    def _refresh_adc_json_token(
        cls, timeout: int = 10
    ) -> Tuple[Optional[str], Optional[float]]:
        """ADC JSON のリフレッシュトークンからアクセストークンと有効期限を取得する。"""
        adc_path = cls._get_adc_json_path()
        if not adc_path:
            return None, None
        try:
            with open(adc_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            return None, None

        if not isinstance(data, dict) or data.get("type") != "authorized_user":
            return None, None

        client_id = data.get("client_id")
        client_secret = data.get("client_secret")
        refresh_token = data.get("refresh_token")
        if not (client_id and client_secret and refresh_token):
            return None, None

        token_uri = data.get("token_uri") or "https://oauth2.googleapis.com/token"
        if not isinstance(token_uri, str) or not token_uri.startswith("https://"):
            return None, None
        try:
            cls.validate_url(token_uri)
        except ValueError:
            return None, None

        form = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            token_uri,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
                body = json.loads(resp.read().decode("utf-8"))
                if isinstance(body, dict):
                    token = body.get("access_token")
                    if isinstance(token, str) and token:
                        try:
                            expires_in = int(body.get("expires_in", 3600))
                        except (ValueError, TypeError):
                            expires_in = 3600
                        return token, time.time() + max(expires_in, 0)
        except (OSError, urllib.error.URLError, ValueError, TypeError):
            return None, None
        return None, None

    @classmethod
    def _get_adc_json_token(cls, timeout: int = 10) -> Optional[str]:
        """ADC JSON のリフレッシュトークンを用いてアクセストークンを再発行する。"""
        token, _ = cls._refresh_adc_json_token(timeout=timeout)
        return token

    @classmethod
    def _get_env_token(cls) -> Tuple[Optional[str], str]:
        """環境変数から設定済みのトークンを取得する（失効済みと記録された値は除外）。"""
        for env_key in ("COLAB_API_TOKEN", "GOOGLE_OAUTH_TOKEN", "GCLOUD_AUTH_TOKEN"):
            val = os.environ.get(env_key)
            if (
                val
                and "YOUR_GOOGLE" not in val
                and val not in cls._invalidated_env_values
            ):
                return val, f"env:{env_key}"
        return None, "none"

    @classmethod
    def get_token_with_source(
        cls, use_gcloud_fallback: bool = True, force_refresh: bool = False
    ) -> Tuple[Optional[str], str]:
        """アクセストークンを解決し、(トークン, 取得元) のタプルを返す。"""
        cls.load_env()
        if force_refresh:
            # 失効した環境変数トークンが再選択されないよう無効化リストへ登録して ADC へ切り替える
            env_token, _ = cls._get_env_token()
            if env_token:
                cls._invalidated_env_values.add(env_token)
            cls.clear_token_cache(clear_invalidated_env=False)
        else:
            env_token, env_source = cls._get_env_token()
            if env_token:
                return env_token, env_source

        if use_gcloud_fallback:
            now = time.time()
            if (
                cls._cached_token
                and now < cls._cached_token_expiry - cls.TOKEN_EXPIRY_MARGIN_SECONDS
            ):
                return cls._cached_token, cls._cached_auth_source

            adc_token = cls._get_gcloud_adc_token()
            if adc_token:
                cls._cached_token = adc_token
                cls._cached_token_expiry = time.time() + 3000
                cls._cached_auth_source = "gcloud:adc"
                return adc_token, "gcloud:adc"

            # gcloud CLI がないヘッドレス環境向けに ADC JSON から直接リフレッシュする
            adc_json_token, expiry = cls._refresh_adc_json_token()
            if adc_json_token:
                cls._cached_token = adc_json_token
                cls._cached_token_expiry = expiry or (time.time() + 3000)
                cls._cached_auth_source = "adc:json"
                return adc_json_token, "adc:json"

        return None, "none"

    @classmethod
    def get_token(
        cls, use_gcloud_fallback: bool = True, force_refresh: bool = False
    ) -> Optional[str]:
        """環境変数または ADC からアクセストークンを解決して返す。"""
        token, _ = cls.get_token_with_source(
            use_gcloud_fallback=use_gcloud_fallback, force_refresh=force_refresh
        )
        return token

    @classmethod
    def gcloud_login_command(cls) -> str:
        """Returns the recommended gcloud ADC login command with Colab scopes."""
        scopes = ",".join(cls.REQUIRED_OAUTH_SCOPES)
        return f"gcloud auth application-default login --scopes={scopes}"

    @classmethod
    def verify(cls, use_gcloud_fallback: bool = True) -> Tuple[bool, str]:
        """認証トークンおよびクオータプロジェクトの設定状態を確認する。"""
        token, source = cls.get_token_with_source(
            use_gcloud_fallback=use_gcloud_fallback
        )
        if not token or "YOUR_GOOGLE" in token:
            return (
                False,
                "COLAB_API_TOKEN is missing or contains placeholder. "
                f"Run `{cls.gcloud_login_command()}` or export COLAB_API_TOKEN.",
            )
        quota_project = cls.get_quota_project()
        if quota_project:
            return True, f"Token configured (source: {source}, quota_project: {quota_project})."
        return True, f"Token configured (source: {source}, quota_project: not set)."
