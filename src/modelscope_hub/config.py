"""Runtime configuration for the ModelScope Hub SDK.

Configuration values follow a clear precedence chain:

1. Explicit constructor argument
2. Process environment variable
3. Persisted file on disk (token only)
4. Sensible default

This separation keeps the SDK trivially testable: tests can supply a
:class:`HubConfig` instance with overrides and never touch real credentials.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    CONFIG_DIR_NAME,
    DEFAULT_CACHE_DIR_NAME,
    DEFAULT_ENDPOINT,
    TOKEN_FILE_NAME,
)
from .errors import CacheError, InvalidParameter

# Environment variable names — kept module-level for discoverability.
ENV_ENDPOINT = "MODELSCOPE_ENDPOINT"
ENV_CACHE = "MODELSCOPE_CACHE"
ENV_TOKEN = "MODELSCOPE_API_TOKEN"
ENV_HOME = "MODELSCOPE_HOME"


def _expand(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(slots=True)
class HubConfig:
    """Centralised runtime configuration object.

    The dataclass form makes it cheap to copy/override in tests via
    :func:`dataclasses.replace`, and keeps fields explicit and discoverable.
    """

    endpoint: str = field(default_factory=lambda: os.environ.get(ENV_ENDPOINT) or DEFAULT_ENDPOINT)
    cache_dir: Path = field(
        default_factory=lambda: _expand(
            os.environ.get(ENV_CACHE) or Path.home() / ".cache" / DEFAULT_CACHE_DIR_NAME
        )
    )
    config_dir: Path = field(
        default_factory=lambda: _expand(
            os.environ.get(ENV_HOME) or Path.home() / CONFIG_DIR_NAME
        )
    )
    token: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # MODELSCOPE_DOMAIN backward compat: if MODELSCOPE_ENDPOINT is not set
        # but the old MODELSCOPE_DOMAIN env var is, use it as the endpoint.
        if not os.environ.get(ENV_ENDPOINT):
            domain = os.environ.get("MODELSCOPE_DOMAIN", "").strip()
            if domain:
                if not domain.startswith("http://") and not domain.startswith("https://"):
                    domain = f"https://{domain}"
                self.endpoint = domain
        # Strip trailing slash so URL composition stays predictable.
        self.endpoint = self.endpoint.rstrip("/")
        if self.token is None:
            self.token = os.environ.get(ENV_TOKEN) or self.load_token()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @property
    def token_path(self) -> Path:
        return self.config_dir / TOKEN_FILE_NAME

    def ensure_dirs(self) -> None:
        """Create the config and cache directories if they do not exist."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise CacheError(f"Failed to create SDK directories: {exc}") from exc

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------
    def save_token(self, token: str) -> None:
        """Persist ``token`` to ``~/.modelscope/token`` with restrictive perms."""
        if not token or not token.strip():
            raise InvalidParameter("token must be a non-empty string")

        self.ensure_dirs()
        path = self.token_path
        path.write_text(token.strip(), encoding="utf-8")
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except OSError:  # pragma: no cover - best-effort on non-POSIX
            pass
        self.token = token.strip()

    def load_token(self) -> str | None:
        """Return the token persisted on disk, or ``None`` if absent."""
        path = self.token_path
        if not path.is_file():
            return None
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def clear_token(self) -> None:
        """Remove any persisted token from disk and from this config."""
        self.token = None
        path = self.token_path
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise CacheError(f"Failed to remove token file {path}: {exc}") from exc


# Singleton-style accessor — kept as a function so tests can monkeypatch it.
_default_config: HubConfig | None = None


def get_default_config() -> HubConfig:
    """Return the lazily-instantiated process-wide default configuration."""
    global _default_config
    if _default_config is None:
        _default_config = HubConfig()
    return _default_config


def set_default_config(config: HubConfig | None) -> None:
    """Override (or clear) the process-wide default configuration."""
    global _default_config
    _default_config = config


__all__ = [
    "ENV_CACHE",
    "ENV_ENDPOINT",
    "ENV_HOME",
    "ENV_TOKEN",
    "HubConfig",
    "get_default_config",
    "set_default_config",
]
