"""Legacy-compatible HubApi wrapper.

Provides the same interface as ``modelscope.hub.api.HubApi`` (old SDK) by
wrapping the new ``modelscope_hub.HubApi``.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..api import HubApi
from ..constants import RepoType


class LegacyHubApi:
    """Drop-in replacement for the old ``modelscope.hub.api.HubApi``.

    Accepts the old constructor signature and maps method calls to the
    new HubApi implementation.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
    ) -> None:
        if endpoint and not endpoint.startswith("http"):
            endpoint = f"https://{endpoint}"
        self._endpoint = endpoint
        self._api = HubApi(endpoint=endpoint, token=token)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def login(self, token: str) -> None:
        """Login with token (old style returns None)."""
        self._api.login(token)

    def get_cookies(self, access_token: str) -> dict:
        """Legacy method — returns empty dict; token-based auth is used."""
        warnings.warn(
            "get_cookies() is deprecated. Token-based auth is used directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {}

    # ------------------------------------------------------------------
    # Repository operations
    # ------------------------------------------------------------------
    def get_model(self, model_id: str) -> dict:
        """Get model info as a raw dict."""
        info = self._api.get_repo(model_id, RepoType.MODEL)
        return _repo_info_to_dict(info)

    def get_model_files(self, model_id: str, recursive: bool = True) -> list[dict]:
        """List files in a model repo."""
        files = self._api.list_repo_files(model_id, RepoType.MODEL, recursive=recursive)
        return [{"Path": f.path, "Size": f.size} for f in files]

    def create_repo(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        visibility: int | str | None = None,
        repo_type: str = "model",
        chinese_name: str | None = None,
        license: str | None = None,
        exist_ok: bool = False,
        create_default_config: bool = False,
        endpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a repository (legacy signature)."""
        api = self._api
        if token:
            api = HubApi(token=token, endpoint=endpoint or self._endpoint)
        try:
            api.create_repo(
                repo_id,
                repo_type=repo_type,
                visibility=visibility,
                license=license,
                chinese_name=chinese_name,
                **kwargs,
            )
        except Exception as exc:
            if exist_ok and "exist" in str(exc).lower():
                return
            raise

    def push_model(self, model_id: str, model_dir: str, **kwargs: Any) -> None:
        """Upload a model directory (legacy signature)."""
        self._api.upload_folder(
            model_id,
            RepoType.MODEL,
            model_dir,
            path_in_repo="",
            commit_message=kwargs.get("commit_message"),
            revision=kwargs.get("revision"),
            max_workers=kwargs.get("max_workers", 4),
        )

    # ------------------------------------------------------------------
    # Download operations
    # ------------------------------------------------------------------
    def download_model(
        self,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_dir: str | None = None,
    ) -> str:
        """Download a model snapshot."""
        result = self._api.download_repo(
            model_id,
            repo_type=RepoType.MODEL,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        return str(result)

    # ------------------------------------------------------------------
    # Studio operations
    # ------------------------------------------------------------------
    def deploy_studio(self, studio_id: str, **kwargs: Any) -> dict:
        self._api.deploy_repo(studio_id, RepoType.STUDIO)
        return {"status": "deploying"}

    def stop_studio(self, studio_id: str, **kwargs: Any) -> dict:
        self._api.stop_repo(studio_id, RepoType.STUDIO)
        return {"status": "stopping"}

    def get_studio_logs(self, studio_id: str, **kwargs: Any) -> dict:
        return self._api.get_repo_logs(studio_id, RepoType.STUDIO, **kwargs)

    def update_studio_settings(self, studio_id: str, **kwargs: Any) -> dict:
        return self._api.update_repo_settings(studio_id, RepoType.STUDIO, **kwargs)

    def list_studio_secrets(self, studio_id: str, **kwargs: Any) -> list:
        return self._api.list_secrets(studio_id, RepoType.STUDIO)

    def add_studio_secret(self, studio_id: str, key: str, value: str, **kwargs: Any) -> None:
        self._api.add_secret(studio_id, key, value, RepoType.STUDIO)

    def update_studio_secret(self, studio_id: str, key: str, value: str, **kwargs: Any) -> None:
        self._api.update_secret(studio_id, key, value, RepoType.STUDIO)

    def delete_studio_secret(self, studio_id: str, key: str, **kwargs: Any) -> None:
        self._api.delete_secret(studio_id, key, RepoType.STUDIO)

    # ------------------------------------------------------------------
    # Collection / Skills
    # ------------------------------------------------------------------
    def get_collection(self, collection_id: str, **kwargs: Any) -> dict:
        """Fetch collection data — delegates to OpenAPI."""
        return self._api.openapi.get_collection(collection_id)

    def download_skill(self, skill_id: str, local_dir: str | None = None, **kwargs: Any) -> str:
        """Download a skill to local directory."""
        result = self._api.download_repo(
            skill_id,
            repo_type=RepoType.SKILL,
            local_dir=local_dir,
        )
        return str(result)


def _repo_info_to_dict(info: Any) -> dict:
    """Convert a RepoInfo to a plain dict."""
    if hasattr(info, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(info)
    if hasattr(info, "__dict__"):
        return {k: v for k, v in info.__dict__.items() if not k.startswith("_")}
    return {}
