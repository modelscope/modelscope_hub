# Copyright (c) Alibaba, Inc. and its affiliates.
"""HTTP client for ModelScope Hub agent-repository API.

Endpoints:

* ``GET  /openapi/v1/users/me``                            -> login
* ``GET  /openapi/v1/agents/{path}/{name}``                -> repo metadata
* ``POST /openapi/v1/agents``                              -> create/update agent
* ``GET  /openapi/v1/agents/{path}/{name}/repo/files``     -> list files
* ``GET  /agents/{path}/{name}/resolve/{rev}/{file}``      -> file download
* ``POST /api/v1/agents/repo/files/upload``                -> two-step OSS upload (step1)
* ``POST /openapi/v1/agents/{path}/{name}/commit/{rev}``   -> commit files
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from ..config import HubConfig
from ..errors import APIError, HubError, NotExistError
from .._openapi import OpenAPIClient

logger = logging.getLogger("modelscope_hub.agent")


@dataclass
class RemoteFileInfo:
    """Metadata for a single file in the remote repository."""
    path: str
    sha256: str
    committed_date: int  # unix timestamp


class AgentApi:
    """HTTP client for ModelScope Hub agent-repository API.

    This is the primary programmatic interface for interacting with agent
    repositories on ModelScope Hub.

    Parameters
    ----------
    config : HubConfig or None
        Pre-built configuration. When provided, *endpoint* and *token* are
        ignored and the config is used directly.
    endpoint : str or None
        Hub API endpoint (fallback: HubConfig default).
    token : str or None
        API token (fallback: HubConfig default / ``ms login``).
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        timeout: int = 60,
        *,
        config: HubConfig | None = None,
    ):
        self._config = config or HubConfig(endpoint=endpoint, token=token)
        self.server = (self._config.endpoint or "").rstrip("/")
        self.token = self._config.token
        self.timeout = timeout
        self._openapi = OpenAPIClient(config=self._config, timeout=float(timeout))

    # ---- repository ----

    def repo_info(self, path: str, name: str) -> dict | None:
        """Repo metadata or None if the repo does not exist (404)."""
        try:
            return self._openapi.request("GET", f"/agents/{path}/{name}")
        except NotExistError:
            return None

    def check_repo(self, path: str, name: str) -> bool:
        """True if the repo exists, False on 404."""
        return self.repo_info(path, name) is not None

    def list_agents(self, owner: str | None = None, page_number: int = 1, page_size: int = 10) -> dict:
        """List agent repositories (GET /agents).

        Returns a dict with 'items' (list of agent metadata dicts) and
        'total_count' (int).
        """
        params = {"page_number": page_number, "page_size": page_size}
        if owner:
            params["owner"] = owner
        data = self._openapi.request(
            "GET", "/agents", params=params, require_token=False)
        if isinstance(data, list):
            return {"items": data, "total_count": len(data)}
        if isinstance(data, dict):
            items = data.get("Data") or []
            total = data.get("Total") or data.get("TotalCount") or len(items)
            return {"items": items, "total_count": total}
        return {"items": [], "total_count": 0}

    def create_repo(
        self, path: str, name: str, framework: str,
        visibility: str = "public",
        system_prompt_files: str | None = None,
    ) -> dict:
        """Create or update an agent (POST /agents).

        When *system_prompt_files* is provided the server uses the uploaded
        file as the agent content.
        """
        body: dict = {
            "path": path,
            "name": name,
            "framework": framework,
            "visibility": visibility,
        }
        if system_prompt_files:
            body["system_prompt_files"] = system_prompt_files
        return self._openapi.request("POST", "/agents", json_body=body)

    def list_repo_files(self, path: str, name: str, revision: str = 'master') -> list[str]:
        """All file paths in the repo, recursing into sub-directories."""
        entries = self._fetch_tree_entries(path, name, revision)
        return [e["path"] for e in entries if e["type"] == "blob" and e["path"]]

    def list_repo_files_detail(self, path: str, name: str, revision: str = 'master') -> list[RemoteFileInfo]:
        """All blob files with sha256 and committed_date."""
        entries = self._fetch_tree_entries(path, name, revision)
        results: list[RemoteFileInfo] = []
        for item in entries:
            if item["type"] != "blob" or not item["path"]:
                continue
            results.append(RemoteFileInfo(
                path=item["path"],
                sha256=item.get("sha256") or "",
                committed_date=int(item.get("committed_date") or 0),
            ))
        return results

    def _fetch_tree_entries(self, path: str, name: str, revision: str) -> list[dict]:
        """Fetch and normalize the repo file tree from the API (with pagination)."""
        page = 1
        page_size = 100
        max_pages = 50
        all_entries: list[dict] = []

        while True:
            data = self._openapi.request(
                "GET", f"/agents/{path}/{name}/repo/files",
                params={
                    "recursive": "true",
                    "page_size": str(page_size),
                    "page": str(page),
                    "revision": revision,
                },
            )

            raw = []
            if isinstance(data, dict):
                raw = data.get("Trees") or data.get("trees") or []
            elif isinstance(data, list):
                raw = data

            for item in raw:
                if not isinstance(item, dict):
                    continue
                all_entries.append({
                    "path": item.get("Path") or item.get("path") or "",
                    "type": item.get("Type") or item.get("type") or "",
                    "sha256": item.get("Sha256") or item.get("sha256") or "",
                    "committed_date": item.get("Committed_date") or item.get("committed_date") or 0,
                })

            if len(raw) < page_size:
                break
            page += 1
            if page > max_pages:
                logger.warning(
                    "Pagination limit reached (%d pages) for %s/%s; results may be incomplete.",
                    max_pages, path, name,
                )
                break

        return all_entries

    def download_repo_file(self, path: str, name: str, file_path: str,
                           revision: str = "master", *, binary: bool = False):
        """Download one repo file.

        Returns bytes when *binary=True*, otherwise str.
        Uses :meth:`OpenAPIClient.request_url` for retry and error handling.
        """
        url = f"{self.server}/agents/{path}/{name}/resolve/{revision}/{file_path}"
        resp = self._openapi.request_url("GET", url)
        return resp.content if binary else resp.text

    # ---- upload (two-step OSS) ----

    def _request_upload_urls(self, filenames: list[str]) -> dict:
        """Step 1: POST /api/v1/agents/repo/files/upload -> {Gid, Urls}."""
        url = f"{self.server}/api/v1/agents/repo/files/upload"
        resp = self._openapi.request_url(
            "POST", url,
            json_body={"FileNames": filenames},
            headers={"Content-Type": "application/json"},
        )
        body = resp.json()
        if not body.get("Success"):
            raise APIError(
                body.get("Message", "upload credential failed"),
                status_code=body.get("Code", 0),
            )
        return body["Data"]

    @staticmethod
    def _normalize_oss_url(url: str) -> str:
        """Decode %2F in the URL path so OSS signature verification passes."""
        parts = url.split("?", 1)
        path_part = parts[0]
        if "%2F" not in path_part and "%2f" not in path_part:
            return url
        decoded_path = path_part.replace("%2F", "/").replace("%2f", "/")
        if len(parts) == 2:
            return decoded_path + "?" + parts[1]
        return decoded_path

    def _upload_to_oss(self, signed_url: str, data: bytes) -> None:
        """Step 2: PUT raw bytes to signed OSS URL."""
        url = self._normalize_oss_url(signed_url)
        self._openapi.request_url(
            "PUT", url,
            data=data,
            headers={
                "Content-Type": "application/octet-stream",
                "x-oss-meta-author": "aliy",
            },
            require_token=False,
            timeout=max(self.timeout, 300),
        )

    def upload_file(self, resources: dict[str, bytes]) -> str:
        """Two-step upload: get signed URLs -> PUT to OSS -> return Gid.

        Returns empty string if *resources* is empty.
        """
        if not resources:
            logger.warning("upload_file called with empty resources; skipping.")
            return ""
        filenames = list(resources.keys())
        data = self._request_upload_urls(filenames)
        gid = data["Gid"]
        url_map = {item["Filename"]: item["Url"] for item in data["Urls"]}
        for fname, content in resources.items():
            signed_url = url_map.get(fname)
            if not signed_url:
                raise APIError(
                    f"Server did not return a signed URL for '{fname}'. "
                    f"Available: {list(url_map.keys())}",
                    status_code=0,
                )
            self._upload_to_oss(signed_url, content)
        return gid

    # ---- commit (incremental) ----

    def commit_files(self, path: str, name: str, actions: list[dict],
                     revision: str = "master", commit_message: str = "sync") -> dict:
        """Commit file changes via POST /openapi/v1/agents/{path}/{name}/commit/{revision}."""
        body = {"commit_message": commit_message, "actions": actions}
        return self._openapi.request(
            "POST", f"/agents/{path}/{name}/commit/{revision}", json_body=body)
