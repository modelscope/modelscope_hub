"""Internal client for ``/api/v1/`` endpoints not covered by the OpenAPI spec.

This module is **private** — external callers should interact through the
public :class:`~modelscope_hub.api.HubApi` facade.

Auth strategy
-------------
The legacy API authenticates via a session cookie (``m_session_id``) rather
than the ``Authorization: Bearer`` header used by the OpenAPI surface. On
first use the client sets ``m_session_id = <access_token>`` on the
``requests.Session`` cookie jar — no round-trip to ``/api/v1/login`` is
needed for API calls.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import IO, Any, BinaryIO
from urllib.parse import quote_plus, urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from requests.structures import CaseInsensitiveDict
from urllib3.exceptions import InvalidHeader

from ._download_control import _DownloadControl, _DownloadDeadlineError, _network_context, _ReadHTTPAdapter
from .config import get_default_config
from .constants import (
    API_CONNECT_TIMEOUT,
    API_CONNECTION_POOL_MAXSIZE,
    API_MAX_RETRIES,
    API_TIMEOUT,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_TIMEOUT,
    LEGACY_API_PREFIX,
    REPO_FILES_TRUNCATION_LIMIT,
    REPO_TREE_MAX_REQUESTS,
    REPO_TREE_PAGE_MAX_ATTEMPTS,
    REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS,
    REPO_TREE_PAGE_TIMEOUT_SECONDS,
    REPO_TREE_WALK_WORKERS,
    UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS,
    UPLOAD_BLOB_READ_TIMEOUT_SECONDS,
    UPLOAD_HTTP_RETRY_ALLOWED_METHODS,
    RepoType,
)
from .errors import (
    APIError,
    InvalidParameter,
    NetworkError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    raise_for_status,
)
from .utils import append_user_agent, build_user_agent
from .utils.logger import get_logger

logger = get_logger("legacy_api")

# ---------------------------------------------------------------------------
# Repo-type routing table: maps RepoType → URL path segment (plural)
# ---------------------------------------------------------------------------
_REPO_TYPE_SEGMENT: dict[str, str] = {
    RepoType.MODEL: "models",
    RepoType.DATASET: "datasets",
    RepoType.STUDIO: "studios",
    RepoType.SKILL: "skills",
    RepoType.MCP: "mcps",
}

# For /api/v1/repos/ style endpoints, all use "{type}s" pattern
_REPOS_SEGMENT: dict[str, str] = {
    RepoType.MODEL: "models",
    RepoType.DATASET: "datasets",
    RepoType.STUDIO: "studios",
    RepoType.SKILL: "skills",
    RepoType.MCP: "mcps",
}


def _resolve_segment(repo_type: str) -> str:
    """Return the URL path segment for the given repo_type."""
    return _REPO_TYPE_SEGMENT.get(repo_type, f"{repo_type}s")


def _is_dataset(repo_type: str) -> bool:
    """Whether the repo_type addresses the dataset endpoints."""
    return repo_type in (RepoType.DATASET, "dataset", "datasets")


def _entry_path(entry: dict) -> str:
    """Return the repo-relative path of a file-tree entry."""
    return entry.get("Path") or entry.get("path") or entry.get("Name") or ""


def _is_dir_entry(entry: dict) -> bool:
    """Whether a file-tree entry is a directory (git tree object)."""
    return (entry.get("Type") or entry.get("type") or "blob") == "tree"


class LegacyClient:
    """Internal client for /api/v1/ endpoints not covered by OpenAPI.

    All network IO goes through :meth:`_request`, which handles auth headers,
    retries, and structured error raising.
    """

    def __init__(
        self,
        token: str | None,
        endpoint: str,
        timeout: int = API_TIMEOUT,
        max_retries: int = API_MAX_RETRIES,
        user_agent: str | None = None,
    ) -> None:
        self._token = token
        self._endpoint = endpoint.rstrip("/")
        self._timeout: int | tuple[int, int] = (API_CONNECT_TIMEOUT, timeout)
        self._session_authenticated = False
        # Set once any file-tree read succeeds. From then on a 403 on a tree
        # request cannot be an authorization result, so it is retried instead of
        # aborting an enumeration that spans hundreds of requests.
        self._tree_reads_ok = False

        self._session = requests.Session()
        self._user_agent = (
            user_agent
            if user_agent and user_agent.startswith("modelscope_hub/")
            else build_user_agent(config=get_default_config(), extra=user_agent)
        )
        self._session.headers["User-Agent"] = self._user_agent
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=UPLOAD_HTTP_RETRY_ALLOWED_METHODS,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=API_CONNECTION_POOL_MAXSIZE,
            pool_maxsize=API_CONNECTION_POOL_MAXSIZE,
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        # Callers manage download/listing retries; this pool leaves upload and regular API policies unchanged.
        self._read_session = requests.Session()
        self._read_session.headers.update(self._session.headers)
        read_adapter = _ReadHTTPAdapter(
            max_retries=0,
            pool_connections=API_CONNECTION_POOL_MAXSIZE,
            pool_maxsize=API_CONNECTION_POOL_MAXSIZE,
        )
        self._read_session.mount("https://", read_adapter)
        self._read_session.mount("http://", read_adapter)
        self._read_state = threading.local()

    @contextlib.contextmanager
    def read_context(self, control: _DownloadControl) -> Iterator[None]:
        previous = getattr(self._read_state, "control", None)
        self._read_state.control = control
        try:
            with _network_context(control):
                yield
        finally:
            self._read_state.control = previous

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def token(self) -> str | None:
        return self._token

    @token.setter
    def token(self, value: str | None) -> None:
        self._token = value
        self._session_authenticated = False
        self._session.cookies.clear()
        self._read_session.cookies.clear()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _build_url(self, path: str) -> str:
        """Construct full URL from a path relative to the legacy API prefix."""
        return f"{self._endpoint}{LEGACY_API_PREFIX}/{path.lstrip('/')}"

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Request-ID": uuid.uuid4().hex,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        merged = CaseInsensitiveDict(headers)
        if extra:
            merged.update(extra)
        merged["User-Agent"] = append_user_agent(self._user_agent, merged.get("User-Agent"))
        return dict(merged)

    def _ensure_session_auth(self) -> None:
        """Set the ``m_session_id`` cookie on the session for legacy API auth.

        Both cookie and Bearer header are sent: older endpoints rely on
        the ``m_session_id`` cookie, while newer endpoints accept the
        ``Authorization: Bearer`` header. Sending both is harmless and
        maximises compatibility.
        """
        if self._session_authenticated or not self._token:
            return
        domain = urlparse(self._endpoint).hostname or ""
        self._session.cookies.set("m_session_id", self._token, domain=domain, path="/")
        self._read_session.cookies.set("m_session_id", self._token, domain=domain, path="/")
        self._session_authenticated = True
        logger.debug("Legacy session cookie set for domain=%s", domain)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        data: Any | None = None,
        files: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | tuple[float, float] | None = None,
        stream: bool = False,
        single_attempt: bool = False,
    ) -> requests.Response:
        """Unified request dispatcher with auth, retry, and error handling."""
        self._ensure_session_auth()
        url = self._build_url(path)
        merged_headers = self._headers(headers)
        # Remove Content-Type for non-json payloads so requests can set
        # the correct boundary for multipart or the right encoding for data.
        if (data is not None or files is not None) and json_body is None:
            merged_headers.pop("Content-Type", None)

        logger.debug("%s %s", method, url)
        control = getattr(self._read_state, "control", None)
        if single_attempt and control is None:
            # Bound both headers and body reads for single-page calls without an external controller.
            with _DownloadControl(total_timeout=REPO_TREE_PAGE_TIMEOUT_SECONDS) as page_control:
                with self.read_context(page_control):
                    return self._request(
                        method,
                        path,
                        params=params,
                        json_body=json_body,
                        data=data,
                        files=files,
                        headers=headers,
                        timeout=timeout,
                        stream=stream,
                        single_attempt=True,
                    )
        session = self._read_session if control is not None or single_attempt else self._session
        request_timeout = timeout or self._timeout
        if control is not None:
            connect, read = (
                (API_CONNECT_TIMEOUT, request_timeout) if isinstance(request_timeout, int) else request_timeout
            )
            request_timeout = control.timeouts(connect, read)
        try:
            resp = session.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                data=data,
                files=files,
                headers=merged_headers,
                timeout=request_timeout,
                stream=stream or control is not None,
            )
        except requests.exceptions.RetryError as exc:
            raise ServerError(f"Max retries exceeded: {exc}") from exc
        except requests.ConnectionError as exc:
            raise NetworkError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise RequestTimeoutError(f"Request timed out: {exc}") from exc

        logger.debug("%s %s -> %s", method, url, resp.status_code)
        if resp.status_code >= 400 and control is None:
            logger.debug(
                "Request failed: %s %s params=%s status=%s body=%s",
                method,
                url,
                params,
                resp.status_code,
                resp.text[:500] if resp.text else "",
            )
        if control is not None:
            try:
                control.track(resp)
                if not stream or resp.status_code >= 400:
                    # Bound error-body reads too, avoiding unbounded reads inside raise_for_status.
                    body = bytearray()
                    for chunk in control.chunks(resp, DOWNLOAD_CHUNK_SIZE):
                        body.extend(chunk)
                        if resp.status_code >= 400 and len(body) >= 65536:
                            break
                    if resp.status_code >= 400 and len(body) >= 65536:
                        resp.raw.close()
                    resp._content = bytes(body)
                    resp._content_consumed = True
                    control.release(resp)
                    resp.close()
                try:
                    raise_for_status(resp)
                except APIError as exc:
                    exc.response = resp  # type: ignore[attr-defined]
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            exc.retry_after = Retry().parse_retry_after(retry_after)  # type: ignore[attr-defined]
                        except (ValueError, InvalidHeader):
                            pass
                    raise
                return resp
            except BaseException:
                control.release(resp)
                resp.close()
                raise
        raise_for_status(resp)
        return resp

    def _json_data(self, resp: requests.Response) -> Any:
        """Extract the 'Data' field from a standard legacy API JSON response."""
        body = resp.json()
        return body.get("Data", body)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def login(self, access_token: str) -> tuple[dict, requests.cookies.RequestsCookieJar]:
        """Authenticate via access token and return (user_data, cookies).

        POST /api/v1/login

        Performs a full login round-trip that returns a git token and
        server-issued session cookies. For routine API access, the
        ``m_session_id`` cookie set by :meth:`_ensure_session_auth` is
        sufficient — this method is only needed when the caller wants the
        git token or the full cookie set from the server.
        """
        self._token = access_token
        self._session_authenticated = False
        self._session.cookies.clear()
        self._ensure_session_auth()
        resp = self._request("POST", "login", json_body={"AccessToken": access_token})
        return self._json_data(resp), resp.cookies

    # ------------------------------------------------------------------
    # Repo CRUD (model / dataset)
    # ------------------------------------------------------------------
    def create_repo(self, repo_type: str, body: dict[str, Any]) -> dict:
        """POST /api/v1/{type}s — create a new repository.

        ``body`` is the fully-constructed request payload (PascalCase keys).

        The dataset endpoint requires multipart form data (matching the
        upstream server implementation), while model uses JSON.
        """
        segment = _resolve_segment(repo_type)
        if repo_type in (RepoType.DATASET, "dataset"):
            files = {k: (None, str(v)) for k, v in body.items() if v is not None}
            resp = self._request("POST", segment, files=files)
        else:
            resp = self._request("POST", segment, json_body=body)
        return self._json_data(resp)

    def create_aigc_model(self, body: dict[str, Any]) -> dict:
        """POST /api/v1/models/aigc — create an AIGC model repository.

        AIGC repositories use a dedicated legacy endpoint and payload shape;
        routing them through :meth:`create_repo` would incorrectly target the
        ordinary ``/models`` endpoint.
        """
        resp = self._request("POST", "models/aigc", json_body=body)
        return self._json_data(resp)

    def get_repo_info(self, repo_id: str, repo_type: str) -> dict:
        """GET /api/v1/{type}s/{repo_id} — fetch repository metadata.

        Uses cookie-based session auth which supports private repositories
        (unlike the OpenAPI Bearer-only endpoint).
        """
        segment = _resolve_segment(repo_type)
        resp = self._request("GET", f"{segment}/{repo_id}")
        return self._json_data(resp)

    def list_datasets(
        self,
        *,
        owner: str | None = None,
        page_number: int = 1,
        page_size: int = 50,
    ) -> dict:
        """GET /api/v1/datasets — list datasets with full private visibility.

        The legacy endpoint returns ALL datasets for the authenticated user
        (including private ones that the OpenAPI search index may miss).
        """
        params: dict[str, Any] = {
            "PageNumber": page_number,
            "PageSize": page_size,
        }
        if owner:
            params["owner"] = owner
        try:
            resp = self._request("GET", "datasets", params=params)
        except InvalidParameter as exc:
            if "fromIndex" in str(exc) and "toIndex" in str(exc):
                return {
                    "items": [],
                    "total_count": 0,
                    "page_number": page_number,
                    "page_size": page_size,
                }
            raise
        body = resp.json()
        return {
            "items": body.get("Data") or [],
            "total_count": body.get("TotalCount") or 0,
            "page_number": body.get("PageNumber") or page_number,
            "page_size": body.get("PageSize") or page_size,
        }

    def delete_repo(self, repo_id: str, repo_type: str) -> None:
        """Delete a repository.

        DELETE /api/v1/models/{owner}/{name}
        DELETE /api/v1/datasets/{dataset_id}
        """
        segment = _resolve_segment(repo_type)
        self._request("DELETE", f"{segment}/{repo_id}")

    # ------------------------------------------------------------------
    # File Tree
    # ------------------------------------------------------------------
    def list_repo_files(
        self,
        repo_id: str,
        repo_type: str,
        revision: str = "master",
        recursive: bool = True,
        root: str | None = None,
    ) -> list[dict]:
        """List files in a repository.

        Models/studios/etc: GET /api/v1/{type}s/{repo_id}/repo/files
        Datasets: GET /api/v1/datasets/{repo_id}/repo/tree

        ``repo/files`` supports no pagination and silently truncates at
        :data:`REPO_FILES_TRUNCATION_LIMIT` entries, so a recursive listing that
        comes back exactly at the limit is re-enumerated directory by directory
        (see :meth:`_walk_repo_files`). Datasets take the ``repo/tree``
        endpoint, which does paginate, so they are paged through instead.
        """
        if _is_dataset(repo_type):
            if recursive:
                return self.list_dataset_files_paginated(
                    repo_id=repo_id,
                    revision=revision,
                    root_path=root or "/",
                )
            return self._list_files_page(repo_id, repo_type, revision, recursive=False, root=root)

        self._read_state.listing_incomplete = False
        entries = self._list_files_page(repo_id, repo_type, revision, recursive=recursive, root=root)
        if len(entries) < REPO_FILES_TRUNCATION_LIMIT:
            return entries
        if not recursive:
            self._read_state.listing_incomplete = True
            logger.warning(
                "Directory %r of %s holds at least %d entries and the server returns no more "
                "than that for a single listing; the result is incomplete.",
                root or "/",
                repo_id,
                REPO_FILES_TRUNCATION_LIMIT,
            )
            return entries
        return self._walk_repo_files(repo_id, repo_type, revision, root=root, prefetched=entries)

    def _list_files_page(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        *,
        recursive: bool,
        root: str | None = None,
    ) -> list[dict]:
        """Perform a single file-tree request and unwrap the entry list."""
        segment = _resolve_segment(repo_type)
        params: dict[str, Any] = {
            "Revision": revision,
            "Recursive": str(recursive),
        }
        if root:
            params["Root"] = root

        suffix = "repo/tree" if _is_dataset(repo_type) else "repo/files"
        resp = self._request_repo_tree(
            f"{segment}/{repo_id}/{suffix}",
            params,
            repo_id=repo_id,
            authorized=self._tree_reads_ok,
        )
        data = self._json_data(resp)
        if isinstance(data, list):
            return data
        # Sometimes wrapped: {"Data": {"Files": [...]}}
        if isinstance(data, dict):
            return data.get("Files") or data.get("files") or []
        return []

    def _walk_repo_files(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        *,
        root: str | None = None,
        prefetched: list[dict] | None = None,
    ) -> list[dict]:
        """Enumerate a file tree the server truncated, one directory at a time.

        Since ``repo/files`` caps every response, the only way to see past the
        cap is to scope requests with ``Root``. Each subtree is first requested
        whole; only the subtrees that come back at the cap are listed shallowly
        and walked per child directory. The request count therefore scales with
        the number of oversized directories, not with the directory total.

        The walk proceeds one tree level at a time and issues the listings of a
        level concurrently, since a deep repo needs hundreds of round trips.

        ``prefetched`` is the already-truncated listing of ``root``, reused so
        the caller's request is not repeated. Entries are de-duplicated by path.
        Directories with more direct children than the cap cannot be enumerated
        at all — those are reported through a warning and the returned list is
        then knowingly incomplete.
        """
        limit = REPO_FILES_TRUNCATION_LIMIT
        collected: dict[str, dict] = {}
        oversized: list[str] = []
        requests_made = 0
        progress_mark = 200
        budget_spent = False

        logger.info(
            "Repo %s: file listing came back at the server cap of %d entries; walking the "
            "tree per directory to recover the full list ...",
            repo_id,
            limit,
        )

        def absorb(entries: list[dict]) -> None:
            for entry in entries:
                path = _entry_path(entry)
                if path:
                    collected.setdefault(path, entry)

        def fetch_level(roots: list[str | None], *, recursive: bool) -> dict[str | None, list[dict]]:
            """List several directories at once, clamped to the request budget."""
            nonlocal requests_made, budget_spent, progress_mark
            if not roots:
                return {}
            remaining = REPO_TREE_MAX_REQUESTS - requests_made
            if remaining <= 0:
                budget_spent = True
                return {}
            if len(roots) > remaining:
                roots = roots[:remaining]
                budget_spent = True
            requests_made += len(roots)

            control = getattr(self._read_state, "control", None)

            def one(subroot: str | None) -> list[dict]:
                scope = self.read_context(control) if control is not None else contextlib.nullcontext()
                with scope:
                    if control is not None:
                        control.check()
                    return self._list_files_page(
                        repo_id,
                        repo_type,
                        revision,
                        recursive=recursive,
                        root=subroot,
                    )

            if len(roots) == 1:
                listings = {roots[0]: one(roots[0])}
            else:
                workers = min(REPO_TREE_WALK_WORKERS, len(roots))
                pool = ThreadPoolExecutor(max_workers=workers)
                futures = {pool.submit(one, subroot): subroot for subroot in roots}
                try:
                    listings = {futures[future]: future.result() for future in as_completed(futures)}
                except BaseException:
                    if control is not None:
                        control.cancel_event.set()
                    for future in futures:
                        future.cancel()
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)

            if requests_made >= progress_mark:
                logger.info(
                    "Repo %s: %d listings done, %d entries collected so far ...",
                    repo_id,
                    requests_made,
                    len(collected),
                )
                progress_mark = requests_made + 200
            return listings

        frontier: list[str | None] = [root]
        known: dict[str | None, list[dict]] = {} if prefetched is None else {root: prefetched}

        while frontier:
            known.update(fetch_level([r for r in frontier if r not in known], recursive=True))

            truncated: list[str | None] = []
            for subroot in frontier:
                subtree = known.pop(subroot, None)
                if subtree is None:
                    continue  # budget ran out before this directory was reached
                # A truncated subtree is still a valid prefix, so keep its entries
                # and split the directory up to reach the rest.
                absorb(subtree)
                if len(subtree) >= limit:
                    truncated.append(subroot)
            if not truncated or budget_spent:
                break

            shallow_listings = fetch_level(truncated, recursive=False)
            next_frontier: list[str | None] = []
            for subroot in truncated:
                shallow = shallow_listings.get(subroot)
                if shallow is None:
                    continue
                absorb(shallow)
                child_dirs = [path for path in map(_entry_path, filter(_is_dir_entry, shallow)) if path]
                if len(shallow) >= limit or not child_dirs:
                    # This level alone exceeds the cap, or it has no sub-directories
                    # to split by: part of it stays invisible whatever we do. Still
                    # descend into the children we did see — that recovers strictly
                    # more than giving up here.
                    oversized.append(subroot or "/")
                next_frontier.extend(child_dirs)
            if budget_spent:
                break
            frontier = list(dict.fromkeys(next_frontier))

        self._read_state.listing_incomplete = budget_spent or bool(oversized)
        if budget_spent:
            logger.warning(
                "Repo %s: stopped walking the file tree after %d listings "
                "(MODELSCOPE_REPO_TREE_MAX_REQUESTS=%d); the file list is incomplete.",
                repo_id,
                requests_made,
                REPO_TREE_MAX_REQUESTS,
            )
        if oversized:
            shown = ", ".join(oversized[:5]) + (" ..." if len(oversized) > 5 else "")
            logger.warning(
                "Repo %s: %d director%s hold %d or more direct entries, which the server "
                "cannot enumerate in full; the file list is incomplete: %s",
                repo_id,
                len(oversized),
                "y" if len(oversized) == 1 else "ies",
                limit,
                shown,
            )
        logger.info(
            "Repo %s: collected %d entries from %d directory listings.",
            repo_id,
            len(collected),
            requests_made,
        )
        return list(collected.values())

    def list_dataset_files_paginated(
        self,
        repo_id: str,
        revision: str = "master",
        page_size: int = 200,
        root_path: str = "/",
    ) -> list[dict]:
        """List files in a dataset repo using pagination.

        Datasets can have millions of files, so this method pages through
        ``GET /api/v1/datasets/{repo_id}/repo/tree`` with
        ``PageNumber``/``PageSize`` params.

        A page occasionally answers ``403 access denied`` on a repository the
        caller demonstrably can read. Once any page has succeeded the credential
        is proven, so a later-page denial is a server-side hiccup rather than an
        authorization result, and it is retried instead of discarding every page
        collected so far -- a large listing spans hundreds of pages, which makes
        hitting it near-certain.
        """
        all_files: list[dict] = []
        page_number = 1
        logger.info("Listing %s@%s Root=%s ...", repo_id, revision, root_path)
        while True:
            params: dict[str, Any] = {
                "Revision": revision,
                "Recursive": "True",
                "PageNumber": page_number,
                "PageSize": page_size,
            }
            if root_path and root_path != "/":
                params["Root"] = root_path
            resp = self._request_repo_tree(
                f"datasets/{repo_id}/repo/tree",
                params,
                repo_id=repo_id,
                authorized=self._tree_reads_ok,
            )
            data = self._json_data(resp)
            if isinstance(data, list):
                files = data
            elif isinstance(data, dict):
                files = data.get("Files") or data.get("files") or []
            else:
                files = []

            all_files.extend(files)
            if page_number % 10 == 0 or len(files) < page_size:
                logger.info("Listing %s Root=%s: page %d, %d entries", repo_id, root_path, page_number, len(all_files))
            if len(files) < page_size:
                break
            page_number += 1
        return all_files

    def iter_dataset_file_pages(
        self,
        repo_id: str,
        revision: str = "master",
        root_path: str = "/",
        page_size: int = 200,
        isolated_pages: dict[int, str] | None = None,
    ) -> Iterator[tuple[int, list[dict], Exception | None]]:
        """Yield pages and bounded listing failures without buffering the dataset.

        One failed page does not suppress later pages; two consecutive failures
        stop this root to avoid probing an unknown infinite tail. Consumers must
        report every failed page and mark enumeration incomplete.
        """
        failures = 0
        logger.info("Streaming listing %s@%s Root=%s", repo_id, revision, root_path)
        for page in range(1, REPO_TREE_MAX_REQUESTS + 1):
            params = {"Revision": revision, "Recursive": "True", "PageNumber": page, "PageSize": page_size}
            if root_path != "/":
                params["Root"] = root_path
            try:
                if isolated_pages and page in isolated_pages:
                    raise NetworkError(f"Previously isolated page: {isolated_pages[page]}")
                response = self._request_repo_tree(
                    f"datasets/{repo_id}/repo/tree",
                    params,
                    repo_id=repo_id,
                    authorized=self._tree_reads_ok,
                    max_attempts=2,
                )
                payload = response.json()
                if isinstance(payload, dict) and (
                    payload.get("Success") is False
                    or payload.get("success") is False
                    or payload.get("Code", 200) not in (200, "200", None)
                ):
                    raise NetworkError("File listing returned a business failure with HTTP 200")
                data = payload.get("Data", payload) if isinstance(payload, dict) else payload
                if not isinstance(data, (list, dict)):
                    raise NetworkError("Malformed file listing response")
                files = data if isinstance(data, list) else (data.get("Files") or data.get("files") or [])
                if not isinstance(files, list):
                    raise NetworkError("Malformed file listing entries")
            except (NetworkError, ServerError, RateLimitError) as exc:
                yield page, [], exc
                failures += 1
                if failures >= 2:
                    return
                continue
            failures = 0
            yield page, files, None
            if page % 10 == 0 or len(files) < page_size:
                logger.info("Streaming listing %s Root=%s: page %d", repo_id, root_path, page)
            if len(files) < page_size:
                return
        yield REPO_TREE_MAX_REQUESTS + 1, [], NetworkError("File listing request budget exhausted")

    def _fetch_tree_page(self, path: str, params: dict[str, Any]) -> requests.Response:
        parent = getattr(self._read_state, "control", None)
        budget = float(REPO_TREE_PAGE_TIMEOUT_SECONDS)
        if parent is not None:
            parent.check()
            budget = min(budget, max(0.001, parent.deadline - time.monotonic()))
        try:
            with _DownloadControl(
                total_timeout=budget,
                cancel_event=parent.cancel_event if parent is not None else None,
            ) as page_control:
                with self.read_context(page_control):
                    return self._request(
                        "GET",
                        path,
                        params=params,
                        timeout=(API_CONNECT_TIMEOUT, REPO_TREE_PAGE_TIMEOUT_SECONDS),
                        single_attempt=True,
                    )
        except _DownloadDeadlineError as exc:
            if parent is not None:
                parent.check()
            raise RequestTimeoutError(f"列表单页请求超过 {REPO_TREE_PAGE_TIMEOUT_SECONDS}s") from exc

    def _request_repo_tree(
        self,
        path: str,
        params: dict[str, Any],
        *,
        repo_id: str,
        authorized: bool,
        max_attempts: int | None = None,
    ) -> Any:
        """Fetch one file-tree listing, retrying a spurious denial.

        The server intermittently answers a tree request with ``403 access
        denied`` on a repository the caller has just read successfully -- observed on
        both ``PageNumber``-paginated and ``Root``-scoped listings. Enumerating a
        large repository takes hundreds of such requests, so at that scale a
        single-request failure rate is effectively a guaranteed whole-listing
        failure, and it discards every entry gathered so far.

        ``authorized`` means some tree read already succeeded on this client, so
        the credential is proven and a denial cannot be an authorization result.
        Until then a 403 is taken at face value, keeping a real permission error
        fast and honest.
        """
        last_error: NetworkError | ServerError | RateLimitError | PermissionDeniedError | None = None
        attempts = REPO_TREE_PAGE_MAX_ATTEMPTS if max_attempts is None else max_attempts
        for attempt in range(attempts):
            try:
                resp = self._fetch_tree_page(path, params)
            except (NetworkError, ServerError, RateLimitError, PermissionDeniedError) as error:
                if isinstance(error, PermissionDeniedError) and not authorized:
                    raise
                if not error.retryable and not isinstance(error, PermissionDeniedError):
                    raise
                last_error = error
                if attempt < attempts - 1:
                    requested_wait = getattr(error, "retry_after", None)
                    wait = min(
                        max(0, requested_wait) if requested_wait is not None else 2**attempt,
                        REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS,
                    )
                    logger.warning(
                        "Repo %s: Root=%s page=%s listing failed (%s), retry in %ss (%s/%s)",
                        repo_id,
                        params.get("Root") or "/",
                        params.get("PageNumber", 1),
                        type(error).__name__,
                        wait,
                        attempt + 1,
                        attempts,
                    )
                    control = getattr(self._read_state, "control", None)
                    if control is None:
                        time.sleep(wait)
                    else:
                        control.sleep(wait)
                continue
            self._tree_reads_ok = True
            return resp
        if isinstance(last_error, PermissionDeniedError):
            raise NetworkError(
                f"Repo {repo_id}: tree listing ({params.get('Root') or params.get('PageNumber') or '/'}) kept "
                f"returning a denial after {attempts} attempts on an already-authorized "
                f"repo: {last_error}"
            ) from last_error
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # Revisions
    # ------------------------------------------------------------------
    def list_revisions(self, repo_id: str, repo_type: str) -> list[dict]:
        """List branches and tags of a repository.

        GET /api/v1/{type}s/{repo_id}/revisions
        """
        segment = _resolve_segment(repo_type)
        resp = self._request("GET", f"{segment}/{repo_id}/revisions")
        data = self._json_data(resp)
        if isinstance(data, dict):
            revision_map = data.get("RevisionMap") or {}
            tags = revision_map.get("Tags") or []
            branches = revision_map.get("Branches") or []
            return tags + branches
        return data if isinstance(data, list) else []

    def list_revisions_detail(
        self,
        repo_id: str,
        repo_type: str,
        *,
        end_time: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """List branches and tags as separate lists.

        GET /api/v1/{type}s/{repo_id}/revisions[?EndTime=...]

        Returns ``(branches, tags)`` where each element is a list of dicts
        with at least ``Revision`` and ``CreatedAt`` keys.
        """
        segment = _resolve_segment(repo_type)
        params: dict[str, Any] = {}
        if end_time is not None:
            params["EndTime"] = end_time
        resp = self._request(
            "GET",
            f"{segment}/{repo_id}/revisions",
            params=params or None,
        )
        data = self._json_data(resp)
        if isinstance(data, dict):
            revision_map = data.get("RevisionMap") or {}
            return (
                revision_map.get("Branches") or [],
                revision_map.get("Tags") or [],
            )
        return [], []

    def create_aigc_model_tag(self, body: dict[str, Any]) -> dict:
        """POST /api/v1/models/aigc/repo/tag — create an AIGC model version."""
        resp = self._request("POST", "models/aigc/repo/tag", json_body=body)
        return self._json_data(resp)

    def create_tag(
        self,
        repo_id: str,
        repo_type: str,
        tag: str,
        revision: str,
    ) -> dict:
        """Create a tag on a repository.

        POST /api/v1/{type}s/{repo_id}/repo/tag
        """
        segment = _resolve_segment(repo_type)
        body = {"TagName": tag, "Ref": revision}
        resp = self._request("POST", f"{segment}/{repo_id}/repo/tag", json_body=body)
        return self._json_data(resp)

    # ------------------------------------------------------------------
    # File deletion
    # ------------------------------------------------------------------
    def delete_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str = "master",
    ) -> dict:
        """Delete a single file from the repository.

        DELETE /api/v1/{type}s/{owner}/{name}/file?FilePath=...&Revision=...
        (for models)
        DELETE /api/v1/datasets/{owner}/{name}/repo?FilePath=...
        (for datasets)
        """
        segment = _resolve_segment(repo_type)
        if repo_type == RepoType.DATASET:
            resp = self._request(
                "DELETE",
                f"{segment}/{repo_id}/repo",
                params={"FilePath": file_path},
            )
        else:
            resp = self._request(
                "DELETE",
                f"{segment}/{repo_id}/file",
                params={"FilePath": file_path, "Revision": revision},
            )
        return self._json_data(resp)

    # ------------------------------------------------------------------
    # Git Commits
    # ------------------------------------------------------------------
    def create_commit(
        self,
        repo_id: str,
        repo_type: str,
        operations: list[dict],
        commit_message: str,
        revision: str = "master",
    ) -> dict:
        """Create a Git commit with file operations.

        POST /api/v1/repos/{type}s/{repo_id}/commit/{revision}

        Each operation should be a dict with keys:
        - action: "create" | "update" | "delete"
        - path: path in repo
        - type: "normal" | "lfs"
        - size: file size in bytes
        - sha256: hex digest (empty string for normal files)
        - content: base64 content (for normal files)
        - encoding: "base64" (for normal files)
        """
        segment = _resolve_segment(repo_type)
        payload = {
            "commit_message": commit_message,
            "actions": operations,
        }
        resp = self._request(
            "POST",
            f"repos/{segment}/{repo_id}/commit/{revision}",
            json_body=payload,
        )
        body = resp.json()
        if isinstance(body, dict) and (body.get("Success") is False or body.get("success") is False):
            message = body.get("Message") or body.get("message") or "Commit was rejected by the server"
            request_id = body.get("RequestId") or body.get("requestId") or body.get("request_id")
            raise APIError(
                str(message),
                status_code=resp.status_code,
                request_id=request_id,
                response_body=body,
                url=resp.url,
                method="POST",
            )
        return body.get("Data", body) if isinstance(body, dict) else body

    # ------------------------------------------------------------------
    # Blob Upload (LFS)
    # ------------------------------------------------------------------
    def validate_blobs(
        self,
        repo_id: str,
        repo_type: str,
        objects: list[dict[str, Any]],
    ) -> dict[str, str | None]:
        """Check which blobs need uploading via the LFS batch API.

        POST /api/v1/repos/{type}s/{repo_id}/info/lfs/objects/batch

        Returns mapping: sha256 → upload_url (if needs upload) or None (exists).
        """
        segment = _resolve_segment(repo_type)
        payload = {"operation": "upload", "objects": objects}
        resp = self._request(
            "POST",
            f"repos/{segment}/{repo_id}/info/lfs/objects/batch",
            json_body=payload,
        )
        data = self._json_data(resp)

        result: dict[str, str | None] = {}
        resp_objects = data.get("objects", []) if isinstance(data, dict) else []
        for obj in resp_objects:
            oid = obj.get("oid", "")
            actions = obj.get("actions", {})
            upload_action = actions.get("upload", {})
            href = upload_action.get("href")
            result[oid] = href  # None means already exists
        return result

    def upload_blob(
        self,
        upload_url: str,
        data: str | bytes | BinaryIO | IO[bytes] | Any,
        size: int,
        *,
        headers: dict[str, str] | None = None,
        timeout: int | tuple[int, int] | None = None,
    ) -> dict:
        """Upload a blob to the presigned URL returned by :meth:`validate_blobs`.

        PUT {upload_url}

        Sends both ``Authorization: Bearer`` and ``Cookie: m_session_id``
        headers to authenticate against the LFS domain (which may differ
        from the main API domain).

        Returns the parsed JSON response body on success.
        """
        upload_headers: dict[str, str] = {
            "Content-Length": str(size),
            "X-Request-ID": uuid.uuid4().hex,
        }
        if self._token:
            upload_headers["Authorization"] = f"Bearer {self._token}"
            upload_headers["Cookie"] = f"m_session_id={self._token}"
        merged_headers = CaseInsensitiveDict(upload_headers)
        if headers:
            merged_headers.update(headers)
        merged_headers["User-Agent"] = append_user_agent(self._user_agent, merged_headers.get("User-Agent"))

        try:
            resp = self._session.put(
                upload_url,
                data=data,
                headers=merged_headers,
                timeout=timeout or (UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS, UPLOAD_BLOB_READ_TIMEOUT_SECONDS),
            )
        except requests.ConnectionError as exc:
            raise NetworkError(f"Blob upload connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise RequestTimeoutError(f"Blob upload timed out: {exc}") from exc

        raise_for_status(resp)

        # Presigned URLs (cloud storage) may return empty bodies on success.
        try:
            body = resp.json()
        except (ValueError, RuntimeError):
            return {}
        if isinstance(body, dict) and body.get("Code") not in (200, "200", None):
            from .errors import APIError

            raise APIError(
                body.get("Message") or body.get("message") or f"Blob upload failed (Code={body.get('Code')})",
                status_code=resp.status_code,
                response_body=body,
                url=upload_url,
                method="PUT",
            )
        return body

    # ------------------------------------------------------------------
    # Raw Download URL
    # ------------------------------------------------------------------
    def get_download_url(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str = "master",
    ) -> str:
        """Construct the file download URL (no request is made).

        URL pattern: {endpoint}/api/v1/{type}s/{repo_id}/repo?Revision={rev}&FilePath={path}
        """
        segment = _resolve_segment(repo_type)
        return (
            f"{self._endpoint}{LEGACY_API_PREFIX}/{segment}/{repo_id}/repo"
            f"?Revision={quote_plus(revision)}&FilePath={quote_plus(file_path)}"
        )

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------
    def get_collection(
        self,
        collection_id: str,
        *,
        element_type: str = "skill",
        page_number: int = 1,
        page_size: int = 50,
    ) -> dict:
        """Get collection details and its elements.

        GET /api/v1/collections?Fid=...&ElementType=...
        """
        params = {
            "Fid": collection_id,
            "ElementType": element_type,
            "PageNumber": page_number,
            "PageSize": page_size,
        }
        resp = self._request("GET", "collections", params=params)
        return self._json_data(resp)

    # ------------------------------------------------------------------
    # Archive Download (skill repos)
    # ------------------------------------------------------------------
    def download_archive(
        self,
        repo_id: str,
        repo_type: str,
        revision: str = "master",
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Download the entire repo as a zip archive.

        GET /api/v1/{type}s/{repo_id}/archive/zip/{revision}

        Skills (and potentially other repo types) do not support per-file
        download via ``/repo?FilePath=...``.  This method streams the
        archive endpoint instead.
        """
        segment = _resolve_segment(repo_type)
        return self._request(
            "GET",
            f"{segment}/{repo_id}/archive/zip/{revision}",
            headers=headers,
            stream=True,
            timeout=(API_CONNECT_TIMEOUT, DOWNLOAD_TIMEOUT),
        )

    # ------------------------------------------------------------------
    # Raw Download URL
    # ------------------------------------------------------------------
    def download_stream(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str = "master",
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Start a streaming download of a file.

        Returns a Response with stream=True for chunked reading.
        """
        segment = _resolve_segment(repo_type)
        params = {"Revision": revision, "FilePath": file_path}
        control = getattr(self._read_state, "control", None)
        if control is None:
            # Low-level calls do not add adapter retries; higher-level callers handle retries and integrity checks.
            self._ensure_session_auth()
            resp = self._read_session.get(
                self.get_download_url(repo_id, repo_type, file_path, revision),
                headers=self._headers(headers),
                timeout=(API_CONNECT_TIMEOUT, DOWNLOAD_TIMEOUT),
                stream=True,
            )
            raise_for_status(resp)
            return resp
        return self._request(
            "GET",
            f"{segment}/{repo_id}/repo",
            params=params,
            headers=headers,
            stream=True,
            timeout=(API_CONNECT_TIMEOUT, DOWNLOAD_TIMEOUT),
        )
