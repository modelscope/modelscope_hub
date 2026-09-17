"""File-tree listing resilience against spurious denials.

The server intermittently answers a tree request with ``403 无权访问该数据集`` on a
repository the caller has just read successfully. Observed live on a 25k-file
dataset (paginated listing: page 7 denied, then page 145 denied, then success)
and again on a 10k-file dataset (``Root``-scoped listing denied outright).
Enumerating a large repo takes hundreds of such requests, so a per-request
failure rate becomes a near-certain whole-listing failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import modelscope_hub._legacy_api as legacy_module
from modelscope_hub._legacy_api import LegacyClient
from modelscope_hub.errors import NetworkError, PermissionDeniedError


def _client() -> LegacyClient:
    return LegacyClient(endpoint="https://modelscope.cn", token="ms-test")


def _page(count: int, offset: int = 0) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {"Data": {"Files": [{"Path": f"f{offset + i}"} for i in range(count)]}}
    return response


def _denied() -> PermissionDeniedError:
    return PermissionDeniedError("无权访问该数据集", status_code=403)


def test_transient_denial_on_a_later_page_is_retried(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(legacy_module.time, "sleep", lambda _s: None)
    calls: list[int] = []

    def request(method: str, path: str, **kwargs):
        page = kwargs["params"]["PageNumber"]
        calls.append(page)
        if page == 2 and calls.count(2) == 1:
            raise _denied()
        return _page(200, offset=page * 200) if page < 3 else _page(7, offset=600)

    with patch.object(client, "_request", side_effect=request):
        files = client.list_dataset_files_paginated("owner/ds", page_size=200)

    # Page 2 was retried rather than aborting and discarding page 1.
    assert calls == [1, 2, 2, 3]
    assert len(files) == 407


def test_denial_before_any_successful_tree_read_fails_fast() -> None:
    # Nothing has proven the credential yet, so this really is an authorization
    # answer and must not be retried into a slow, misleading failure.
    client = _client()
    calls: list[int] = []

    def request(method: str, path: str, **kwargs):
        calls.append(kwargs["params"]["PageNumber"])
        raise _denied()

    with patch.object(client, "_request", side_effect=request):
        with pytest.raises(PermissionDeniedError):
            client.list_dataset_files_paginated("owner/ds", page_size=200)

    assert calls == [1]


def test_root_scoped_listing_also_retries_a_spurious_denial(monkeypatch) -> None:
    # The per-directory walk that works around the 3000-entry cap issues the same
    # kind of request, and was seen to be denied the same way.
    client = _client()
    monkeypatch.setattr(legacy_module.time, "sleep", lambda _s: None)
    client._tree_reads_ok = True  # an earlier read already proved the credential
    calls: list[str | None] = []

    def request(method: str, path: str, **kwargs):
        root = kwargs["params"].get("Root")
        calls.append(root)
        if calls.count(root) == 1:
            raise _denied()
        return _page(3)

    with patch.object(client, "_request", side_effect=request):
        entries = client._list_files_page("owner/ds", "dataset", "master", recursive=False, root="level1_000")

    assert calls == ["level1_000", "level1_000"]
    assert len(entries) == 3


def test_a_successful_tree_read_arms_the_retry_for_later_requests(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(legacy_module.time, "sleep", lambda _s: None)
    assert client._tree_reads_ok is False
    outcomes = [_page(2), _denied(), _page(2)]

    def request(method: str, path: str, **kwargs):
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    with patch.object(client, "_request", side_effect=request):
        client._list_files_page("owner/ds", "dataset", "master", recursive=False, root="a")
        assert client._tree_reads_ok is True
        # The denial that follows is now retried instead of raised.
        entries = client._list_files_page("owner/ds", "dataset", "master", recursive=False, root="b")

    assert len(entries) == 2
    assert not outcomes


def test_a_persistently_denied_listing_is_reported_as_a_transport_failure(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(legacy_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(legacy_module, "REPO_TREE_PAGE_MAX_ATTEMPTS", 3)

    def request(method: str, path: str, **kwargs):
        if kwargs["params"]["PageNumber"] == 1:
            return _page(200)
        raise _denied()

    with patch.object(client, "_request", side_effect=request):
        # Surfaced as a transport failure, not as "permission denied": the caller
        # can read the repo, so reporting a denial would misdirect the diagnosis.
        with pytest.raises(NetworkError, match="already-authorized"):
            client.list_dataset_files_paginated("owner/ds", page_size=200)


def test_retry_budget_is_env_tunable() -> None:
    from modelscope_hub.constants import (
        REPO_TREE_PAGE_MAX_ATTEMPTS,
        REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS,
    )

    assert REPO_TREE_PAGE_MAX_ATTEMPTS >= 2
    assert REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS >= 1
