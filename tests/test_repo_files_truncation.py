"""Unit tests for the client-side mitigation of the ``repo/files`` entry cap.

The server truncates every ``repo/files`` listing at
``REPO_FILES_TRUNCATION_LIMIT`` entries, returns ``HTTP 200`` with no truncation
marker, and ignores all pagination parameters. ``LegacyClient.list_repo_files``
therefore treats a listing that lands exactly on the cap as "there may be more"
and re-enumerates the tree with ``Root``-scoped requests.

Network-free: ``_list_files_page`` is replaced by an in-memory repo that mimics
the endpoint, cap included. The cap is patched down to a small number so the
fixtures stay readable.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import _legacy_api
from modelscope_hub._legacy_api import LegacyClient

CAP = 5


def _blob(path: str) -> dict:
    return {"Path": path, "Name": path.rsplit("/", 1)[-1], "Type": "blob", "Size": 1}


def _tree(path: str) -> dict:
    return {"Path": path, "Name": path.rsplit("/", 1)[-1], "Type": "tree", "Size": 0}


class _FakeRepo:
    """Stand-in for ``repo/files``, including its silent truncation."""

    def __init__(self, blobs: list[str], cap: int = CAP) -> None:
        self.blobs = sorted(blobs)
        self.cap = cap
        self.calls: list[tuple[str | None, bool]] = []

    def _entries(self, root: str | None, recursive: bool) -> list[dict]:
        prefix = f"{root}/" if root else ""
        found: dict[str, dict] = {}
        for blob in self.blobs:
            if not blob.startswith(prefix):
                continue
            rest = blob[len(prefix) :]
            if "/" not in rest:
                found[blob] = _blob(blob)
            elif recursive:
                parts = rest.split("/")
                for depth in range(1, len(parts)):
                    nested = prefix + "/".join(parts[:depth])
                    found[nested] = _tree(nested)
                found[blob] = _blob(blob)
            else:
                child = prefix + rest.split("/")[0]
                found[child] = _tree(child)
        return list(found.values())

    def listing(self, repo_id, repo_type, revision, *, recursive, root=None) -> list[dict]:
        """Signature-compatible replacement for ``_list_files_page``."""
        self.calls.append((root, recursive))
        return self._entries(root, recursive)[: self.cap]

    def full_paths(self, recursive: bool = True) -> set[str]:
        """Every path the endpoint would expose if it did not truncate."""
        return {_legacy_api._entry_path(e) for e in self._entries(None, recursive)}


@pytest.fixture
def client() -> LegacyClient:
    return LegacyClient(token=None, endpoint="https://example.com")


def _install(repo: _FakeRepo):
    """Patch the transport and shrink the cap to the fake repo's cap."""
    return (
        mock.patch.object(LegacyClient, "_list_files_page", side_effect=repo.listing),
        mock.patch.object(_legacy_api, "REPO_FILES_TRUNCATION_LIMIT", repo.cap),
    )


class TestFastPath:
    def test_small_repo_takes_exactly_one_request(self, client):
        repo = _FakeRepo(["config.json", "model.safetensors", "sub/extra.bin"])
        transport, cap = _install(repo)
        with transport, cap:
            out = client.list_repo_files("owner/name", "model")

        assert len(repo.calls) == 1
        assert repo.calls[0] == (None, True)
        assert {_legacy_api._entry_path(e) for e in out} == {
            "config.json",
            "model.safetensors",
            "sub",
            "sub/extra.bin",
        }

    def test_listing_just_below_cap_is_not_rewalked(self, client):
        # cap - 1 entries: a complete tree that must not trigger the fallback.
        repo = _FakeRepo([f"f{i}.bin" for i in range(CAP - 1)])
        transport, cap = _install(repo)
        with transport, cap:
            out = client.list_repo_files("owner/name", "model")

        assert len(repo.calls) == 1
        assert len(out) == CAP - 1


class TestTruncatedTreeIsRewalked:
    def _repo(self) -> _FakeRepo:
        # Root recursive listing hits the cap, so the tree must be walked:
        # 3 top-level blobs + two directories holding 4 blobs each.
        return _FakeRepo(
            [
                "README.md",
                "config.json",
                "model.safetensors",
                *[f"weights/part{i}.bin" for i in range(4)],
                *[f"tokenizer/vocab{i}.txt" for i in range(4)],
            ]
        )

    def test_full_tree_is_recovered(self, client):
        repo = self._repo()
        transport, cap = _install(repo)
        with transport, cap:
            out = client.list_repo_files("owner/name", "model")

        got = {_legacy_api._entry_path(e) for e in out}
        assert got == repo.full_paths()
        # The truncated single call would have returned only `cap` entries.
        assert len(got) > repo.cap

    def test_walk_scopes_requests_with_root(self, client):
        repo = self._repo()
        transport, cap = _install(repo)
        with transport, cap:
            client.list_repo_files("owner/name", "model")

        roots_visited = {root for root, _ in repo.calls}
        assert {"weights", "tokenizer"} <= roots_visited
        # The caller's initial listing is reused, never repeated.
        assert [call for call in repo.calls if call == (None, True)] == [(None, True)]
        # The root level is listed shallowly to discover the child directories.
        assert (None, False) in repo.calls

    def test_entries_are_deduplicated(self, client):
        repo = self._repo()
        transport, cap = _install(repo)
        with transport, cap:
            out = client.list_repo_files("owner/name", "model")

        paths = [_legacy_api._entry_path(e) for e in out]
        assert len(paths) == len(set(paths))

    def test_nested_oversized_subtree_is_split_further(self, client):
        # `deep` alone exceeds the cap and only resolves via its children.
        repo = _FakeRepo(
            [
                "README.md",
                *[f"deep/a/f{i}.bin" for i in range(4)],
                *[f"deep/b/f{i}.bin" for i in range(4)],
            ]
        )
        transport, cap = _install(repo)
        with transport, cap:
            out = client.list_repo_files("owner/name", "model")

        assert {_legacy_api._entry_path(e) for e in out} == repo.full_paths()
        assert {"deep", "deep/a", "deep/b"} <= {root for root, _ in repo.calls}


class TestIncompleteResultsAreReported:
    @staticmethod
    def _warnings(warn_mock) -> list[str]:
        return [call.args[0] for call in warn_mock.call_args_list]

    def test_flat_oversized_directory_warns(self, client):
        # A single directory with more direct blob children than the cap cannot
        # be enumerated: no sub-directories exist to scope requests by.
        repo = _FakeRepo([f"flat/f{i}.bin" for i in range(CAP + 3)])
        transport, cap = _install(repo)
        with transport, cap, mock.patch.object(_legacy_api.logger, "warning") as warn:
            out = client.list_repo_files("owner/name", "model")

        assert len(out) <= repo.cap + 1  # partial, best effort
        assert any("incomplete" in message for message in self._warnings(warn))

    def test_non_recursive_listing_at_cap_warns_without_walking(self, client):
        repo = _FakeRepo([f"f{i}.bin" for i in range(CAP + 3)])
        transport, cap = _install(repo)
        with transport, cap, mock.patch.object(_legacy_api.logger, "warning") as warn:
            out = client.list_repo_files("owner/name", "model", recursive=False)

        assert len(repo.calls) == 1  # no fallback walk for a shallow listing
        assert len(out) == repo.cap
        assert any("incomplete" in message for message in self._warnings(warn))

    def test_request_budget_stops_the_walk(self, client):
        # Root holds 3 directories (a shallow listing fits under the cap), but the
        # recursive listing does not — so the walk starts and then runs out of budget.
        repo = _FakeRepo([f"d{d}/f{i}.bin" for d in range(3) for i in range(4)])
        transport, cap = _install(repo)
        with (
            transport,
            cap,
            mock.patch.object(_legacy_api, "REPO_TREE_MAX_REQUESTS", 3),
            mock.patch.object(_legacy_api.logger, "warning") as warn,
        ):
            out = client.list_repo_files("owner/name", "model")

        assert len(repo.calls) <= 1 + 3  # initial detection listing + the walk's budget
        assert len(out) > 0  # whatever was collected is still returned
        assert any("MODELSCOPE_REPO_TREE_MAX_REQUESTS" in message for message in self._warnings(warn))


class TestDatasetRouting:
    def test_recursive_dataset_listing_uses_the_paginated_endpoint(self, client):
        pages = [{"Path": "data/train.csv", "Type": "blob", "Size": 1}]
        with (
            mock.patch.object(LegacyClient, "list_dataset_files_paginated", return_value=pages) as paged,
            mock.patch.object(LegacyClient, "_list_files_page") as single,
        ):
            out = client.list_repo_files("owner/ds", "dataset", revision="v1", root="data")

        assert out == pages
        single.assert_not_called()
        _, kwargs = paged.call_args
        assert kwargs["revision"] == "v1"
        assert kwargs["root_path"] == "data"

    def test_non_recursive_dataset_listing_stays_single_page(self, client):
        with (
            mock.patch.object(LegacyClient, "list_dataset_files_paginated") as paged,
            mock.patch.object(LegacyClient, "_list_files_page", return_value=[]) as single,
        ):
            client.list_repo_files("owner/ds", "dataset", recursive=False)

        paged.assert_not_called()
        single.assert_called_once()
