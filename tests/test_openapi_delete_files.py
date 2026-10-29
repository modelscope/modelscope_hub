"""The OpenAPI delete-files endpoint, and why ``delete_files`` does not use it.

The endpoint is strictly narrower than the commit-based deletion the SDK
already performs: it caps at 100 paths, takes no directories or glob patterns,
accepts only a branch, and is all-or-nothing. The commit path has none of those
limits, so :meth:`HubApi.delete_files` keeps using it; routing it here would
turn working calls into failures.

The endpoint is still exposed on the OpenAPI client for callers that want its
``commit_sha`` and its strict guarantee. The regression guard below is the
point of this file: ``HubApi.delete_files`` must stay on the commit path.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import HubApi
from modelscope_hub.constants import OPENAPI_DELETE_FILES_MAX_PATHS
from modelscope_hub.errors import InvalidParameter

_RESULT = {
    "deleted_files": ["a.txt", "b.txt"],
    "failed_files": [],
    "total_files": 2,
    "commit_sha": "a1b2c3",
}


@pytest.fixture
def api() -> HubApi:
    return HubApi(endpoint="https://modelscope.cn", token="ms-test")


# ---------------------------------------------------------------------------
# The endpoint itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("delete_model_files", "/models/alice/m/files"),
        ("delete_dataset_files", "/datasets/alice/m/files"),
    ],
)
def test_the_request_uses_delete_with_a_json_body(api, method, path):
    with mock.patch.object(api.openapi, "_request", return_value=_RESULT) as request_mock:
        result = getattr(api.openapi, method)("alice", "m", {"file_paths": ["a.txt", "b.txt"]})

    assert request_mock.call_args.args == ("DELETE", path)
    assert request_mock.call_args.kwargs["json_body"] == {"file_paths": ["a.txt", "b.txt"]}
    assert result["commit_sha"] == "a1b2c3"


def test_revision_and_commit_message_are_forwarded(api):
    with mock.patch.object(api.openapi, "_request", return_value=_RESULT) as request_mock:
        api.openapi.delete_model_files(
            "alice",
            "m",
            {"file_paths": ["a.txt"], "revision": "dev", "commit_message": "tidy up"},
        )

    body = request_mock.call_args.kwargs["json_body"]
    assert body["revision"] == "dev"
    assert body["commit_message"] == "tidy up"


def test_exactly_the_maximum_number_of_paths_is_accepted(api):
    paths = [f"f{i}.txt" for i in range(OPENAPI_DELETE_FILES_MAX_PATHS)]

    with mock.patch.object(api.openapi, "_request", return_value=_RESULT) as request_mock:
        api.openapi.delete_model_files("alice", "m", {"file_paths": paths})

    assert len(request_mock.call_args.kwargs["json_body"]["file_paths"]) == OPENAPI_DELETE_FILES_MAX_PATHS


def test_too_many_paths_are_refused_before_the_request(api):
    paths = [f"f{i}.txt" for i in range(OPENAPI_DELETE_FILES_MAX_PATHS + 1)]

    with mock.patch.object(api.openapi, "_request") as request_mock:
        with pytest.raises(InvalidParameter, match="at most 100"):
            api.openapi.delete_model_files("alice", "m", {"file_paths": paths})
    request_mock.assert_not_called()


@pytest.mark.parametrize("paths", [[], None, "a.txt"])
def test_an_empty_or_non_list_path_set_is_refused(api, paths):
    with mock.patch.object(api.openapi, "_request") as request_mock:
        with pytest.raises(InvalidParameter, match="non-empty list"):
            api.openapi.delete_dataset_files("alice", "m", {"file_paths": paths})
    request_mock.assert_not_called()


def test_duplicate_paths_are_refused(api):
    with mock.patch.object(api.openapi, "_request") as request_mock:
        with pytest.raises(InvalidParameter, match="duplicate"):
            api.openapi.delete_model_files("alice", "m", {"file_paths": ["a.txt", "a.txt"]})
    request_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Regression guard: HubApi.delete_files must not move
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_delete_files_still_commits_instead_of_calling_the_endpoint(api, repo_type):
    with (
        mock.patch.object(api.uploader, "delete_files", return_value=_RESULT) as uploader_delete,
        mock.patch.object(api.openapi, "_request") as openapi_request,
    ):
        api.delete_files("alice/m", repo_type, ["a.txt", "b.txt"])

    openapi_request.assert_not_called()
    uploader_delete.assert_called_once()
    assert uploader_delete.call_args.kwargs["file_paths"] == ["a.txt", "b.txt"]


def test_delete_files_has_no_hundred_path_ceiling(api):
    """The commit path splits on the server's own action limit, far above 100."""
    paths = [f"f{i}.txt" for i in range(250)]

    with (
        mock.patch.object(api.uploader, "delete_files", return_value=_RESULT) as uploader_delete,
        mock.patch.object(api.openapi, "_request") as openapi_request,
    ):
        api.delete_files("alice/m", "model", paths)

    openapi_request.assert_not_called()
    assert len(uploader_delete.call_args.kwargs["file_paths"]) == 250


def test_delete_files_still_resolves_glob_patterns(api):
    """The endpoint takes no patterns; the commit path resolves them remotely."""
    remote = [
        mock.Mock(path="keep.bin", type="blob"),
        mock.Mock(path="drop/a.json", type="blob"),
        mock.Mock(path="drop/b.json", type="blob"),
    ]

    with (
        mock.patch.object(api, "list_repo_files", return_value=remote),
        mock.patch.object(api.uploader, "delete_files", return_value=_RESULT) as uploader_delete,
        mock.patch.object(api.openapi, "_request") as openapi_request,
    ):
        api.delete_files("alice/m", "model", delete_patterns="*.json")

    openapi_request.assert_not_called()
    assert uploader_delete.call_args.kwargs["file_paths"] == ["drop/a.json", "drop/b.json"]


def test_delete_files_accepts_a_tag_revision(api):
    """``revision`` is forwarded as given; the endpoint would reject a tag."""
    with (
        mock.patch.object(api.uploader, "delete_files", return_value=_RESULT) as uploader_delete,
        mock.patch.object(api.openapi, "_request") as openapi_request,
    ):
        api.delete_files("alice/m", "model", ["a.txt"], revision="v1.0")

    openapi_request.assert_not_called()
    assert uploader_delete.call_args.kwargs["revision"] == "v1.0"
