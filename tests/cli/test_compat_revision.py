"""Unit tests for LegacyHubApi.get_valid_revision and related methods.

Uses mock-based testing — no network calls. Verifies the revision
resolution logic that the old ``modelscope.hub.api.HubApi`` provided.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub.compat.hub_api import LegacyHubApi
from modelscope_hub.errors import NotExistError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_rev(name: str, created_at: int = 0) -> dict:
    return {"Revision": name, "CreatedAt": created_at}


def _make_api(branches: list[dict], tags: list[dict]) -> LegacyHubApi:
    """Build a LegacyHubApi with a mocked legacy client."""
    with patch.object(LegacyHubApi, "__init__", lambda self, **kw: None):
        api = LegacyHubApi.__new__(LegacyHubApi)
    api._api = MagicMock()
    api._api.legacy.list_revisions_detail.return_value = (branches, tags)
    api._endpoint = None
    return api


# ---------------------------------------------------------------------------
# get_model_branches_and_tags_details
# ---------------------------------------------------------------------------
class TestGetModelBranchesAndTagsDetails:
    def test_returns_tuple(self):
        branches = [_make_rev("master")]
        tags = [_make_rev("v1.0")]
        api = _make_api(branches, tags)
        b, t = api.get_model_branches_and_tags_details("owner/model")
        assert b == branches
        assert t == tags

    def test_delegates_to_legacy(self):
        api = _make_api([], [])
        api.get_model_branches_and_tags_details("owner/model")
        api._api.legacy.list_revisions_detail.assert_called_once_with("owner/model", "model")


# ---------------------------------------------------------------------------
# get_model_branches_and_tags
# ---------------------------------------------------------------------------
class TestGetModelBranchesAndTags:
    def test_extracts_names(self):
        api = _make_api(
            [_make_rev("master"), _make_rev("dev")],
            [_make_rev("v1.0"), _make_rev("v2.0")],
        )
        branches, tags = api.get_model_branches_and_tags("owner/model")
        assert branches == ["master", "dev"]
        assert tags == ["v1.0", "v2.0"]

    def test_empty(self):
        api = _make_api([], [])
        branches, tags = api.get_model_branches_and_tags("owner/model")
        assert branches == []
        assert tags == []


# ---------------------------------------------------------------------------
# get_valid_revision — no release_timestamp (simplified mode)
# ---------------------------------------------------------------------------
class TestGetValidRevisionSimplified:
    def test_explicit_branch(self):
        api = _make_api([_make_rev("master"), _make_rev("dev")], [])
        assert api.get_valid_revision("o/m", revision="dev") == "dev"

    def test_explicit_tag(self):
        api = _make_api([_make_rev("master")], [_make_rev("v1.0")])
        assert api.get_valid_revision("o/m", revision="v1.0") == "v1.0"

    def test_default_master_when_no_revision(self):
        api = _make_api([_make_rev("master")], [])
        assert api.get_valid_revision("o/m") == "master"

    def test_default_master_when_no_revision_with_tags(self):
        api = _make_api([_make_rev("master")], [_make_rev("v1.0")])
        assert api.get_valid_revision("o/m") == "master"

    def test_nonexistent_revision_raises(self):
        api = _make_api([_make_rev("master")], [_make_rev("v1.0")])
        with pytest.raises(NotExistError, match="no revision"):
            api.get_valid_revision("o/m", revision="v999")


# ---------------------------------------------------------------------------
# get_valid_revision_detail — returns dict
# ---------------------------------------------------------------------------
class TestGetValidRevisionDetail:
    def test_returns_dict_with_revision_key(self):
        api = _make_api([_make_rev("master", 100)], [])
        detail = api.get_valid_revision_detail("o/m", revision="master")
        assert detail["Revision"] == "master"
        assert detail["CreatedAt"] == 100

    def test_tag_detail_preserved(self):
        tag = {"Revision": "v1.0", "CreatedAt": 1000, "CommitId": "abc123"}
        api = _make_api([_make_rev("master")], [tag])
        detail = api.get_valid_revision_detail("o/m", revision="v1.0")
        assert detail["Revision"] == "v1.0"
        assert detail["CommitId"] == "abc123"


# ---------------------------------------------------------------------------
# get_valid_revision_detail — dev mode (release_timestamp far future)
# ---------------------------------------------------------------------------
class TestDevMode:
    FAR_FUTURE = int(time.time()) + 2 * 365 * 24 * 60 * 60

    def test_dev_mode_defaults_to_master(self):
        api = _make_api([_make_rev("master")], [_make_rev("v1.0")])
        detail = api.get_valid_revision_detail(
            "o/m", release_timestamp=self.FAR_FUTURE,
        )
        assert detail["Revision"] == "master"

    def test_dev_mode_explicit_tag(self):
        api = _make_api([_make_rev("master")], [_make_rev("v2.0")])
        detail = api.get_valid_revision_detail(
            "o/m", revision="v2.0", release_timestamp=self.FAR_FUTURE,
        )
        assert detail["Revision"] == "v2.0"

    def test_dev_mode_nonexistent_raises(self):
        api = _make_api([_make_rev("master")], [])
        with pytest.raises(NotExistError):
            api.get_valid_revision_detail(
                "o/m", revision="nope", release_timestamp=self.FAR_FUTURE,
            )


# ---------------------------------------------------------------------------
# get_valid_revision_detail — release mode
# ---------------------------------------------------------------------------
class TestReleaseMode:
    RELEASE_TS = 2000

    def test_explicit_branch_returns_immediately(self):
        api = _make_api(
            [_make_rev("master", 100), _make_rev("dev", 200)],
            [_make_rev("v1.0", 500)],
        )
        detail = api.get_valid_revision_detail(
            "o/m", revision="dev", release_timestamp=self.RELEASE_TS,
        )
        assert detail["Revision"] == "dev"

    def test_no_tags_defaults_to_master(self):
        api = _make_api([_make_rev("master", 100)], [])
        detail = api.get_valid_revision_detail(
            "o/m", release_timestamp=self.RELEASE_TS,
        )
        assert detail["Revision"] == "master"

    def test_no_tags_explicit_master(self):
        api = _make_api([_make_rev("master", 100)], [])
        detail = api.get_valid_revision_detail(
            "o/m", revision="master", release_timestamp=self.RELEASE_TS,
        )
        assert detail["Revision"] == "master"

    def test_no_tags_explicit_nonexistent_raises(self):
        api = _make_api([_make_rev("master")], [])
        with pytest.raises(NotExistError):
            api.get_valid_revision_detail(
                "o/m", revision="v1.0", release_timestamp=self.RELEASE_TS,
            )

    def test_auto_selects_latest_tag_before_release(self):
        api = _make_api(
            [_make_rev("master")],
            [
                _make_rev("v3.0", 1500),
                _make_rev("v2.0", 1000),
                _make_rev("v1.0", 500),
            ],
        )
        detail = api.get_valid_revision_detail(
            "o/m", release_timestamp=self.RELEASE_TS,
        )
        # v3.0 (1500) is the newest with CreatedAt <= 2000
        assert detail["Revision"] == "v3.0"

    def test_no_tag_before_release_falls_back_to_master(self):
        api = _make_api(
            [_make_rev("master")],
            [_make_rev("v1.0", 3000)],
        )
        detail = api.get_valid_revision_detail(
            "o/m", release_timestamp=self.RELEASE_TS,
        )
        assert detail["Revision"] == "master"

    def test_explicit_valid_tag(self):
        api = _make_api(
            [_make_rev("master")],
            [_make_rev("v1.0", 500), _make_rev("v2.0", 1000)],
        )
        detail = api.get_valid_revision_detail(
            "o/m", revision="v1.0", release_timestamp=self.RELEASE_TS,
        )
        assert detail["Revision"] == "v1.0"

    def test_explicit_invalid_tag_raises(self):
        api = _make_api(
            [_make_rev("master")],
            [_make_rev("v1.0", 500)],
        )
        with pytest.raises(NotExistError, match="valid tags"):
            api.get_valid_revision_detail(
                "o/m", revision="v999", release_timestamp=self.RELEASE_TS,
            )

    def test_explicit_master_with_tags_allowed(self):
        """master is accepted even when tags exist, with a warning."""
        api = _make_api(
            [_make_rev("master", 100)],
            [_make_rev("v1.0", 500)],
        )
        detail = api.get_valid_revision_detail(
            "o/m", revision="master", release_timestamp=self.RELEASE_TS,
        )
        assert detail["Revision"] == "master"
