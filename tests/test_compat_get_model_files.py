"""Unit tests for the legacy-compatible ``LegacyHubApi.get_model_files``.

Network-free: the underlying ``HubApi.list_repo_files`` is mocked so we only
verify the compat wrapper's signature and parameter forwarding. Regression
guard for callers (e.g. vLLM) that pass the historical ``revision`` / ``root``
keyword arguments.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from modelscope_hub.compat import LegacyHubApi


def _fake_files():
    return [
        SimpleNamespace(path="config.json", size=10),
        SimpleNamespace(path="model.safetensors", size=100),
        SimpleNamespace(path="subdir/extra.bin", size=5),
    ]


class TestGetModelFilesLegacyCompat:
    def test_revision_is_accepted_and_forwarded(self):
        lha = LegacyHubApi()
        with mock.patch.object(
                lha._api, "list_repo_files",
                return_value=_fake_files()) as m:
            out = lha.get_model_files(
                "Qwen/Qwen2.5-1.5B-Instruct", revision="v2")

        assert [f["Path"] for f in out] == [
            "config.json", "model.safetensors", "subdir/extra.bin",
        ]
        _, kwargs = m.call_args
        assert kwargs["revision"] == "v2"

    def test_root_restricts_to_subpath(self):
        lha = LegacyHubApi()
        with mock.patch.object(
                lha._api, "list_repo_files", return_value=_fake_files()):
            out = lha.get_model_files("owner/name", root="subdir")

        assert [f["Path"] for f in out] == ["subdir/extra.bin"]

    def test_tolerates_legacy_transport_kwargs(self):
        lha = LegacyHubApi()
        with mock.patch.object(
                lha._api, "list_repo_files", return_value=_fake_files()):
            # Historical kwargs must not raise "unexpected keyword argument".
            out = lha.get_model_files(
                "owner/name", revision="master",
                use_cookies=True, headers={})

        assert len(out) == 3

    def test_default_revision_none_forwarded(self):
        lha = LegacyHubApi()
        with mock.patch.object(
                lha._api, "list_repo_files",
                return_value=_fake_files()) as m:
            lha.get_model_files("owner/name")

        _, kwargs = m.call_args
        assert kwargs["revision"] is None
        assert kwargs["recursive"] is True
