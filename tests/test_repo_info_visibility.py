"""``RepoInfo.visibility`` must stay a :class:`Visibility` integer.

The model and dataset detail payloads gained ``visibility: "gated"``, a third
string state the 1/3/5 integer encoding cannot express. Letting it fall through
as a raw string would break ``.label``, ``int()`` and equality against the enum
-- and the compat dict, which the legacy SDK reads as an integer.

A gated repository maps to PRIVATE: that is what it is on the legacy wire
(``Visibility=PRIVATE`` plus ``ProtectedMode=1``), and the approval mode travels
separately in ``gated`` / ``gated_mode``. The server reports ``private: false``
for gated repos, so without that mapping they would read back as PUBLIC.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import HubApi
from modelscope_hub.compat import LegacyHubApi
from modelscope_hub.constants import RepoType, Visibility


def _info(payload: dict, repo_type: RepoType = RepoType.MODEL):
    return HubApi._repo_info_from_payload(payload, repo_type, owner_hint="alice", name_hint="thing")


# ---------------------------------------------------------------------------
# String visibility (current contract)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"visibility": "public", "private": False, "gated": False}, Visibility.PUBLIC),
        ({"visibility": "private", "private": True, "gated": False}, Visibility.PRIVATE),
        # A gated repo: the server says ``private: false`` yet it is not public.
        ({"visibility": "gated", "private": False, "gated": True}, Visibility.PRIVATE),
    ],
)
def test_string_visibility_maps_to_the_integer_enum(payload, expected):
    info = _info(payload)

    assert info.visibility == expected
    assert isinstance(info.visibility, Visibility)


def test_gated_visibility_implies_the_gated_flag():
    """``gated`` may be absent from a summary payload; visibility still implies it."""
    info = _info({"visibility": "gated"})

    assert info.visibility == Visibility.PRIVATE
    assert info.gated is True


def test_an_explicit_gated_flag_is_not_overwritten():
    info = _info({"visibility": "gated", "gated": True, "gated_mode": "manual"})

    assert info.gated is True
    assert info.gated_mode == "manual"


# ---------------------------------------------------------------------------
# Older payloads with no visibility field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"private": True}, Visibility.PRIVATE),
        ({"private": False}, Visibility.PUBLIC),
        # ``gated`` is checked first: a gated repo reports ``private: false``,
        # so trusting that flag would report it as public.
        ({"private": False, "gated": True}, Visibility.PRIVATE),
        ({"gated": True}, Visibility.PRIVATE),
    ],
)
def test_boolean_payloads_still_resolve(payload, expected):
    info = _info(payload)

    assert info.visibility == expected
    assert isinstance(info.visibility, Visibility)


def test_an_integer_visibility_is_preserved():
    assert _info({"Visibility": int(Visibility.INTERNAL)}).visibility == Visibility.INTERNAL


# ---------------------------------------------------------------------------
# Studio visibility must not regress
# ---------------------------------------------------------------------------
def test_studio_protected_is_preserved_verbatim():
    """``protected`` has no integer equivalent, so it is kept as the string."""
    info = _info({"visibility": "protected"}, RepoType.STUDIO)

    assert info.visibility == "protected"


# ---------------------------------------------------------------------------
# Compat surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("visibility", ["public", "private", "gated"])
def test_the_compat_dict_reports_visibility_as_an_integer(visibility):
    """The legacy SDK reads ``Visibility`` as an int, never an enum or string."""
    compat = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
    payload = {"id": "alice/thing", "visibility": visibility, "private": False}

    with mock.patch.object(compat._api.openapi, "_request", return_value=payload):
        out = compat.get_model("alice/thing")

    assert isinstance(out["Visibility"], int)
    assert out["Visibility"] in {int(v) for v in Visibility}


def test_a_gated_repo_is_not_reported_as_public_through_compat():
    compat = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
    payload = {"id": "alice/thing", "visibility": "gated", "private": False, "gated": True}

    with mock.patch.object(compat._api.openapi, "_request", return_value=payload):
        out = compat.get_model("alice/thing")

    assert out["Visibility"] == int(Visibility.PRIVATE)
