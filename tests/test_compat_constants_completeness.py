"""Completeness / consistency of the legacy compat constant surface.

``modelscope.hub.constants`` re-exports these names from
``modelscope_hub.compat.constants``, which in turn re-exports them from the root
``modelscope_hub.constants`` (the single source of truth). Two invariants must
hold and are pinned below:

* every legacy constant is present in the compat surface with the exact legacy
  value and type (what the modelscope SDK imports); and
* the root module is a superset of the compat surface for the shared prefixes
  (so a legacy alias is never defined only in the compat shim again).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import modelscope_hub.compat.constants as hub_mod

# Exact values (and types) as historically defined in modelscope.hub.constants.
_EXPECTED: dict[str, object] = {
    "MODEL_ID_SEPARATOR": "/",
    "DEFAULT_MODELSCOPE_GROUP": "damo",
    "DEFAULT_MODELSCOPE_DOMAIN": "www.modelscope.cn",
    "DEFAULT_MODELSCOPE_INTL_DOMAIN": "www.modelscope.ai",
    "DEFAULT_MODELSCOPE_DATA_ENDPOINT": "https://www.modelscope.cn",
    "DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT": "https://www.modelscope.ai",
    "DEFAULT_SKILLS_DIR": os.path.join(os.path.expanduser("~"), ".agents", "skills"),
    "DEFAULT_CREDENTIALS_PATH": Path.home().joinpath(".modelscope", "credentials"),
}


@pytest.mark.parametrize(("name", "expected"), sorted(_EXPECTED.items()))
def test_compat_exposes_legacy_constant(name, expected):
    assert hasattr(hub_mod, name), f"{name} missing from modelscope_hub.compat.constants"
    value = getattr(hub_mod, name)
    assert value == expected
    # Type must match too: modelscope derives MODELSCOPE_CREDENTIALS_PATH via
    # DEFAULT_CREDENTIALS_PATH.as_posix(), so a str stand-in would break it.
    assert type(value) is type(expected)


def test_credentials_path_is_a_path_supporting_as_posix():
    assert isinstance(hub_mod.DEFAULT_CREDENTIALS_PATH, Path)
    assert hub_mod.DEFAULT_CREDENTIALS_PATH.as_posix().endswith("/.modelscope/credentials")


def test_matches_installed_modelscope_when_available():
    """Mirror the field completeness/consistency check against the real SDK.

    Skipped when ``modelscope`` is not importable (e.g. the hub CI matrix),
    so the pinned assertions above remain the source of truth there.
    """
    legacy = pytest.importorskip("modelscope.hub.constants")
    prefixes = ("UPLOAD_", "REPO_", "MODEL_", "DEFAULT_", "TEMPORARY_", "FILE_")
    targets = [a for a in dir(legacy) if not a.startswith("_") and a.startswith(prefixes)]
    missing = [a for a in targets if not hasattr(hub_mod, a)]
    mismatch = [
        (a, getattr(legacy, a), getattr(hub_mod, a))
        for a in targets
        if hasattr(hub_mod, a) and getattr(legacy, a) != getattr(hub_mod, a)
    ]
    assert not missing, f"missing from compat: {missing}"
    assert not mismatch, f"value mismatch: {mismatch}"


def test_root_module_is_superset_of_compat_surface():
    """``modelscope_hub.constants`` must expose every prefixed compat constant.

    compat re-exports from the root module, so the root is the single source of
    truth; a name present in compat but missing from (or unequal in) the root
    means a legacy alias regressed into being compat-only again.
    """
    import modelscope_hub.constants as root_mod

    prefixes = ("UPLOAD_", "REPO_", "MODEL_", "DEFAULT_", "TEMPORARY_", "FILE_", "MCP_")
    keys = [a for a in dir(hub_mod) if not a.startswith("_") and a.startswith(prefixes)]
    missing = [k for k in keys if not hasattr(root_mod, k)]
    mismatch = [
        (k, getattr(hub_mod, k), getattr(root_mod, k))
        for k in keys
        if hasattr(root_mod, k) and getattr(hub_mod, k) != getattr(root_mod, k)
    ]
    assert not missing, f"missing from modelscope_hub.constants: {missing}"
    assert not mismatch, f"value mismatch: {mismatch}"
