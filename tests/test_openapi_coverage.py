"""Guard against the OpenAPI specification drifting away from the client.

Every gap this work closed had the same cause: the service published new
operations and nothing in the suite noticed. `tests/data/openapi.json` is a
vendored copy of the live document, and these tests assert that the tags the SDK
claims to cover are covered *completely*.

When the spec is refreshed and a covered tag gained an operation, the first test
fails and names it. Tags still to be implemented are listed in `_DEFERRED_TAGS`,
which doubles as the remaining to-do list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelscope_hub._openapi import OPERATION_REGISTRY, OpenAPIClient
from modelscope_hub.constants import TokenScope

_SPEC_PATH = Path(__file__).parent / "data" / "openapi.json"
_HTTP_METHODS = frozenset({"get", "post", "put", "delete", "patch", "head", "options"})

# Tags the SDK implements end to end.
_COVERED_TAGS = frozenset({"Agent-IDP", "MCP", "Studios"})

# Tags not implemented yet. Remove an entry as its tag lands; until then the
# guard would otherwise fail on operations nobody has promised.
_DEFERRED_TAGS = frozenset(
    {
        "User",
        "Models",
        "Datasets",
        "Skills",
        "Files",
        "Collections",
        "Galleries",
        "Magicube",
    }
)


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(_SPEC_PATH.read_text(encoding="utf-8"))


def _operations(spec: dict) -> list[tuple[str, str, str, str]]:
    """Flatten the spec into ``(tag, operation_id, method, path)`` rows."""
    rows: list[tuple[str, str, str, str]] = []
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            if method not in _HTTP_METHODS:
                continue
            tags = operation.get("tags") or ["<untagged>"]
            rows.append((tags[0], operation["operationId"], method.upper(), path))
    return rows


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def test_every_covered_operation_is_registered(spec):
    expected = {op_id for tag, op_id, _, _ in _operations(spec) if tag in _COVERED_TAGS}
    missing = sorted(expected - set(OPERATION_REGISTRY))
    assert not missing, (
        f"{len(missing)} operation(s) in the covered tags {sorted(_COVERED_TAGS)} are not in "
        f"OPERATION_REGISTRY: {missing}. Implement them on OpenAPIClient and register them, "
        f"or move the tag to _DEFERRED_TAGS."
    )


def test_registry_has_no_unknown_operations(spec):
    known = {op_id for _, op_id, _, _ in _operations(spec)}
    unknown = sorted(set(OPERATION_REGISTRY) - known)
    assert not unknown, f"OPERATION_REGISTRY references operationIds absent from the spec: {unknown}"


def test_registry_does_not_claim_deferred_tags(spec):
    deferred = {op_id for tag, op_id, _, _ in _operations(spec) if tag in _DEFERRED_TAGS}
    overlap = sorted(deferred & set(OPERATION_REGISTRY))
    assert not overlap, (
        f"These operations are registered but their tag is still listed as deferred: {overlap}. "
        f"Remove the tag from _DEFERRED_TAGS once it is fully covered."
    )


def test_every_tag_is_accounted_for(spec):
    """A brand-new tag must be an explicit decision, not an oversight."""
    tags = {tag for tag, _, _, _ in _operations(spec)}
    unclassified = sorted(tags - _COVERED_TAGS - _DEFERRED_TAGS)
    assert not unclassified, f"Unclassified tag(s) {unclassified}: add each to _COVERED_TAGS or _DEFERRED_TAGS."


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("operation_id", "entry"), sorted(OPERATION_REGISTRY.items()))
def test_registered_method_exists_and_is_callable(operation_id, entry):
    method_name, scope = entry
    method = getattr(OpenAPIClient, method_name, None)
    assert method is not None, f"{operation_id} maps to OpenAPIClient.{method_name}, which does not exist"
    assert callable(method)
    assert isinstance(scope, TokenScope)


def test_registry_maps_each_operation_to_a_distinct_method():
    methods = [method for method, _ in OPERATION_REGISTRY.values()]
    duplicates = sorted({m for m in methods if methods.count(m) > 1})
    assert not duplicates, f"One method serves several operations: {duplicates}"


# ---------------------------------------------------------------------------
# The vendored spec itself
# ---------------------------------------------------------------------------
def test_spec_is_the_expected_document(spec):
    assert spec["openapi"].startswith("3.1")
    assert spec["info"]["title"] == "ModelScope OpenAPI"
    assert spec["servers"][0]["url"].endswith("/openapi/v1")


def test_covered_tags_account_for_every_registered_entry(spec):
    """Registry size must equal the covered operation count -- no silent drift."""
    covered = [op_id for tag, op_id, _, _ in _operations(spec) if tag in _COVERED_TAGS]
    assert len(OPERATION_REGISTRY) == len(covered)
