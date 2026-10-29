"""Routing of ``create_repo`` between the OpenAPI and legacy create endpoints.

The OpenAPI schemas are closed: ``visibility`` has no ``internal``,
``POST /models`` restricts ``license`` to a published set, and both endpoints
answer 400 for undeclared fields. Inputs the schema cannot express therefore
keep using the legacy endpoint, so a call that works today keeps working. These
tests pin which surface each input lands on, asserting the method and path
actually sent rather than a mock's return value.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import HubApi
from modelscope_hub import api as api_module
from modelscope_hub.constants import GatedMode, RepoType, RepoVisibility, Visibility
from modelscope_hub.errors import AlreadyExistsError, InvalidParameter, NotExistError, ServerError


@pytest.fixture
def api() -> HubApi:
    return HubApi(endpoint="https://modelscope.cn", token="ms-test")


def _legacy_response(data=None):
    response = mock.MagicMock()
    response.json.return_value = {"Data": data or {}}
    return response


class _Routes:
    """Captures which surface a create call reached, and with what body."""

    def __init__(self, openapi_request, legacy_request):
        self.openapi = openapi_request
        self.legacy = legacy_request

    @property
    def went_openapi(self) -> bool:
        return self.openapi.called and not self.legacy.called

    @property
    def went_legacy(self) -> bool:
        return self.legacy.called and not self.openapi.called

    @property
    def openapi_body(self) -> dict:
        return self.openapi.call_args.kwargs["json_body"]

    @property
    def legacy_body(self) -> dict:
        kwargs = self.legacy.call_args.kwargs
        if "json_body" in kwargs:
            return kwargs["json_body"]
        # The dataset endpoint posts multipart form fields.
        return {k: v[1] for k, v in kwargs["files"].items()}


@pytest.fixture
def routes(api: HubApi):
    created = {"id": "alice/thing", "owner": "alice", "repo_name": "thing"}
    with (
        mock.patch.object(api.openapi, "_request", return_value=created) as openapi_request,
        mock.patch.object(api.legacy, "_request", return_value=_legacy_response(created)) as legacy_request,
    ):
        yield _Routes(openapi_request, legacy_request)


# ---------------------------------------------------------------------------
# Inputs the OpenAPI schema expresses
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize(
    ("visibility", "expected"),
    [
        (None, None),
        ("public", "public"),
        ("private", "private"),
        (Visibility.PUBLIC, "public"),
        (Visibility.PRIVATE, "private"),
        ("gated", "gated"),
    ],
)
def test_ordinary_visibility_uses_openapi(api, routes, repo_type, visibility, expected):
    api.create_repo("alice/thing", repo_type, visibility=visibility)

    assert routes.went_openapi
    assert routes.openapi.call_args.args == ("POST", f"/{repo_type}s")
    assert routes.openapi_body.get("visibility") == expected


def test_license_in_the_enum_uses_openapi(api, routes):
    api.create_repo("alice/thing", "model", license="apache-2.0")

    assert routes.went_openapi
    assert routes.openapi_body["license"] == "apache-2.0"


def test_display_name_license_other_and_task_reach_openapi(api, routes):
    api.create_repo(
        "alice/thing",
        "model",
        chinese_name="我的模型",
        description="desc",
        license="other",
        license_name="Custom",
        license_link="https://example.com/LICENSE",
        task="text-generation",
    )

    assert routes.went_openapi
    assert routes.openapi_body == {
        "owner": "alice",
        "repo_name": "thing",
        "display_name": "我的模型",
        "description": "desc",
        "license": "other",
        "license_name": "Custom",
        "license_link": "https://example.com/LICENSE",
        "task": "text-generation",
    }


def test_omitted_license_is_not_sent(api, routes):
    """The legacy fallback value ``Apache-2.0`` is outside the OpenAPI enum."""
    api.create_repo("alice/thing", "model")

    assert routes.went_openapi
    assert "license" not in routes.openapi_body


def test_task_is_rejected_for_datasets(api, routes):
    """Kept as a defensive guard; datasets are routed to legacy before this."""
    with pytest.raises(InvalidParameter, match="task is only supported"):
        api._create_via_openapi(
            RepoType.DATASET,
            "alice",
            "thing",
            visibility=None,
            repo_visibility=None,
            license=None,
            chinese_name=None,
            description=None,
            gated=None,
            license_name=None,
            license_link=None,
            task="text-classification",
            extra={},
        )


# ---------------------------------------------------------------------------
# Gated access
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["auto", "manual", GatedMode.MANUAL])
def test_gated_mode_is_sent_with_gated_visibility(api, routes, mode):
    api.create_repo("alice/thing", "model", visibility="gated", gated=mode)

    assert routes.went_openapi
    body = routes.openapi_body
    assert body["visibility"] == "gated"
    assert body["gated_mode"] == str(mode)


def test_gated_without_visibility_implies_gated_visibility(api, routes):
    api.create_repo("alice/thing", "model", gated="manual")

    assert routes.openapi_body["visibility"] == "gated"
    assert routes.openapi_body["gated_mode"] == "manual"


def test_legacy_boolean_gated_mode_keeps_the_legacy_wire_format(api, routes):
    """The historical boolean must produce the request it always produced.

    It carries no approval mode, and the two surfaces default differently:
    legacy sends ``ProtectedMode=1`` and lets the server decide, while the
    OpenAPI schema defaults ``gated_mode`` to ``auto``. Translating it could
    silently turn a review-gated repository into an auto-approving one.
    """
    api.create_repo("alice/thing", "model", gated_mode=True)

    assert routes.went_legacy
    body = routes.legacy_body
    assert body["Visibility"] == int(Visibility.PRIVATE)
    assert body["ProtectedMode"] == 1


def test_gated_mode_false_keeps_the_legacy_wire_format(api, routes):
    """Preserves the legacy behaviour of ``--no-gated`` with no visibility."""
    api.create_repo("alice/thing", "model", gated_mode=False)

    assert routes.went_legacy
    body = routes.legacy_body
    assert body["Visibility"] == int(Visibility.PRIVATE)
    assert body["ProtectedMode"] == 2


def test_gated_mode_with_public_visibility_warns_and_is_ignored(api, routes, caplog, monkeypatch):
    """Unchanged legacy behaviour: warn, ignore the gating, stay public."""
    monkeypatch.setattr(api_module.logger, "handlers", [*api_module.logger.handlers, caplog.handler])
    with caplog.at_level("WARNING", logger="modelscope_hub.api"):
        api.create_repo("alice/thing", "model", visibility="public", gated_mode=True)

    assert routes.went_legacy
    body = routes.legacy_body
    assert body["Visibility"] == int(Visibility.PUBLIC)
    assert "ProtectedMode" not in body
    assert "only effective when visibility is PRIVATE" in caplog.text


def test_the_new_gated_parameter_goes_through_openapi(api, routes):
    """``gated`` can state the approval mode, so it has no legacy equivalent."""
    api.create_repo("alice/thing", "model", gated="manual")

    assert routes.went_openapi
    assert routes.openapi_body["visibility"] == "gated"
    assert routes.openapi_body["gated_mode"] == "manual"


def test_gated_false_without_visibility_implies_private_on_openapi(api, routes):
    api.create_repo("alice/thing", "model", gated=False)

    assert routes.went_openapi
    assert routes.openapi_body["visibility"] == RepoVisibility.PRIVATE.value
    assert "gated_mode" not in routes.openapi_body


def test_gated_with_public_visibility_warns_and_is_dropped(api, routes, caplog, monkeypatch):
    """The endpoint 400s on the pair, so mirror the legacy warn-and-ignore."""
    # The SDK logger deliberately does not propagate, so caplog needs attaching.
    monkeypatch.setattr(api_module.logger, "handlers", [*api_module.logger.handlers, caplog.handler])
    with caplog.at_level("WARNING", logger="modelscope_hub.api"):
        api.create_repo("alice/thing", "model", visibility="public", gated="manual")

    assert routes.openapi_body["visibility"] == "public"
    assert "gated_mode" not in routes.openapi_body
    assert "only effective when visibility is gated" in caplog.text


def test_an_unknown_gated_value_is_rejected(api, routes):
    with pytest.raises(InvalidParameter, match="gated must be"):
        api.create_repo("alice/thing", "model", gated="sometimes")


@pytest.mark.parametrize(
    ("extra", "expected_route"),
    [({}, "openapi"), ({"create_default_config": True}, "legacy")],
)
def test_private_plus_an_approval_mode_means_gated_on_either_route(api, routes, extra, expected_route):
    """``private`` + a mode is what the legacy encoding always called gated.

    The two routes must agree: adding an unrelated field should not change the
    access policy the repository ends up with.
    """
    api.create_repo("alice/thing", "model", visibility="private", gated="manual", **extra)

    if expected_route == "openapi":
        assert routes.went_openapi
        assert routes.openapi_body["visibility"] == "gated"
        assert routes.openapi_body["gated_mode"] == "manual"
    else:
        assert routes.went_legacy
        assert routes.legacy_body["Visibility"] == int(Visibility.PRIVATE)
        assert routes.legacy_body["ProtectedMode"] == 1


def test_gated_visibility_survives_a_forced_legacy_route(api, routes):
    """``visibility="gated"`` states the intent through visibility alone, so the
    legacy encoding has to restate it as ``ProtectedMode``."""
    api.create_repo("alice/thing", "model", visibility="gated", create_default_config=True)

    assert routes.went_legacy
    assert routes.legacy_body["Visibility"] == int(Visibility.PRIVATE)
    assert routes.legacy_body["ProtectedMode"] == 1


# ---------------------------------------------------------------------------
# Inputs the OpenAPI schema cannot express -> legacy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("visibility", ["internal", Visibility.INTERNAL, 3, "3"])
def test_internal_visibility_falls_back_to_legacy(api, routes, visibility):
    api.create_repo("alice/thing", "model", visibility=visibility)

    assert routes.went_legacy
    assert routes.legacy.call_args.args == ("POST", "models")
    assert routes.legacy_body["Visibility"] == int(Visibility.INTERNAL)


@pytest.mark.parametrize("license_value", ["bsd-2-clause", "unlicense", "my-company-eula"])
def test_a_model_license_outside_the_enum_falls_back_to_legacy(api, routes, license_value):
    api.create_repo("alice/thing", "model", license=license_value)

    assert routes.went_legacy
    assert routes.legacy_body["License"] == license_value


def test_a_dataset_license_is_unconstrained_and_stays_on_openapi(api, routes):
    """``POST /datasets`` declares no license enum, unlike ``POST /models``."""
    api.create_repo("alice/thing", "dataset", license="my-company-eula")

    assert routes.went_openapi
    assert routes.openapi_body["license"] == "my-company-eula"


def test_an_undeclared_extra_field_falls_back_to_legacy(api, routes):
    api.create_repo("alice/thing", "model", create_default_config=True)

    assert routes.went_legacy
    assert routes.legacy_body["create_default_config"] is True


def test_protected_visibility_is_rejected_for_models(api, routes):
    with pytest.raises(ValueError):
        api.create_repo("alice/thing", "model", visibility="protected")


def test_task_is_routed_to_legacy_for_datasets(api, routes):
    """Only the model schema declares ``task``; legacy took it through extra."""
    api.create_repo("alice/thing", "dataset", task="text-classification")

    assert routes.went_legacy
    assert routes.legacy_body["task"] == "text-classification"


def test_the_newly_named_params_keep_their_legacy_key_names(api, routes):
    """They were ``extra`` keys before the signature grew; renaming them on the
    legacy wire would change a request that used to work."""
    api.create_repo(
        "alice/thing",
        "model",
        gated_mode=True,  # forces the legacy route
        task="text-generation",
        license_name="Custom",
        license_link="https://example.com/LICENSE",
    )

    assert routes.went_legacy
    body = routes.legacy_body
    # Appended verbatim, after ProtectedMode, exactly as ``**extra`` always was.
    assert [k for k in body if k not in ("Path", "Name", "Visibility", "License")] == [
        "ProtectedMode",
        "task",
        "license_name",
        "license_link",
    ]
    assert "Task" not in body
    assert "LicenseName" not in body
    assert "LicenseLink" not in body


# ---------------------------------------------------------------------------
# Endpoint availability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [404, 501])
def test_an_unpublished_endpoint_degrades_to_legacy(api, status):
    error = NotExistError("absent", status_code=status) if status == 404 else ServerError("absent", status_code=status)
    with (
        mock.patch.object(api.openapi, "_request", side_effect=error) as openapi_request,
        mock.patch.object(api.legacy, "_request", return_value=_legacy_response()) as legacy_request,
    ):
        api.create_repo("alice/thing", "model")

    assert openapi_request.called
    assert legacy_request.call_args.args == ("POST", "models")


def test_other_server_errors_are_not_swallowed(api):
    with (
        mock.patch.object(api.openapi, "_request", side_effect=ServerError("boom", status_code=500)),
        mock.patch.object(api.legacy, "_request", return_value=_legacy_response()) as legacy_request,
    ):
        with pytest.raises(ServerError):
            api.create_repo("alice/thing", "model")

    legacy_request.assert_not_called()


def test_a_duplicate_dataset_reported_as_400_becomes_already_exists(api):
    """``POST /datasets`` declares no 409, so callers would otherwise miss it."""
    rejection = InvalidParameter("dataset already exists", status_code=400)
    with mock.patch.object(api.openapi, "_request", side_effect=rejection):
        with pytest.raises(AlreadyExistsError):
            api.create_repo("alice/thing", "dataset")


def test_a_genuine_bad_request_stays_invalid_parameter(api):
    with mock.patch.object(api.openapi, "_request", side_effect=InvalidParameter("bad name", status_code=400)):
        with pytest.raises(InvalidParameter) as excinfo:
            api.create_repo("alice/thing", "dataset")
    assert not isinstance(excinfo.value, AlreadyExistsError)


# ---------------------------------------------------------------------------
# Returned metadata
# ---------------------------------------------------------------------------
def test_request_values_backfill_the_sparse_create_response(api, routes):
    """The create response carries identity only; legacy echoed the settings."""
    info = api.create_repo(
        "alice/thing",
        "model",
        visibility="private",
        license="mit",
        description="desc",
    )

    assert info.repo_id == "alice/thing"
    assert info.visibility == Visibility.PRIVATE
    assert info.license == "mit"
    assert info.description == "desc"
    assert info.gated is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"visibility": "private"}, Visibility.PRIVATE),
        ({"visibility": "public"}, Visibility.PUBLIC),
        ({"visibility": 5}, Visibility.PUBLIC),
        # Gating alone implies a visibility the caller never named.
        ({"gated": False}, Visibility.PRIVATE),
        ({"gated": "manual"}, Visibility.PRIVATE),
    ],
)
def test_the_backfilled_visibility_is_an_enum_not_a_bare_int(api, routes, kwargs, expected):
    """Assigned after ``__init__``, so it bypasses the dataclass's coercion.

    ``1 == Visibility.PRIVATE`` holds for a plain int, so this asserts on
    ``.label`` -- which only exists on the enum.
    """
    info = api.create_repo("alice/thing", "model", **kwargs)

    assert isinstance(info.visibility, Visibility)
    assert info.visibility.label == expected.label


def test_a_gated_repo_reports_private_visibility_and_its_mode(api, routes):
    info = api.create_repo("alice/thing", "model", visibility="gated", gated="manual")

    assert info.visibility == Visibility.PRIVATE
    assert info.gated is True
    assert info.gated_mode == "manual"
