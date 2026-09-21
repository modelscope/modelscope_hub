from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub import HubApi
from modelscope_hub.config import HubConfig

_SRC = str(Path(__file__).parents[1] / "src")


def _run_constants(
    *names: str,
    env: dict[str, str] | None = None,
    call: str | None = None,
) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    for name in list(child_env):
        if name.startswith(("MODELSCOPE_UPLOAD_", "UPLOAD_")):
            child_env.pop(name)
    child_env.update(env or {})
    child_env["PYTHONPATH"] = _SRC
    result = f"getattr(c, {call!r})()" if call is not None else f"{{name: getattr(c, name) for name in {names!r}}}"
    script = (
        "from dataclasses import asdict, is_dataclass; import json; "
        "import modelscope_hub.constants as c; "
        f"print(json.dumps({result}, "
        "default=lambda value: asdict(value) if is_dataclass(value) else sorted(value)))"
    )
    return subprocess.run(
        [sys.executable, "-W", "always::FutureWarning", "-c", script],
        check=True,
        capture_output=True,
        env=child_env,
        text=True,
    )


def test_upload_timeout_defaults_are_explicit() -> None:
    result = _run_constants(
        "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
        "UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
    )
    values = json.loads(result.stdout)
    assert values == {
        "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS": 30,
        "UPLOAD_BLOB_READ_TIMEOUT_SECONDS": 3600,
    }


def test_canonical_timeout_env_wins_over_deprecated_aliases() -> None:
    result = _run_constants(
        "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
        "UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
        env={
            "MODELSCOPE_UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS": "11",
            "MODELSCOPE_UPLOAD_BLOB_READ_TIMEOUT_SECONDS": "22",
            "MODELSCOPE_UPLOAD_CONNECT_TIMEOUT": "33",
            "UPLOAD_BLOB_TIMEOUT_SECONDS": "44",
        },
    )
    values = json.loads(result.stdout)
    assert values == {
        "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS": 11,
        "UPLOAD_BLOB_READ_TIMEOUT_SECONDS": 22,
    }
    assert "FutureWarning" not in result.stderr


def test_legacy_timeout_seconds_only_controls_read_timeout() -> None:
    result = _run_constants(
        "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
        "UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
        env={"UPLOAD_BLOB_TIMEOUT_SECONDS": "123"},
    )
    values = json.loads(result.stdout)
    assert values == {
        "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS": 30,
        "UPLOAD_BLOB_READ_TIMEOUT_SECONDS": 123,
    }
    assert "UPLOAD_BLOB_TIMEOUT_SECONDS" in result.stderr
    assert "FutureWarning" in result.stderr
    assert "will be removed in a future version" in result.stderr


def test_upload_ignore_file_pattern_env_migration() -> None:
    legacy = _run_constants(
        env={"UPLOAD_IGNORE_FILE_PATTERN": "*.legacy"},
        call="get_upload_ignore_file_pattern",
    )
    assert json.loads(legacy.stdout) == "*.legacy"
    assert "UPLOAD_IGNORE_FILE_PATTERN" in legacy.stderr
    assert "will be removed in a future version" in legacy.stderr

    canonical = _run_constants(
        env={
            "MODELSCOPE_UPLOAD_IGNORE_FILE_PATTERN": "*.canonical",
            "UPLOAD_IGNORE_FILE_PATTERN": "*.legacy",
        },
        call="get_upload_ignore_file_pattern",
    )
    assert json.loads(canonical.stdout) == "*.canonical"
    assert "FutureWarning" not in canonical.stderr


def test_size_units_and_http_method_normalization() -> None:
    result = _run_constants(
        "UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES",
        "UPLOAD_MAX_FILE_SIZE_BYTES",
        "UPLOAD_HTTP_RETRY_ALLOWED_METHODS",
        env={
            "MODELSCOPE_UPLOAD_BLOB_PROGRESS_THRESHOLD_MB": "7",
            "UPLOAD_MAX_FILE_SIZE_MB": "12",
            "MODELSCOPE_UPLOAD_HTTP_RETRY_ALLOWED_METHODS": " get, post, ,HEAD ",
        },
    )
    values = json.loads(result.stdout)
    assert values["UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES"] == 7 * 1024 * 1024
    assert values["UPLOAD_MAX_FILE_SIZE_BYTES"] == 12 * 1024 * 1024
    assert values["UPLOAD_HTTP_RETRY_ALLOWED_METHODS"] == ["GET", "HEAD", "POST"]
    assert "UPLOAD_MAX_FILE_SIZE_MB" in result.stderr


@pytest.mark.parametrize(
    ("legacy_env", "constant", "raw", "expected"),
    [
        (
            "MODELSCOPE_UPLOAD_CONNECT_TIMEOUT",
            "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
            "15",
            15,
        ),
        (
            "UPLOAD_BLOB_CONNECT_TIMEOUT",
            "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
            "16",
            16,
        ),
        (
            "MODELSCOPE_UPLOAD_READ_TIMEOUT",
            "UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
            "17",
            17,
        ),
        (
            "UPLOAD_BLOB_READ_TIMEOUT",
            "UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
            "18",
            18,
        ),
        ("UPLOAD_BLOB_MAX_RETRIES", "UPLOAD_BLOB_MAX_ATTEMPTS", "7", 7),
        ("UPLOAD_BLOB_RETRY_BACKOFF", "UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS", "3", 3),
        ("UPLOAD_BLOB_RETRY_MAX_WAIT", "UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS", "9", 9),
        ("UPLOAD_COMMIT_BATCH_SIZE", "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", "12", 12),
        (
            "UPLOAD_VALIDATE_BLOB_BATCH_SIZE",
            "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS",
            "13",
            13,
        ),
        ("UPLOAD_ADAPTIVE_BATCH_SIZE", "UPLOAD_ADAPTIVE_BATCHING_ENABLED", "false", False),
        ("UPLOAD_COMMIT_MAX_RETRIES", "UPLOAD_COMMIT_MAX_ATTEMPTS", "4", 4),
        (
            "MODELSCOPE_UPLOAD_COMMIT_MAX_TOTAL_WAIT",
            "UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS",
            "21",
            21,
        ),
        (
            "MODELSCOPE_UPLOAD_BATCH_CONSECUTIVE_FAILURE_LIMIT",
            "UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES",
            "6",
            6,
        ),
        (
            "UPLOAD_FAILED_FILE_MAX_RETRIES",
            "UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS",
            "8",
            8,
        ),
        ("UPLOAD_REACT_ENABLED", "UPLOAD_RECOVERY_ENABLED", "false", False),
        (
            "UPLOAD_REACT_ROUND2_BASE_DELAY",
            "UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS",
            "10",
            10,
        ),
        (
            "UPLOAD_REACT_ROUND3_FILE_DELAY",
            "UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS",
            "11",
            11,
        ),
        (
            "UPLOAD_REACT_BACKOFF_MAX_EXPONENT",
            "UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT",
            "12",
            12,
        ),
        ("UPLOAD_REACT_MAX_DELAY", "UPLOAD_RECOVERY_MAX_DELAY_SECONDS", "14", 14),
        ("DEFAULT_MAX_WORKERS", "UPLOAD_MAX_CONCURRENT_WORKERS", "2", 2),
        (
            "MODELSCOPE_UPLOAD_MAX_WORKERS",
            "UPLOAD_MAX_CONCURRENT_WORKERS",
            "3",
            3,
        ),
        ("UPLOAD_USE_CACHE", "UPLOAD_CACHE_ENABLED", "false", False),
        ("MODELSCOPE_UPLOAD_CACHE", "UPLOAD_CACHE_ENABLED", "false", False),
        (
            "UPLOAD_LFS_ENFORCE_THRESHOLD",
            "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
            "1233",
            1233,
        ),
        (
            "UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS",
            "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
            "1234",
            1234,
        ),
        ("UPLOAD_MAX_FILE_COUNT", "UPLOAD_MAX_FILE_COUNT", "42", 42),
        (
            "UPLOAD_MAX_FILE_SIZE",
            "UPLOAD_MAX_FILE_SIZE_BYTES",
            "9876",
            9876,
        ),
        (
            "UPLOAD_MAX_FILE_COUNT_IN_DIR",
            "UPLOAD_MAX_FILES_PER_DIRECTORY",
            "43",
            43,
        ),
        (
            "UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT",
            "UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES",
            "4321",
            4321,
        ),
    ],
)
def test_deprecated_upload_env_aliases(
    legacy_env: str,
    constant: str,
    raw: str,
    expected: int | bool,
) -> None:
    result = _run_constants(constant, env={legacy_env: raw})
    assert json.loads(result.stdout)[constant] == expected
    assert legacy_env in result.stderr
    assert "FutureWarning" in result.stderr
    assert "will be removed in a future version" in result.stderr


def test_deprecated_http_retry_methods_are_normalized() -> None:
    result = _run_constants(
        "UPLOAD_HTTP_RETRY_ALLOWED_METHODS",
        env={"UPLOAD_RETRY_ALLOWED_METHODS": " get, patch,HEAD "},
    )
    assert json.loads(result.stdout)["UPLOAD_HTTP_RETRY_ALLOWED_METHODS"] == [
        "GET",
        "HEAD",
        "PATCH",
    ]
    assert "UPLOAD_RETRY_ALLOWED_METHODS" in result.stderr
    assert "will be removed in a future version" in result.stderr


def test_dead_lfs_threshold_is_not_registered() -> None:
    result = _run_constants("ENV_REGISTRY")
    names = {item["name"] for item in json.loads(result.stdout)["ENV_REGISTRY"]}
    assert "UPLOAD_LFS_THRESHOLD" not in names


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("32KiB", 32 * 1024),
        ("512K", 512 * 1024),
        ("65536", 65536),
        ("1MiB", 1024 * 1024),
        ("2MB", 2 * 1000 * 1000),
        ("0", 0),
    ],
)
def test_lfs_threshold_accepts_byte_sizes_and_unit_suffixes(raw: str, expected: int) -> None:
    # A megabyte-only knob cannot express the sub-MB range that decides whether
    # small files ride inline in a commit or go to object storage.
    result = _run_constants(
        "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
        env={"MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD": raw},
    )
    assert json.loads(result.stdout)["UPLOAD_LFS_FORCE_THRESHOLD_BYTES"] == expected
    assert "FutureWarning" not in result.stderr


def test_deprecated_lfs_threshold_mb_alias_keeps_megabyte_units() -> None:
    result = _run_constants(
        "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
        env={"MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD_MB": "4"},
    )
    assert json.loads(result.stdout)["UPLOAD_LFS_FORCE_THRESHOLD_BYTES"] == 4 * 1024 * 1024
    assert "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD_MB" in result.stderr
    assert "expects bytes" in result.stderr


def test_canonical_lfs_threshold_wins_over_deprecated_aliases() -> None:
    result = _run_constants(
        "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
        env={
            "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD": "32KiB",
            "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD_MB": "4",
            "UPLOAD_LFS_ENFORCE_THRESHOLD": "999",
        },
    )
    assert json.loads(result.stdout)["UPLOAD_LFS_FORCE_THRESHOLD_BYTES"] == 32 * 1024
    assert "FutureWarning" not in result.stderr


@pytest.mark.parametrize(
    ("env", "constant", "expected", "needle"),
    [
        (
            {"MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD": "32ZB"},
            "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
            64 * 1024,
            "unknown size unit",
        ),
        (
            {"MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD": "-1"},
            "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
            64 * 1024,
            "must not be negative",
        ),
        (
            {"MODELSCOPE_UPLOAD_COMMIT_BATCH_MAX_OPERATIONS": "lots"},
            "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS",
            256,
            "not an integer",
        ),
        (
            {"MODELSCOPE_UPLOAD_COMMIT_BATCH_MAX_OPERATIONS": "0"},
            "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS",
            256,
            "must be a positive integer",
        ),
        (
            {"MODELSCOPE_UPLOAD_COMMIT_MAX_INLINE_BYTES": "8bogus"},
            "UPLOAD_COMMIT_MAX_INLINE_BYTES",
            8 * 1024 * 1024,
            "unknown size unit",
        ),
    ],
)
def test_invalid_upload_env_values_warn_instead_of_silently_reverting(
    env: dict[str, str],
    constant: str,
    expected: int,
    needle: str,
) -> None:
    # Silently falling back made misconfiguration invisible: the value had no
    # effect and nothing said so.
    result = _run_constants(constant, env=env)
    assert json.loads(result.stdout)[constant] == expected
    assert needle in result.stderr
    assert "is invalid" in result.stderr


def test_commit_rate_governor_accepts_explicit_zero_without_warning() -> None:
    result = _run_constants(
        "UPLOAD_COMMIT_MAX_PER_HOUR",
        env={"MODELSCOPE_UPLOAD_COMMIT_MAX_PER_HOUR": "0"},
    )
    assert json.loads(result.stdout)["UPLOAD_COMMIT_MAX_PER_HOUR"] == 0
    assert "is invalid" not in result.stderr


def test_new_upload_knobs_are_registered_under_upload() -> None:
    result = _run_constants("ENV_REGISTRY")
    registry = {item["name"]: item for item in json.loads(result.stdout)["ENV_REGISTRY"]}
    for name in (
        "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD",
        "MODELSCOPE_UPLOAD_COMMIT_MAX_INLINE_BYTES",
        "MODELSCOPE_UPLOAD_COMMIT_MAX_PER_HOUR",
        "MODELSCOPE_UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS",
        "MODELSCOPE_UPLOAD_INLINE_METADATA_PATHS",
    ):
        assert registry[name]["category"] == "Upload"
    assert registry["MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD"]["default"] == "64KiB"
    assert "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD_MB" not in registry


def test_http_sessions_size_the_pool_for_concurrent_workers() -> None:
    # urllib3 defaults the pool to 10. A folder upload runs max_workers requests
    # at once, so the default discarded the excess connections and paid for a
    # fresh TLS handshake on their next use.
    from modelscope_hub.constants import API_CONNECTION_POOL_MAXSIZE

    api = HubApi(config=HubConfig(token="ms-test"))
    for session in (api.legacy._session, api.openapi._session):
        adapter = session.get_adapter("https://modelscope.cn")
        pool_kw = adapter.poolmanager.connection_pool_kw
        assert pool_kw["maxsize"] == API_CONNECTION_POOL_MAXSIZE
        assert API_CONNECTION_POOL_MAXSIZE >= 16


def test_connection_pool_size_is_env_tunable() -> None:
    result = _run_constants(
        "API_CONNECTION_POOL_MAXSIZE",
        env={"MODELSCOPE_API_CONNECTION_POOL_MAXSIZE": "64"},
    )
    assert json.loads(result.stdout)["API_CONNECTION_POOL_MAXSIZE"] == 64


def _response(data: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.request = None
    response.url = "https://modelscope.cn/test"
    response.json.return_value = {"Data": data or {}}
    return response


def test_hubapi_blob_put_receives_explicit_connect_and_read_timeouts() -> None:
    config = HubConfig(token="ms-test")
    api = HubApi(config=config)
    api.uploader._create_repo_fn = None
    digest = hashlib.sha256(b"weights").hexdigest()

    def request(method: str, path: str, **kwargs):
        if path.endswith("info/lfs/objects/batch"):
            return _response(
                {
                    "objects": [
                        {
                            "oid": digest,
                            "actions": {"upload": {"href": "https://storage/upload"}},
                        }
                    ]
                }
            )
        return _response()

    put_response = _response()
    put_response.json.side_effect = ValueError

    def consume_upload(*args, **kwargs):
        data = kwargs["data"]
        while data.read(1024):
            pass
        return put_response

    with (
        patch.object(api.legacy, "_request", side_effect=request),
        patch.object(api.legacy._session, "put", side_effect=consume_upload) as put,
    ):
        api.upload_file(
            "owner/repo",
            "model",
            b"weights",
            "model.bin",
            disable_tqdm=True,
        )

    put.assert_called_once()
    assert put.call_args.kwargs["timeout"] == (30, 3600)
