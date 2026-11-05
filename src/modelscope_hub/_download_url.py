"""URL 下载的共享传输机制；不从任意 URL 反推 Hub 仓库或注入认证。"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from urllib3.util.retry import Retry

from ._cache_layout import contained_path, validate_relative
from ._download import ProgressCallback, TqdmCallback, _checkable_content_length, _parse_content_range
from ._download_control import (
    _DownloadCancelledError,
    _DownloadControl,
    _DownloadDeadlineError,
    _network_context,
    _ReadHTTPAdapter,
)
from .constants import DOWNLOAD_LOW_SPEED_WINDOW_SECONDS, DOWNLOAD_MIN_BYTES_PER_SECOND, DOWNLOAD_TOTAL_TIMEOUT
from .errors import FileIntegrityError, InvalidParameter
from .utils import _FALLBACK_SESSION_ID, append_user_agent, build_user_agent


def _headers(headers: dict[str, str] | None, user_agent: str | dict | None) -> dict[str, str]:
    result = {key: value for key, value in (headers or {}).items() if key.lower() != "user-agent"}
    extra = next((value for key, value in (headers or {}).items() if key.lower() == "user-agent"), None)
    base = build_user_agent(session_id=_FALLBACK_SESSION_ID, extra="invoked_by/compat")
    result["User-Agent"] = append_user_agent(append_user_agent(base, extra), user_agent)
    result.setdefault("Accept-Encoding", "identity")
    return result


def transfer_url(
    url: str,
    target: Path,
    *,
    file_size: int | None = None,
    expected_sha256: str | None = None,
    cookies: Any = None,
    headers: dict[str, str] | None = None,
    user_agent: str | dict | None = None,
    callbacks: list[type[ProgressCallback]] | None = None,
    disable_tqdm: bool = False,
    retry_times: int = 5,
    timeout: float = 60,
    chunk_size: int = 1024 * 1024,
) -> str | None:
    """续传至 target；严格验证 Range/长度，重试后保持旧 API 的 None 哈希契约。"""
    if urlparse(url).scheme not in ("http", "https") or not urlparse(url).hostname:
        raise InvalidParameter("Download URL must be HTTP(S)")
    if file_size is not None and file_size < 0:
        raise InvalidParameter("file_size must not be negative")
    if target.is_symlink():
        raise InvalidParameter("URL download target must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    progress_classes = list(callbacks or [])
    if not disable_tqdm:
        progress_classes.append(TqdmCallback)
    progress = [callback(target.name, file_size or 0) for callback in progress_classes]
    existing = target.stat().st_size if target.is_file() else 0
    retried = existing > 0
    digest = hashlib.sha256()
    retry = Retry(total=retry_times, backoff_factor=1, allowed_methods=["GET"])
    request_headers = _headers(headers, user_agent)
    announced_total = file_size
    try:
        if existing and file_size is not None and existing > file_size:
            target.write_bytes(b"")
            existing = 0
        for callback in progress:
            if existing:
                callback.update(existing)
        with (
            requests.Session() as session,
            _DownloadControl(
                total_timeout=DOWNLOAD_TOTAL_TIMEOUT,
                min_rate=DOWNLOAD_MIN_BYTES_PER_SECOND,
                speed_window=DOWNLOAD_LOW_SPEED_WINDOW_SECONDS,
                header_timeout=timeout,
                budget_label="URL download",
                budget_env="MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT",
            ) as control,
        ):
            session.mount("http://", _ReadHTTPAdapter())
            session.mount("https://", _ReadHTTPAdapter())
            if cookies is not None:
                session.cookies.update(cookies)
            while True:
                control.check()
                offset = target.stat().st_size if target.is_file() else 0
                if announced_total is not None and offset == announced_total:
                    if not target.exists():
                        target.touch()
                    break
                current = {key: value for key, value in request_headers.items() if key.lower() != "range"}
                end = str(announced_total - 1) if announced_total is not None else ""
                current["Range"] = f"bytes={offset}-{end}"
                current["X-Request-ID"] = uuid.uuid4().hex
                try:
                    with _network_context(control):
                        response = session.get(
                            url, headers=current, stream=True, timeout=control.timeouts(timeout, timeout)
                        )
                    with control.response(response):
                        response.raise_for_status()
                        content_range = _parse_content_range(response.headers.get("Content-Range"))
                        # 部分 Hub/OSS 路由以 HTTP 200 返回有效 Content-Range；与核心下载一致，
                        # 以经过校验的 range 元数据判断是否续传，不仅依赖状态码。
                        if response.status_code == 206 or (response.status_code == 200 and content_range is not None):
                            if content_range is None or content_range[0] != offset:
                                raise FileIntegrityError("Invalid Content-Range in URL download response")
                            if announced_total is not None and content_range[2] not in (None, announced_total):
                                raise FileIntegrityError("URL response total size contradicts expected size")
                            if content_range[2] is not None:
                                announced_total = content_range[2]
                        elif response.status_code == 200:
                            if offset:
                                for callback in progress:
                                    callback.update(-offset)
                                offset = 0
                                digest = hashlib.sha256()
                            length = _checkable_content_length(response)
                            if announced_total is None:
                                announced_total = length
                        else:
                            raise FileIntegrityError(f"Unexpected URL download HTTP status {response.status_code}")
                        body_length = _checkable_content_length(response)
                        received = 0
                        control.track(response)
                        with target.open("ab" if offset else "wb") as stream:
                            for chunk in control.chunks(response, chunk_size):
                                if not chunk:
                                    continue
                                stream.write(chunk)
                                received += len(chunk)
                                digest.update(chunk)
                                for callback in progress:
                                    callback.update(len(chunk))
                        if body_length is not None and received != body_length:
                            raise requests.ConnectionError("URL response body is incomplete")
                        if content_range is not None and received != content_range[1] - content_range[0] + 1:
                            raise requests.ConnectionError("URL range body length does not match Content-Range")
                        actual = target.stat().st_size
                        if announced_total is not None and actual != announced_total:
                            if actual > announced_total:
                                target.write_bytes(b"")
                                raise FileIntegrityError("URL response exceeded expected total size")
                            raise requests.ConnectionError("URL file transfer is incomplete")
                        break
                except (_DownloadCancelledError, _DownloadDeadlineError):
                    raise
                except Exception as exc:
                    if isinstance(exc, OSError) and not isinstance(exc, requests.RequestException):
                        raise
                    retried = True
                    # 不把签名 URL 放进异常日志；保留旧 MaxRetryError 的捕获契约。
                    retry = retry.increment("GET", "<URL download>", error=exc)
                    retry.sleep()
                    control.check()
        if expected_sha256 is not None:
            with target.open("rb") as stream:
                actual_hash = (
                    hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
                )
            if actual_hash is None:
                hasher = hashlib.sha256()
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(chunk_size), b""):
                        hasher.update(chunk)
                actual_hash = hasher.hexdigest()
            if actual_hash.lower() != expected_sha256.lower():
                raise FileIntegrityError("URL download SHA-256 mismatch")
        return None if retried else digest.hexdigest()
    finally:
        for callback in progress:
            callback.end()


def http_get_model_file(
    url: str,
    local_dir: str,
    file_name: str,
    file_size: int,
    cookies: Any,
    headers: dict[str, str] | None = None,
    disable_tqdm: bool = False,
    progress_callbacks: list[type[ProgressCallback]] | None = None,
    *,
    retry_times: int = 5,
    timeout: float = 60,
    chunk_size: int = 1024 * 1024,
) -> str | None:
    """旧指定长度 URL 下载入口；文件就在 local_dir/file_name，可恢复部分字节。"""
    validate_relative(file_name, "file_name")
    root = Path(local_dir).expanduser().absolute()
    target = contained_path(root, root / file_name)
    return transfer_url(
        url,
        target,
        file_size=file_size,
        cookies=cookies,
        headers=headers,
        callbacks=progress_callbacks,
        disable_tqdm=disable_tqdm,
        retry_times=retry_times,
        timeout=timeout,
        chunk_size=chunk_size,
    )


def http_get_file(
    url: str,
    local_dir: str,
    file_name: str,
    cookies: Any,
    headers: dict[str, str] | None = None,
    *,
    retry_times: int = 5,
    timeout: float = 60,
    chunk_size: int = 1024 * 1024,
) -> None:
    """旧未知长度入口；成功验证后原子替换目标，避免部分文件被误认为完成。"""
    validate_relative(file_name, "file_name")
    root = Path(local_dir).expanduser().absolute()
    target = contained_path(root, root / file_name)
    partial = target.with_suffix(target.suffix + ".incomplete")
    transfer_url(
        url, partial, cookies=cookies, headers=headers, retry_times=retry_times, timeout=timeout, chunk_size=chunk_size
    )
    os.replace(partial, target)
