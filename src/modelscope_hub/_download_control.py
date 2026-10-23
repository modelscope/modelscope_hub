"""Download cancellation, deadlines, and stream monitoring without changing uploads."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from .errors import NetworkError, RequestTimeoutError


class _DownloadCancelledError(Exception):
    """The parent download was cancelled; do not retry or report a remote failure."""


class _DownloadDeadlineError(RequestTimeoutError):
    """The total download time budget has been exhausted."""

    retryable = False


class _DownloadTooSlowError(NetworkError):
    """Download progress stayed below the minimum rate for a full observation window."""


@dataclass
class _WatchedResponse:
    sock: Any
    started: float
    received: int = 0
    error: Exception | None = None
    monitor_speed: bool = True


def _response_socket(response: requests.Response) -> Any:
    # urllib3 may release connection.sock while HTTPResponse still retains
    # SocketIO. Only shut down interrupted reads; leave socket timeouts and data unchanged.
    raw = response.raw
    connection = getattr(raw, "_connection", None)
    sock = getattr(connection, "sock", None)
    if sock is None:
        fp = getattr(getattr(raw, "_fp", None), "fp", None)
        sock = getattr(getattr(fp, "raw", None), "_sock", None)
    return sock


class _DownloadControl:
    """A per-call budget whose workers share a parent cancellation event, not global state."""

    def __init__(
        self,
        *,
        total_timeout: float,
        cancel_event: threading.Event | None = None,
        min_rate: int = 0,
        speed_window: float = 60,
    ) -> None:
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.deadline = time.monotonic() + total_timeout
        self.min_rate = min_rate
        self.speed_window = speed_window
        self._responses: dict[requests.Response, _WatchedResponse] = {}
        self._transports: dict[Any, _WatchedResponse] = {}
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._monitor: threading.Thread | None = None

    def __enter__(self) -> _DownloadControl:
        return self

    def __exit__(self, *args: Any) -> None:
        self._finished.set()
        if self._monitor is not None:
            self._monitor.join(timeout=1)
        with self._lock:
            responses = list(self._responses)
            self._responses.clear()
        for response in responses:
            response.close()

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise _DownloadCancelledError("下载已取消")
        if time.monotonic() >= self.deadline:
            raise _DownloadDeadlineError("下载总时限已耗尽；已保存可继续下载的临时文件")

    def timeouts(self, connect: float, read: float) -> tuple[float, float]:
        self.check()
        remaining = max(0.001, self.deadline - time.monotonic())
        return min(connect, remaining), min(read, remaining)

    def sleep(self, seconds: float) -> None:
        self.check()
        self.cancel_event.wait(min(max(0, seconds), max(0, self.deadline - time.monotonic())))
        self.check()

    def _start_monitor(self) -> None:
        if self._monitor is None:
            self._monitor = threading.Thread(target=self._watch, name="download-watchdog", daemon=True)
            self._monitor.start()

    def track(self, response: requests.Response) -> None:
        self.check()
        with self._lock:
            self._responses[response] = _WatchedResponse(_response_socket(response), time.monotonic())
            self._start_monitor()

    @contextlib.contextmanager
    def transport(self, sock: Any) -> Iterator[None]:
        """Interrupt sockets while reading headers so trickling headers cannot bypass deadlines."""
        self.check()
        state = _WatchedResponse(sock, time.monotonic(), monitor_speed=False)
        with self._lock:
            self._transports[sock] = state
            self._start_monitor()
        try:
            yield
        finally:
            with self._lock:
                self._transports.pop(sock, None)
            self.check()
            if state.error is not None:
                raise state.error

    def release(self, response: requests.Response) -> None:
        with self._lock:
            self._responses.pop(response, None)

    @contextlib.contextmanager
    def response(self, response: requests.Response) -> Iterator[requests.Response]:
        try:
            yield response
        finally:
            self.release(response)
            response.close()

    def _check_response(self, response: requests.Response, received: int = 0) -> None:
        self.check()
        with self._lock:
            state = self._responses.get(response)
            if state is not None:
                state.received += received
                if state.error is not None:
                    raise state.error

    def _watch(self) -> None:
        # A separate watchdog interrupts reads that keep receiving tiny chunks,
        # including with urllib3 1.26. Do not call response.close() here: the
        # reader may hold the buffered stream lock.
        while not self._finished.wait(0.1):
            now = time.monotonic()
            with self._lock:
                for state in [*self._responses.values(), *self._transports.values()]:
                    if state.error is not None:
                        continue
                    if self.cancel_event.is_set():
                        state.error = _DownloadCancelledError("下载已取消")
                    elif now >= self.deadline:
                        state.error = _DownloadDeadlineError("下载总时限已耗尽")
                    elif state.monitor_speed and self.min_rate and now - state.started >= self.speed_window:
                        rate = state.received / (now - state.started)
                        if rate < self.min_rate:
                            state.error = _DownloadTooSlowError(
                                f"下载速率 {rate:.0f} B/s 低于 {self.min_rate} B/s，观察窗口 {self.speed_window:g}s"
                            )
                        else:
                            state.started, state.received = now, 0
                    if state.error is not None and state.sock is not None:
                        with contextlib.suppress(OSError):
                            state.sock.shutdown(socket.SHUT_RDWR)

    def abort_responses(self) -> None:
        """Abort active responses without cancelling the parent, allowing single-stream fallback."""
        with self._lock:
            for state in [*self._responses.values(), *self._transports.values()]:
                state.error = _DownloadCancelledError("分片组已取消")
                if state.sock is not None:
                    with contextlib.suppress(OSError):
                        state.sock.shutdown(socket.SHUT_RDWR)

    def chunks(self, response: requests.Response, chunk_size: int) -> Iterator[bytes]:
        """Yield received bytes promptly instead of waiting for a full chunk to report progress."""
        raw = response.raw
        try:
            if hasattr(raw, "stream"):
                read1 = getattr(raw, "read1", None)
                if read1 is None and not response.headers.get("Content-Encoding"):
                    read1 = getattr(getattr(raw, "_fp", None), "read1", None)
                    decoded = False
                else:
                    decoded = True
                while True:
                    self._check_response(response)
                    if read1 is not None:
                        data = read1(chunk_size, decode_content=True) if decoded else read1(chunk_size)
                    else:
                        data = raw.read(chunk_size, decode_content=True)
                    self._check_response(response, len(data))
                    if not data:
                        break
                    yield data
            else:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    self._check_response(response, len(chunk))
                    if chunk:
                        yield chunk
        except (Urllib3HTTPError, OSError) as exc:
            self._check_response(response)
            raise requests.ConnectionError(f"下载流读取失败：{type(exc).__name__}") from exc
        finally:
            self.release(response)


_network_state = threading.local()


@contextlib.contextmanager
def _network_context(control: _DownloadControl) -> Iterator[None]:
    previous = getattr(_network_state, "control", None)
    _network_state.control = control
    try:
        yield
    finally:
        _network_state.control = previous


class _WatchedConnectionMixin:
    def _watched(self, method: str) -> Any:
        original = getattr(super(), method)
        control = getattr(_network_state, "control", None)
        if control is None:
            return original()
        with control.transport(getattr(self, "sock", None)):
            return original()

    def getresponse(self) -> Any:
        return self._watched("getresponse")

    def _tunnel(self) -> Any:
        # HTTP CONNECT tunnel headers can also arrive as an endless trickle.
        return self._watched("_tunnel")


class _ReadHTTPAdapter(HTTPAdapter):
    """Download/listing pools with instance-local connection classes, leaving urllib3 globals intact."""

    @staticmethod
    def _configure(manager: Any) -> None:
        classes = dict(manager.pool_classes_by_scheme)
        for scheme, pool in classes.items():
            if getattr(pool, "_download_controlled", False):
                continue
            connection = getattr(pool, "ConnectionCls", None)
            if connection is None:
                continue
            # Preserve the original TLS, HTTP proxy, and SOCKS connection classes and arguments.
            watched = type(f"_Download{connection.__name__}", (_WatchedConnectionMixin, connection), {})
            classes[scheme] = type(
                f"_Download{pool.__name__}",
                (pool,),
                {"ConnectionCls": watched, "_download_controlled": True},
            )
        manager.pool_classes_by_scheme = classes

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False, **kwargs: Any) -> None:
        super().init_poolmanager(connections, maxsize, block=block, **kwargs)
        self._configure(self.poolmanager)

    def proxy_manager_for(self, proxy: str, **kwargs: Any) -> Any:
        manager = super().proxy_manager_for(proxy, **kwargs)
        self._configure(manager)
        return manager
