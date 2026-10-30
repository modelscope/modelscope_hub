"""Download cancellation, deadlines, and stream monitoring without changing uploads."""

from __future__ import annotations

import contextlib
import math
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from .errors import NetworkError, RequestTimeoutError

if TYPE_CHECKING:
    from .types import DownloadRetryPolicy


class _RecoveryBudget:
    """One durable recovery budget per file, shared by range and integrity retries."""

    def __init__(
        self,
        policy: DownloadRetryPolicy,
        retries: int,
        seconds: float,
        persist: Callable[[int, float], None],
    ) -> None:
        self.policy = policy
        self.retries = retries
        self.seconds = seconds
        self.persist = persist
        self.lock = threading.Lock()
        self.activity = threading.local()

    def progress(self) -> None:
        self.activity.last = time.monotonic()

    def retry(self, delay: float) -> None:
        with self.lock:
            elapsed = max(0, time.monotonic() - getattr(self.activity, "last", time.monotonic()))
            self.seconds += elapsed
            self.persist(self.retries, self.seconds)
            if delay > self.policy.max_retry_delay:
                raise _DownloadDeferredError("Server cooldown exceeds the short recovery budget")
            if self.retries >= self.policy.max_retries or self.seconds + delay >= self.policy.recovery_timeout:
                raise _DownloadDeferredError("File recovery budget exhausted; automatic retry is disabled")
            self.retries += 1
            self.seconds += delay
            self.persist(self.retries, self.seconds)
            self.progress()


class _DownloadDeferredError(NetworkError):
    """Recovery is exhausted or a server cooldown requires a later explicit run."""

    retryable = False


class _DownloadCancelledError(Exception):
    """The parent download was cancelled; do not retry or report a remote failure."""


class _DownloadDeadlineError(RequestTimeoutError):
    """A configured time budget has been exhausted.

    Stays non-retryable so the current run never loops on an expired budget.
    Callers that track per-file state treat it as a resumable stop rather than
    a rejected file, because the partial file on disk is still valid.
    """

    retryable = False
    suggestion = (
        "A time budget expired rather than the transfer failing. Partial files are kept, so re-running "
        "resumes them; raise MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT or set it to 0 to remove the per-file limit."
    )


class _DownloadTooSlowError(NetworkError):
    """Download progress stayed below the minimum rate for a full observation window."""


@dataclass
class _WatchedResponse:
    sock: Any
    started: float
    received: int = 0
    error: Exception | None = None
    monitor_speed: bool = True
    header_deadline: float | None = None
    header_limit: float = 0


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
        header_timeout: float = 0,
        recovery: _RecoveryBudget | None = None,
        budget_label: str = "Download",
        budget_env: str = "",
    ) -> None:
        self.cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.total_timeout = total_timeout
        # A non-positive budget means "no overall limit": idle timeouts, the
        # low-speed watchdog, bounded retries and cancellation still apply, so
        # an infinite deadline removes only the wall-clock cap on healthy work.
        self.deadline = math.inf if total_timeout <= 0 else time.monotonic() + total_timeout
        self.min_rate = min_rate
        self.speed_window = speed_window
        self.header_timeout = header_timeout
        self.recovery = recovery
        self.budget_label = budget_label
        self.budget_env = budget_env
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

    def _deadline_error(self) -> _DownloadDeadlineError:
        hint = f"; raise {self.budget_env} or set it to 0 to disable the limit" if self.budget_env else ""
        return _DownloadDeadlineError(f"{self.budget_label} time budget of {self.total_timeout:g}s is exhausted{hint}")

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise _DownloadCancelledError("Download cancelled")
        if time.monotonic() >= self.deadline:
            raise self._deadline_error()

    def timeouts(self, connect: float, read: float) -> tuple[float, float]:
        self.check()
        remaining = max(0.001, self.deadline - time.monotonic())
        if self.recovery is not None:
            self.recovery.progress()
            with self.recovery.lock:
                remaining = min(remaining, max(0.001, self.recovery.policy.recovery_timeout - self.recovery.seconds))
            connect = min(connect, self.recovery.policy.idle_timeout)
            read = min(read, self.recovery.policy.idle_timeout)
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
        """Interrupt sockets while reading headers so trickling headers cannot bypass deadlines.

        ``requests`` cannot apply its read timeout before the status line
        arrives, so a server that emits one header byte at a time never trips
        an idle timeout. The header budget is therefore the only bound on that
        fault when no overall deadline is configured.
        """
        self.check()
        state = _WatchedResponse(sock, time.monotonic(), monitor_speed=False)
        if self.recovery is not None:
            started = getattr(self.recovery.activity, "last", time.monotonic())
            state.header_limit = self.recovery.policy.idle_timeout
            state.header_deadline = min(self.deadline, started + state.header_limit)
        elif self.header_timeout > 0:
            state.header_limit = self.header_timeout
            state.header_deadline = min(self.deadline, state.started + state.header_limit)
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
                if received and self.recovery is not None:
                    self.recovery.progress()
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
                        state.error = _DownloadCancelledError("Download cancelled")
                    elif now >= self.deadline:
                        state.error = self._deadline_error()
                    elif state.header_deadline is not None and now >= state.header_deadline:
                        state.error = RequestTimeoutError(
                            f"No response headers within the {state.header_limit:g}s idle time limit"
                        )
                    elif state.monitor_speed and self.min_rate and now - state.started >= self.speed_window:
                        rate = state.received / (now - state.started)
                        if rate < self.min_rate:
                            state.error = _DownloadTooSlowError(
                                f"Download rate {rate:.0f} B/s stayed below {self.min_rate} B/s "
                                f"over a {self.speed_window:g}s window"
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
                state.error = _DownloadCancelledError("Range part group cancelled")
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
            raise requests.ConnectionError(f"Download stream read failed: {type(exc).__name__}") from exc
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

    def build_response(self, req: Any, resp: Any) -> requests.Response:
        response = super().build_response(req, resp)
        control = getattr(_network_state, "control", None)
        if control is not None:
            # Requests consumes intermediate redirect bodies outside chunks().
            # Track them here, and release before their connection can be reused.
            original_close = response.close

            def close() -> None:
                control.release(response)
                original_close()

            response.close = close  # type: ignore[method-assign]
            try:
                control.track(response)
            except BaseException:
                response.close()
                raise
        return response

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False, **kwargs: Any) -> None:
        super().init_poolmanager(connections, maxsize, block=block, **kwargs)
        self._configure(self.poolmanager)

    def proxy_manager_for(self, proxy: str, **kwargs: Any) -> Any:
        manager = super().proxy_manager_for(proxy, **kwargs)
        self._configure(manager)
        return manager
