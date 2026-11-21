"""Per-call concurrency, cancellation and transport ownership."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ._multipart_config import MultipartSettings
from ._multipart_protocol import MultipartError
from .constants import API_CONNECTION_POOL_MAXSIZE
from .errors import InvalidParameter

if TYPE_CHECKING:
    from ._legacy_api import LegacyClient


class UploadExecution:
    def __init__(self, settings: MultipartSettings, client: LegacyClient, workers: int | None, use_cache: bool) -> None:
        if workers is not None and (type(workers) is not int or workers < 1):
            raise InvalidParameter("Multipart upload workers must be a positive integer.")
        self.settings = settings
        self.client = client
        self.use_cache = use_cache
        self.limit = min(settings.requests, API_CONNECTION_POOL_MAXSIZE, workers or settings.requests)
        self.hash_limit = min(settings.hash_workers, workers or settings.hash_workers)
        self._credential = client.token
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._active = 0
        self._control_waiters = 0
        self._hash_slots = threading.BoundedSemaphore(self.hash_limit)
        self._queued = threading.BoundedSemaphore(2 * self.limit)
        self._futures: set[Future[Any]] = set()
        self._future_lock = threading.Lock()
        self._sessions: list[Any] = []
        self._session_lock = threading.Lock()
        self._local = threading.local()
        self._executor: ThreadPoolExecutor | None = None

    def check(self) -> None:
        if self._cancelled.is_set():
            raise MultipartError("Upload cancelled; retained state can be resumed.")
        if self.client.token != self._credential:
            self.stop()
            raise MultipartError("Upload identity changed; resume with the original authenticated identity.")

    @contextmanager
    def request_slot(self, *, control: bool = False) -> Iterator[None]:
        with self._condition:
            if control:
                self._control_waiters += 1
            try:
                while self._active >= self.limit or (not control and self._control_waiters):
                    self.check()
                    self._condition.wait(0.1)
                self.check()
                self._active += 1
            finally:
                if control:
                    self._control_waiters -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @contextmanager
    def scanner_slot(self) -> Iterator[None]:
        while not self._hash_slots.acquire(timeout=0.1):
            self.check()
        try:
            self.check()
            yield
        finally:
            self._hash_slots.release()

    def sleep(self, seconds: float) -> None:
        if self._cancelled.wait(max(0.0, seconds)):
            self.check()
        self.check()

    def session(self) -> Any:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self.client._multipart_session()
            with self._session_lock:
                self._sessions.append(session)
            self._local.session = session
        return session

    def register(self, future: Future[Any]) -> None:
        with self._future_lock:
            self._futures.add(future)
        future.add_done_callback(self._discard)
        if self._cancelled.is_set():
            future.cancel()

    def _discard(self, future: Future[Any]) -> None:
        with self._future_lock:
            self._futures.discard(future)

    def submit_part(self, function: Callable[..., Any], *args: Any) -> Future[Any]:
        while not self._queued.acquire(timeout=0.1):
            self.check()
        try:
            self.check()
            with self._future_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self.limit, thread_name_prefix="ms-part")
                executor = self._executor
            future = executor.submit(function, *args)
        except BaseException:
            self._queued.release()
            raise
        self.register(future)
        future.add_done_callback(lambda _: self._queued.release())
        return future

    def stop(self) -> None:
        with self._condition:
            self._cancelled.set()
            self._condition.notify_all()
        with self._future_lock:
            pending = list(self._futures)
        for future in pending:
            future.cancel()

    def close(self) -> None:
        self.stop()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        for session in self._sessions:
            session.close()

    def __enter__(self) -> UploadExecution:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
