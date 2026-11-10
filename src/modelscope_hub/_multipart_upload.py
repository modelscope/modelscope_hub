"""Bounded resumable multipart upload state machine."""

from __future__ import annotations

import random
import threading
from bisect import bisect_left
from concurrent.futures import Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._multipart_config import action_url
from ._multipart_control import UploadExecution
from ._multipart_protocol import MultipartError, MultipartFaultError, MultipartPlan, parse_response, terminal
from ._multipart_source import HashManifest, RangeReader, UploadSource
from ._multipart_state import StateStore, UploadState
from .constants import (
    API_CONNECT_TIMEOUT,
    API_TIMEOUT,
    UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS,
    UPLOAD_BLOB_MAX_ATTEMPTS,
    UPLOAD_BLOB_READ_TIMEOUT_SECONDS,
    UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS,
    UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS,
)
from .errors import HubError, NetworkError, RequestTimeoutError


@dataclass(frozen=True)
class BlobTransferResult:
    ready: bool
    reused: bool
    uploaded_bytes: int
    bytes_read: int
    part_requests: int
    completed_via_status: bool


class MultipartUploader:
    """Own all recovery for one blob; outer upload retry loops must not replay it."""

    def __init__(
        self,
        execution: UploadExecution,
        *,
        home: Path,
        repo_id: str,
        repo_type: str,
        sha256: str,
        size: int,
        data: str | Path | bytes,
    ) -> None:
        self.execution = execution
        self.settings = execution.settings
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.sha256 = sha256
        self.size = size
        self.source = UploadSource(data, size, sha256, self.settings.chunk_size)
        self.store = StateStore(
            home,
            execution.client.endpoint,
            execution.client.token,
            repo_id,
            repo_type,
            sha256,
            size,
            execution.use_cache,
            self.settings.lock_timeout,
            self.settings.response_bytes,
        )
        self.state = UploadState("")
        self._plan = MultipartPlan("", sha256, size, 0, 1, 1, frozenset())
        self.manifest: HashManifest | None = None
        self.uploaded_bytes = 0
        self.bytes_read = 0
        self.part_requests = 0
        self._confirmed_ranges: list[tuple[int, int]] = []
        self._metrics_lock = threading.Lock()
        self._part_limit = self.settings.part_workers
        self._allow_init = True
        self._attempts: dict[tuple[int, int], int] = {}
        self._initializations = 0
        self._token_refreshes = 0
        self._rounds = 0
        self._actions = 0

    def _pause(self, attempt: int, fault: MultipartFaultError | None = None) -> None:
        requested = fault.retry_after if fault is not None else None
        if requested is not None and requested > self.settings.retry_after:
            raise MultipartError("Server Retry-After exceeds the configured multipart wait limit.") from fault
        delay = (
            requested
            if requested is not None
            else min(
                UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS ** max(0, attempt - 1),
                UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS,
            )
        )
        self.execution.sleep(delay + random.random() * 0.1)

    def _send(
        self,
        action: str,
        token: str,
        *,
        number: int | None = None,
        manifest: HashManifest | None = None,
    ) -> MultipartPlan | None:
        self.execution.check()
        params = {"part_size": self.settings.part_size} if action == "init" and self.settings.part_size else {}
        if number is not None:
            params["part_number"] = number
        url = action_url(self.state.url, action, **params)
        headers = {"Content-Length": "0"}
        data: Any = b""
        timeout = (API_CONNECT_TIMEOUT, API_TIMEOUT)
        reader: RangeReader | None = None
        part: tuple[int, HashManifest] | None = None
        if action == "part":
            if number is None or manifest is None:
                raise MultipartError("Multipart part request is missing its source manifest.")
            part = (number, manifest)
            length = self._plan.length(number)
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(length),
                "X-Part-SHA256": manifest.digest(number).hex(),
                "X-Oss-Hash-Ctx": manifest.header(number),
            }
            timeout = (UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS, UPLOAD_BLOB_READ_TIMEOUT_SECONDS)
        if token:
            headers["X-Upload-Token"] = token
        with self.execution.request_slot(control=action != "part"):
            self.execution.check()
            try:
                if part is not None:
                    number, manifest = part
                    reader = self.source.reader(
                        (number - 1) * self._plan.part_size,
                        self._plan.length(number),
                        manifest.digest(number),
                    )
                    data = reader
                    with self._metrics_lock:
                        self.part_requests += 1
                response = self.execution.client._multipart_request(
                    self.execution.session(),
                    url,
                    headers=headers,
                    data=data,
                    timeout=timeout,
                    max_response_bytes=self.settings.response_bytes,
                )
                if reader is not None:
                    reader.verify()
                result = parse_response(response, action, self.sha256, self.size)
                if action == "part" and (
                    result is None
                    or result.token != token
                    or result.part_size != self._plan.part_size
                    or result.total_parts != self._plan.total_parts
                ):
                    raise MultipartError("Multipart part response changed the active upload plan.")
                return result
            finally:
                if reader is not None:
                    with self._metrics_lock:
                        self.bytes_read += reader.bytes_read
                    reader.close()

    def _control(self, action: str, token: str) -> MultipartPlan | None:
        last_error: BaseException | None = None
        for attempt in range(1, UPLOAD_BLOB_MAX_ATTEMPTS + 1):
            try:
                return self._send(action, token)
            except MultipartFaultError as fault:
                retry = (
                    fault.name == "CAPACITY_LIMIT"
                    or (action == "status" and fault.name == "UPSTREAM_FAILURE")
                    or (action == "complete" and fault.name == "SOURCE_CLEANUP_FAILED")
                )
                if not retry:
                    raise
                if attempt == UPLOAD_BLOB_MAX_ATTEMPTS:
                    raise MultipartError(f"Multipart {action} recovery attempts were exhausted.") from fault
                last_error = fault
                self._pause(attempt, fault)
            except (NetworkError, RequestTimeoutError) as error:
                if action == "complete":
                    raise
                if attempt == UPLOAD_BLOB_MAX_ATTEMPTS:
                    raise terminal(error) from error
                last_error = error
                self._pause(attempt)
        raise MultipartError("Multipart control request failed.") from last_error

    def _adopt(self, plan: MultipartPlan, *, explicit_init: bool = False) -> None:
        old_part_size = self.state.part_size
        old_dirty = list(self.state.dirty)
        changed = self.state.token != plan.token or old_part_size != plan.part_size
        if old_part_size != plan.part_size and old_dirty:
            if old_part_size <= 0:
                dirty = set(range(1, plan.total_parts + 1))
            else:
                dirty = set()
                for number in old_dirty:
                    start = (number - 1) * old_part_size
                    end = min(self.size, start + old_part_size)
                    first = start // plan.part_size + 1
                    last = (end - 1) // plan.part_size + 1
                    dirty.update(range(first, last + 1))
            self.state.dirty = sorted(dirty)
            changed = True
        if explicit_init:
            self.state.generation += 1
            self.state.dirty = []
            changed = True
        self.state.token = plan.token
        self.state.part_size = plan.part_size
        self.state.last_action = "init" if explicit_init else "status"
        self.state.ready = False
        self.manifest = None if old_part_size != plan.part_size or explicit_init else self.manifest
        self._plan = plan
        if changed or explicit_init:
            self.store.save(self.state)

    def _query(self, token: str) -> MultipartPlan | None:
        plan = self._control("status", token)
        if plan is not None:
            self._adopt(plan)
        return plan

    def _initialize(self) -> MultipartPlan:
        if self._initializations >= 1 + self.settings.reinitializations:
            raise MultipartError("Multipart reinitialization budget was exhausted.")
        self._initializations += 1
        self.state.last_action = "init"
        self.store.save(self.state)
        for attempt in range(1, UPLOAD_BLOB_MAX_ATTEMPTS + 1):
            try:
                plan = self._send("init", "")
                if plan is None:
                    raise MultipartError("Multipart init did not return an upload plan.")
                # Persist an admitted init's token before observing cancellation.
                self._adopt(plan, explicit_init=True)
                return plan
            except MultipartFaultError as fault:
                if fault.name != "CAPACITY_LIMIT" or attempt == UPLOAD_BLOB_MAX_ATTEMPTS:
                    raise
                self._pause(attempt, fault)
        raise MultipartError("Multipart init failed.")

    def _save_dirty(self, numbers: set[int]) -> None:
        dirty = set(self.state.dirty) | numbers
        if any(not 1 <= number <= self._plan.total_parts for number in dirty):
            raise MultipartError("Server returned an invalid dirty part number.")
        self.state.dirty = sorted(dirty)
        self.store.save(self.state)

    def _upload_part(self, number: int, manifest: HashManifest, token: str) -> None:
        key = (self.state.generation, number)
        while True:
            attempt = self._attempts.get(key, 0) + 1
            if attempt > UPLOAD_BLOB_MAX_ATTEMPTS:
                raise MultipartError(f"Multipart part {number} exhausted its request budget.")
            self._attempts[key] = attempt
            try:
                self._send("part", token, number=number, manifest=manifest)
                return
            except MultipartFaultError as fault:
                if fault.name != "CAPACITY_LIMIT" or attempt == UPLOAD_BLOB_MAX_ATTEMPTS:
                    raise
                with self._metrics_lock:
                    self._part_limit = max(1, self._part_limit // 2)
                self._pause(attempt, fault)

    def _confirm(self, plan: MultipartPlan, number: int) -> None:
        start = (number - 1) * plan.part_size
        end = start + plan.length(number)
        left = bisect_left(self._confirmed_ranges, (start, -1))
        if left and self._confirmed_ranges[left - 1][1] >= start:
            left -= 1
        right = left
        covered = 0
        merged_start, merged_end = start, end
        while right < len(self._confirmed_ranges) and self._confirmed_ranges[right][0] <= end:
            old_start, old_end = self._confirmed_ranges[right]
            covered += max(0, min(end, old_end) - max(start, old_start))
            merged_start, merged_end = min(merged_start, old_start), max(merged_end, old_end)
            right += 1
        self.uploaded_bytes += end - start - covered
        self._confirmed_ranges[left:right] = [(merged_start, merged_end)]

    def _upload_parts(self, work: set[int], manifest: HashManifest) -> None:
        plan = self._plan
        token = plan.token
        generation = self.state.generation
        remaining = iter(sorted(work))
        futures: dict[Future[Any], int] = {}
        first_error: BaseException | None = None

        def fill() -> None:
            with self._metrics_lock:
                part_limit = self._part_limit
            parallelism = min(plan.parallelism, part_limit, self.execution.limit, len(work))
            while len(futures) < parallelism and first_error is None:
                try:
                    number = next(remaining)
                except StopIteration:
                    return
                future = self.execution.submit_part(self._upload_part, number, manifest, token)
                futures[future] = number

        def collect(future: Future[Any], number: int) -> None:
            nonlocal first_error
            try:
                future.result()
                if generation == self.state.generation and token == self._plan.token:
                    self._confirm(plan, number)
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    for pending in futures:
                        pending.cancel()

        try:
            fill()
            while futures:
                done, _ = wait(futures, timeout=0.1)
                if not done:
                    self.execution.check()
                    continue
                for future in done:
                    number = futures.pop(future)
                    collect(future, number)
                fill()
        finally:
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)
                for future, number in list(futures.items()):
                    futures.pop(future)
                    collect(future, number)
        if first_error is not None:
            raise first_error

    def _recovery_budget(self) -> None:
        self._rounds += 1
        self._actions += 1
        if self._rounds > self.settings.recovery_rounds or self._actions > self.settings.recovery_actions:
            raise MultipartError("Multipart recovery budget was exhausted; rerun to resume manually.")

    def _recover(self, fault: MultipartFaultError) -> str:
        name = fault.name
        if name == "UPLOAD_NOT_FOUND" and self._initializations == 0:
            if not self._allow_init:
                raise MultipartError("The pending completion is gone but batch reported an existing blob.") from fault
            return "init"
        self._recovery_budget()
        if name == "UPLOAD_NOT_FOUND":
            if not self._allow_init:
                raise MultipartError("A completion-only resume cannot initialize a replacement upload.") from fault
            return "init"
        if name in {"TOKEN_INVALID", "TOKEN_REFRESH_REQUIRED"}:
            self._token_refreshes += 1
            if self._token_refreshes > self.settings.token_refreshes:
                raise MultipartError("Multipart token refresh budget was exhausted.") from fault
            return "refresh"
        if name in {"UPSTREAM_FAILURE", "UPLOAD_CONFLICT", "PARTS_MISSING", "PART_SIZE_CONFLICT"}:
            return "status"
        if name == "PART_INTEGRITY":
            if type(fault.part_number) is not int:
                raise MultipartError("Part integrity failure did not identify a part to overwrite.") from fault
            self._save_dirty({fault.part_number})
            return "plan"
        if name == "HASH_CONTEXT_MISMATCH":
            number = fault.part_number
            if type(number) is int and 1 <= number <= self._plan.total_parts:
                dirty = set(range(number, self._plan.total_parts + 1))
            else:
                dirty = set(range(1, self._plan.total_parts + 1))
            self._save_dirty(dirty)
            return "plan"
        if name in {"FILE_SHA256_MISMATCH", "UPLOAD_SESSION_GONE", "COMPLETION_UNVERIFIED"}:
            if not self._allow_init:
                raise MultipartError("A completion-only resume cannot initialize a replacement upload.") from fault
            self.source.verify_content(self.execution)
            return "init"
        if name == "SOURCE_CLEANUP_FAILED":
            return "complete"
        raise fault

    def _ready(self, *, via_status: bool) -> BlobTransferResult:
        self.state.ready = True
        self.state.dirty = []
        self.state.last_action = "complete"
        self.store.save(self.state)
        return BlobTransferResult(
            True,
            self.bytes_read == 0,
            self.uploaded_bytes,
            self.bytes_read,
            self.part_requests,
            via_status,
        )

    def run(self, upload_url: str | None) -> BlobTransferResult:
        self._allow_init = upload_url is not None
        pending = self.store.pending() if upload_url is None else None
        if upload_url is None and pending is None:
            return BlobTransferResult(True, True, 0, 0, 0, True)
        target = upload_url or (pending.url if pending is not None else "")
        self.settings.validate_target(target, self.repo_id, self.repo_type, self.sha256, self.size)
        with self.store.lease(target, self.execution.check) as saved:
            if saved is not None and saved.url != target:
                raise MultipartError("Multipart recovery target does not match its local binding.")
            self.state = saved or UploadState(target)
            self.state.url = target
            stage = "status"
            try:
                while True:
                    self.execution.check()
                    try:
                        if stage == "status":
                            plan = self._query(self.state.token)
                            if plan is None:
                                return self._ready(via_status=True)
                            stage = "plan"
                        elif stage == "refresh":
                            plan = self._query("")
                            if plan is None:
                                return self._ready(via_status=True)
                            stage = "plan"
                        elif stage == "init":
                            self._initialize()
                            stage = "plan"
                        elif stage == "complete":
                            self.source.verify()
                            self.state.last_action = "complete"
                            self.store.save(self.state)
                            if self._control("complete", self.state.token) is None:
                                return self._ready(via_status=False)
                            raise MultipartError("Multipart complete returned an unexpected upload plan.")
                        else:
                            missing = set(range(1, self._plan.total_parts + 1)) - set(self._plan.uploaded)
                            work = missing | set(self.state.dirty)
                            if not work:
                                stage = "complete"
                                continue
                            self.source.verify()
                            if self.manifest is None:
                                self.manifest = HashManifest(self.source, self._plan, self.execution)
                            stage = "part"
                            self._upload_parts(work, self.manifest)
                            stage = "status"
                    except MultipartFaultError as fault:
                        stage = self._recover(fault)
                    except (NetworkError, RequestTimeoutError):
                        self._recovery_budget()
                        stage = "refresh" if stage == "init" else "status"
            except MultipartError:
                raise
            except HubError as error:
                raise terminal(error) from error
