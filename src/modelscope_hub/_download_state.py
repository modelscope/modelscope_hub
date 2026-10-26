"""Durable report-mode state; kept outside downloaded repository contents."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import tempfile
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO
from urllib.parse import urlsplit, urlunsplit

from .errors import InvalidParameter
from .types import DownloadFileResult, DownloadReport
from .utils.logger import get_logger

logger = get_logger("download")


def safe_error(error: BaseException) -> str:
    """Keep diagnostics without persisting credentials or presigned queries."""
    text = str(error)
    text = re.sub(
        r"https?://[^\s)]+",
        lambda m: urlunsplit(urlsplit(m[0])._replace(query="", fragment="", netloc=urlsplit(m[0]).hostname or "")),
        text,
    )
    text = re.sub(r"(?i)(token|signature|authorization|cookie|password|secret)\s*[=:]\s*[^\s,;]+", r"\1=***", text)
    return text[:2000]


class _DownloadState:
    """Sparse JSON recovery hints, never an authority for file completeness.

    Persist exceptional files, failed pages, and the latest summary together.
    Per-file results stay in an anonymous spool owned by the returned report;
    no permanent JSONL or per-run summary files are created. Healthy transfers
    do not rewrite the tracker; actual partial files suffice after a crash.
    """

    def __init__(self, path: Path, identity: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.identity = dict(identity)
        self.lock = threading.Lock()
        self._exceptions: dict[str, DownloadFileResult] = {}
        self._pages: dict[str, dict[int, str]] = {}
        self._active: dict[str, DownloadFileResult] = {}
        self._seen: set[bytes] = set()
        self._run: str | None = None
        self._spool: TextIO | None = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._counts: dict[str, int] = {}
        self._summary: dict | None = None
        self.warnings: list[str] = []
        try:
            self._load()
            if not path.exists():
                self._save()
        except BaseException:
            self.close()
            raise

    def _warn(self, message: str) -> None:
        if message not in self.warnings and len(self.warnings) < 10:
            self.warnings.append(message)
            logger.warning("%s: %s", message, self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("identity"), dict):
                raise ValueError("Invalid tracker header")
            if data["identity"] != self.identity:
                raise InvalidParameter(
                    "Download tracker belongs to a different endpoint/repository/revision/output directory"
                )
            files, pages = data.get("files", {}), data.get("pages", {})
            if not isinstance(files, dict) or not isinstance(pages, dict):
                raise ValueError("Invalid recovery records")
            for path, raw in files.items():
                try:
                    if not isinstance(raw, dict) or raw.get("path") != path:
                        raise ValueError("Invalid file record")
                    result = DownloadFileResult.from_dict(raw)
                    if result.status not in ("FAILED", "DEFERRED", "DOWNLOADING", "PENDING"):
                        raise ValueError("Not a recovery hint")
                    if (
                        not isinstance(result.retries, int)
                        or result.retries < 0
                        or not isinstance(result.recovery_seconds, (int, float))
                        or not math.isfinite(result.recovery_seconds)
                        or result.recovery_seconds < 0
                    ):
                        raise ValueError("Invalid recovery budget")
                    if result.status == "DOWNLOADING":
                        result.status = "DEFERRED"
                        result.error = "Previous process stopped during recovery"
                        result.error_type = "InterruptedDownload"
                    self._exceptions[path] = result
                except (ValueError, TypeError):
                    self._warn("Ignoring malformed file recovery hints; actual files will be revalidated")
            for root, records in pages.items():
                try:
                    if not isinstance(records, dict):
                        raise ValueError("Invalid page records")
                    self._pages[root] = {
                        int(page): error
                        for page, error in records.items()
                        if str(page).isdigit() and int(page) > 0 and isinstance(error, str)
                    }
                except (ValueError, TypeError):
                    self._warn("Ignoring malformed listing recovery hints")
        except InvalidParameter:
            raise
        except (ValueError, UnicodeError, OSError):
            self._exceptions.clear()
            self._pages.clear()
            self._warn(
                "Cannot load download recovery hints; continuing with actual-file validation and bounded retries"
            )

    def _save(self) -> None:
        """Atomically replace sparse hints; write failures do not invalidate data."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {
                        "version": 1,
                        "identity": self.identity,
                        "files": {path: asdict(result) for path, result in self._exceptions.items()},
                        "pages": self._pages,
                        "report": self._summary,
                    },
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            temporary.replace(self.path)
        except OSError:
            self._warn("Cannot save download recovery hints; restart isolation may be unavailable")
        finally:
            if "temporary" in locals():
                with contextlib.suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def close(self) -> None:
        if self._spool is not None:
            self._spool.close()
            self._spool = None

    def failed_pages(self, root: str, *, reset: bool = False) -> dict[int, str]:
        with self.lock:
            if reset and self._pages.pop(root, None) is not None:
                self._save()
            return dict(self._pages.get(root, {}))

    def fail_page(self, root: str, page: int, error: Exception) -> None:
        with self.lock:
            message = safe_error(error)
            if self._pages.setdefault(root, {}).get(page) != message:
                self._pages[root][page] = message
                self._save()

    def select(self, run: str, path: str, size: int | None, sha256: str | None) -> DownloadFileResult | None:
        with self.lock:
            if self._run is None:
                self._run = run
            if self._run != run:
                raise InvalidParameter("A download state instance belongs to one invocation")
            key = hashlib.sha256(path.encode()).digest()
            if key in self._seen:
                return None
            self._seen.add(key)
            old = self._exceptions.get(path)
            if old is not None and (old.size, old.sha256) == (size, sha256):
                result = replace(old)
            else:
                result = DownloadFileResult(path=path, size=size, sha256=sha256)
                if self._exceptions.pop(path, None) is not None:
                    self._save()
            self._active[path] = result
            return replace(result)

    def begin(self, path: str, *, reset: bool = False) -> DownloadFileResult:
        with self.lock:
            result = self._active[path]
            if reset:
                result.retries, result.recovery_seconds = 0, 0
            result.status = "DOWNLOADING"
            result.error, result.error_type, result.retryable = None, None, False
            return replace(result)

    def budget(self, path: str, retries: int, seconds: float) -> None:
        with self.lock:
            result = self._active[path]
            result.retries, result.recovery_seconds = retries, seconds
            self._exceptions[path] = replace(result)
            self._save()

    def finish(self, path: str, status: str, error: BaseException | None = None) -> None:
        with self.lock:
            result = self._active.pop(path, None)
            if result is None:
                return
            result.status = status
            if error is not None:
                result.error, result.error_type = safe_error(error), type(error).__name__
                result.retryable = bool(getattr(error, "retryable", False))
            elif status in ("VERIFIED", "PENDING"):
                result.error, result.error_type, result.retryable = None, None, False
            old = self._exceptions.get(path)
            keep = status in ("FAILED", "DEFERRED") or (status == "PENDING" and result.retries > 0)
            if keep:
                self._exceptions[path] = replace(result)
            else:
                self._exceptions.pop(path, None)
            if old != self._exceptions.get(path):
                self._save()
            assert self._spool is not None
            self._spool.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            self._counts[status] = self._counts.get(status, 0) + 1

    def report(self, run: str, report: DownloadReport) -> DownloadReport:
        for path in list(self._active):
            self.finish(path, "PENDING")
        report.total = sum(self._counts.values())
        report.succeeded = self._counts.get("VERIFIED", 0)
        report.failed = self._counts.get("FAILED", 0)
        report.deferred = self._counts.get("DEFERRED", 0)
        report.pending = self._counts.get("PENDING", 0)
        report.state_warnings = list(self.warnings)
        if report.status != "CANCELLED":
            report.status = (
                "COMPLETE"
                if report.listing_complete and not report.global_error and report.succeeded == report.total
                else "PARTIAL"
            )
        report.report_path = str(self.path)
        report.results_path = ""  # No persistent per-file manifest.
        assert self._spool is not None
        self._spool.flush()
        report._attach_file_results(self._spool)
        self._spool = None
        self._summary = {**asdict(report), "run_id": run}
        self._save()
        report.state_warnings = list(self.warnings)
        return report
