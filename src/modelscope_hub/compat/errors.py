"""旧 SDK 下载异常的捕获契约；不改变核心错误码映射。"""

from __future__ import annotations

from typing import Any

from ..errors import FileDownloadError as CoreFileDownloadError
from ..errors import NetworkError


class FileDownloadError(CoreFileDownloadError, NetworkError):
    """保留旧 NetworkError 捕获关系，同时携带整仓下载失败详情。"""

    error_code = NetworkError.error_code

    def __init__(self, message: str = "", *args: Any, source: CoreFileDownloadError | None = None, **kwargs: Any):
        CoreFileDownloadError.__init__(self, message, **kwargs)
        if source is None and not self.failed_files:
            self.retryable = NetworkError.retryable
        if source is not None:
            self.failed_files = dict(source.failed_files)
            self.total = source.total
            self.output_dir = source.output_dir
            self.retryable = source.retryable
            self.error_code = source.error_code

    def __str__(self) -> str:
        return CoreFileDownloadError.__str__(self)
