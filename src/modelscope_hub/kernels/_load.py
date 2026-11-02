"""Public entry points: scope-managed wrappers around the ``kernels`` library.

Each wrapper applies :func:`modelscope_hub.kernels._patch.patch_scope` around
the corresponding ``kernels`` call, so Hub traffic is redirected to ModelScope
for that call only — ``import kernels`` elsewhere keeps using the Hugging
Face Hub.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from modelscope_hub.kernels._patch import patch_scope

__all__ = ["get_kernel", "get_kernel_variants", "get_local_kernel", "has_kernel"]


def _require_kernels() -> ModuleType:
    """Import ``kernels`` or raise an actionable ``ImportError``."""
    try:
        import kernels
    except ImportError as e:
        raise ImportError(
            "The `kernels` package is required by `modelscope_hub.kernels` but is not installed. "
            'Install it with: pip install "modelscope-hub[kernels]".'
        ) from e
    return kernels


def get_kernel(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    user_agent: str | dict | None = None,
    trust_remote_code: bool | list[str] = False,
    check_arch: bool = True,
) -> ModuleType:
    """Load a kernel from the ModelScope Hub.

    This function downloads a kernel to the local ModelScope cache directory
    (if it was not downloaded before) and then loads the kernel. The kernel
    repository must be a ModelScope model repository mirroring the HF kernel
    repo layout (``build/<variant>/...`` files plus ``v1``-style version
    branches or tags).

    Args:
        repo_id (`str`):
            The ModelScope model repository containing the kernel,
            e.g. ``"kernels-community/activation"``.
        revision (`str`, *optional*):
            The specific revision (branch, tag, or commit) to download.
            Cannot be used together with `version`.
        version (`int`, *optional*):
            The kernel version to download (resolved against the repository's
            ``v<N>`` branches/tags). Cannot be used together with `revision`.
            Either `version` or `revision` must be specified.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the
            backend that Torch is compiled for. The backend is detected
            automatically if not provided.
        user_agent (`Union[str, dict]`, *optional*):
            Extra telemetry info appended to the ModelScope SDK user agent.
        trust_remote_code (`bool | list[str]`, *optional*, defaults to `False`):
            Whether to allow loading kernels from untrusted organisations.
            ModelScope has no trusted-publisher flag, so trust is decided
            against the built-in allowlist (see
            :data:`modelscope_hub.kernels._hub.TRUSTED_KERNEL_PUBLISHERS`); a
            list of repository IDs allows those repositories in addition.
        check_arch (`bool`, *optional*, defaults to `True`):
            Whether to check that the kernel build supports the architecture
            of the current device. `check_arch=False` skips the check.

    Returns:
        `ModuleType`: The imported kernel module.

    Example:
        ```python
        import torch
        from modelscope_hub.kernels import get_kernel

        activation = get_kernel("kernels-community/activation", version=1)
        x = torch.randn(10, 20, device="cpu")
        out = torch.empty_like(x)
        result = activation.relu(out, x)
        ```
    """
    kernels = _require_kernels()
    with patch_scope():
        return kernels.get_kernel(
            repo_id,
            revision=revision,
            version=version,
            backend=backend,
            user_agent=user_agent,
            trust_remote_code=trust_remote_code,
            check_arch=check_arch,
        )


def has_kernel(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    trust_remote_code: bool | list[str] = False,
    check_arch: bool = True,
) -> bool:
    """Check whether a kernel build exists for the current environment.

    Arguments are the same as :func:`get_kernel` (minus ``user_agent``).

    Returns:
        `bool`: `True` if a kernel is available for the current environment.
    """
    kernels = _require_kernels()
    with patch_scope():
        return kernels.has_kernel(
            repo_id,
            revision=revision,
            version=version,
            backend=backend,
            trust_remote_code=trust_remote_code,
            check_arch=check_arch,
        )


def get_kernel_variants(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
) -> list:
    """Resolve all build variants of a kernel against the current environment.

    Arguments are the same as :func:`get_kernel` (minus ``user_agent``).

    Returns:
        `list`: One ``VariantAccepted`` or ``VariantRejected`` per build
        variant in the repository, compatible variants first.
    """
    kernels = _require_kernels()
    with patch_scope():
        return kernels.get_kernel_variants(
            repo_id,
            revision=revision,
            version=version,
            backend=backend,
        )


def get_local_kernel(
    repo_path: Path,
    *,
    backend: str | None = None,
    trust_remote_code: bool | list[str] = False,
) -> ModuleType:
    """Import a kernel from a local kernel repository path.

    If the kernel has any (transitive) dependencies, they are downloaded from
    the ModelScope Hub.

    Args:
        repo_path (`Path`):
            The local path to the kernel repository.
        backend (`str`, *optional*):
            The backend to load the kernel for.
        trust_remote_code (`bool | list[str]`, *optional*, defaults to `False`):
            Whether to allow loading kernel dependencies from untrusted
            organisations.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    kernels = _require_kernels()
    with patch_scope():
        return kernels.get_local_kernel(
            repo_path,
            backend=backend,
            trust_remote_code=trust_remote_code,
        )
