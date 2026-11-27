"""Monkey-patching of the ``kernels`` library onto the ModelScope Hub.

Only kernels' Hub-API factory (``_get_hf_api`` in ``kernels.hf_hub``, or
``kernels._internals.hf_hub`` since 0.19) is replaced; every download, file
listing, and revision lookup that ``kernels`` performs goes through
:class:`modelscope_hub.kernels._hub.MsKernelApi`, so the kernel loading,
variant resolution, and dependency logic stay untouched.

``patch_scope()`` scopes the patch to a single call (used by the public
wrappers in :mod:`modelscope_hub.kernels._load`), so ``from kernels import
get_kernel`` keeps using the Hugging Face Hub unless the user explicitly
applies ``patch_kernels_hub()``.

Adapted from ``modelscope/utils/hf_util/patcher.py`` (Apache-2.0, Alibaba),
re-targeted from the legacy ``modelscope`` SDK to ``modelscope_hub``.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from types import ModuleType

from modelscope_hub.kernels._hub import MsKernelApi

__all__ = ["patch_kernels_hub", "patch_scope", "unpatch_kernels_hub"]


def _kernels_package() -> str:
    """The package holding kernels' private modules: whatever the installed layout.

    kernels 0.19 moved every private module under ``kernels._internals``
    (``kernels.hf_hub`` became ``kernels._internals.hf_hub``); earlier releases
    keep them at the top level. Requires ``kernels`` to be installed.
    """
    try:
        importlib.import_module("kernels._internals")
        return "kernels._internals"
    except ImportError:
        return "kernels"


def _kernels_submodule(name: str) -> ModuleType:
    """Import a private ``kernels`` submodule (e.g. ``"locking"``) whatever its layout.

    Importing by name (rather than ``from kernels.x import y``) resolves to the
    very module object ``kernels`` itself bound its helpers from, which is what
    the patch below has to replace. Raises ``ImportError`` when the submodule
    does not exist.
    """
    return importlib.import_module(f"{_kernels_package()}.{name}")


def _import_kernels_api_module():
    """Return the module holding ``_get_hf_api`` in the installed ``kernels``.

    kernels 0.19 keeps it in ``kernels._internals.hf_hub``, 0.17–0.18 in
    ``kernels.hf_hub``, and older releases in ``kernels.utils``; support all
    three layouts. Returns ``None`` when ``kernels`` is not installed.
    """
    try:
        import kernels  # noqa: F401
    except ImportError:
        return None
    for module_name in ("_internals.hf_hub", "hf_hub", "utils"):
        try:
            return importlib.import_module(f"kernels.{module_name}")
        except ImportError:
            continue
    return None


def _ms_api(user_agent=None):
    """Replacement for ``kernels.hf_hub._get_hf_api``."""
    return MsKernelApi(user_agent=user_agent)


def patch_kernels_hub() -> None:
    """Monkey-patch the ``kernels`` library to route Hub API calls to ModelScope.

    Only ``_get_hf_api`` is replaced (both in its home module and in the
    submodules that bind it at import time, e.g. ``kernels.load``). Returns
    silently when ``kernels`` is not installed or already patched.
    """
    kernels_api = _import_kernels_api_module()
    if kernels_api is None or hasattr(kernels_api, "_get_hf_api_origin"):
        return

    kernels_api._get_hf_api_origin = kernels_api._get_hf_api
    kernels_api._get_hf_api = _ms_api
    # kernels >= 0.17 binds `_get_hf_api` at import time in submodules
    # (e.g. `from kernels.hf_hub import _get_hf_api` in `kernels.load`);
    # patch those already-bound references as well.
    for name, module in list(sys.modules.items()):
        if name != "kernels" and not name.startswith("kernels."):
            continue
        if module is kernels_api or hasattr(module, "_get_hf_api_origin"):
            continue
        if getattr(module, "_get_hf_api", None) is not None:
            setattr(module, "_get_hf_api_origin", getattr(module, "_get_hf_api"))
            setattr(module, "_get_hf_api", _ms_api)


def unpatch_kernels_hub() -> None:
    """Undo :func:`patch_kernels_hub` and restore the original HfApi factory."""
    kernels_api = _import_kernels_api_module()
    if kernels_api is None:
        return
    origin = getattr(kernels_api, "_get_hf_api_origin", None)
    if origin is None:
        return
    patched = kernels_api._get_hf_api
    for name, module in list(sys.modules.items()):
        if name != "kernels" and not name.startswith("kernels."):
            continue
        if module is kernels_api:
            continue
        module_origin = getattr(module, "_get_hf_api_origin", None)
        if module_origin is not None:
            setattr(module, "_get_hf_api", module_origin)
            delattr(module, "_get_hf_api_origin")
        elif getattr(module, "_get_hf_api", None) is patched:
            # Imported while the patch was active (kernels >= 0.17).
            setattr(module, "_get_hf_api", origin)
    kernels_api._get_hf_api = origin
    del kernels_api._get_hf_api_origin


@contextlib.contextmanager
def patch_scope():
    """Apply :func:`patch_kernels_hub` for the duration of the ``with`` block.

    Nesting is safe: when an outer patch is already in effect the block simply
    reuses it, so only the outermost scope restores the original factory.
    """
    kernels_api = _import_kernels_api_module()
    if kernels_api is None or hasattr(kernels_api, "_get_hf_api_origin"):
        yield
        return
    patch_kernels_hub()
    try:
        yield
    finally:
        unpatch_kernels_hub()
