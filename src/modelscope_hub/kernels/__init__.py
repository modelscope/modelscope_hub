"""ModelScope-flavoured ``kernels``: load HF-style compute kernels from ModelScope.

``modelscope_hub.kernels`` is a drop-in wrapper around the `kernels
<https://github.com/huggingface/kernels>`_ package that redirects every Hub
interaction (revision lookup, metadata download, build snapshot) from the
Hugging Face Hub to the ModelScope Hub. Kernel repositories on ModelScope are
ordinary model repositories (e.g. ``kernels-community/activation``) that
mirror the HF kernel repo layout: ``build/<variant>/...`` files plus ``v1``-style
version branches or tags.

The kernel loading, variant resolution, and dependency logic of ``kernels``
are reused unchanged; only its Hub API factory is swapped for a ModelScope
backed shim while a ``modelscope_hub.kernels`` call is in flight. This is
also the implementation behind the ``kernels`` monkey-patching applied by the
``modelscope`` package's ``patch_hub()``.

Requires the optional ``kernels`` dependency::

    pip install "modelscope-hub[kernels]"

Example:
    ```python
    from modelscope_hub.kernels import get_kernel

    # Download optimized kernels from the ModelScope hub
    activation = get_kernel("kernels-community/activation", version=1)
    ```
"""

from __future__ import annotations

from modelscope_hub.kernels._load import (
    get_kernel,
    get_kernel_variants,
    get_local_kernel,
    get_locked_kernel,
    has_kernel,
    install_kernel,
    install_kernel_all_variants,
    load_kernel,
)
from modelscope_hub.kernels._patch import patch_kernels_hub, patch_scope, unpatch_kernels_hub

__all__ = [
    "get_kernel",
    "get_kernel_variants",
    "get_local_kernel",
    "has_kernel",
    "install_kernel",
    "install_kernel_all_variants",
    "get_locked_kernel",
    "load_kernel",
    "patch_kernels_hub",
    "patch_scope",
    "unpatch_kernels_hub",
]
