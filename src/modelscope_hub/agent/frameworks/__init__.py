# Copyright (c) Alibaba, Inc. and its affiliates.
"""Auto-register all built-in framework workspace specifications.

Importing this package triggers registration of all bundled frameworks into
:data:`modelscope_hub.agent.FRAMEWORK_REGISTRY`.
"""
from . import hermes  # noqa: F401
from . import nanobot  # noqa: F401
from . import openclaw  # noqa: F401
from . import openhuman  # noqa: F401
from . import qoder  # noqa: F401
from . import qwenpaw  # noqa: F401
