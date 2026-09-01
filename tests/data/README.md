# Vendored ModelScope OpenAPI specification

`openapi.json` in this directory is a verbatim copy of the live ModelScope
OpenAPI document, kept in-tree so that spec drift becomes a test failure instead
of a silent gap.

| | |
|---|---|
| Source | `https://modelscope.cn/openapi/v1` (document served alongside it) |
| `info.version` | `1.1.0+master.20260813T030041Z` |
| `openapi` | `3.1.1` |
| Operations | 56 across 11 tags |

## Why it is here

`tests/test_openapi_coverage.py` reads this file and asserts that every
`operationId` in the tags the SDK claims to cover is registered in
`modelscope_hub._openapi.OPERATION_REGISTRY`. When the service publishes a new
operation, refreshing this file makes the guard fail by name rather than leaving
the gap to be discovered by hand later.

Tags not yet implemented are listed in that test's `_DEFERRED_TAGS`; the set
doubles as the remaining to-do list.

## Refreshing

Replace the file with the current document and update `info.version` above. Any
new operation in a covered tag will fail the guard until it is either
implemented and registered, or explicitly deferred.
