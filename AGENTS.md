# AGENTS.md — modelscope_hub Engineering Conventions

This document defines repository-level engineering conventions that apply to every automated or manual change in this repository. In addition to the general requirements, it **strictly governs changes to the `upload_folder` upload path**: every change that touches this path must perform the specified scenario-based simulation review and include an Upload Simulation Regression Report.

---

## 1. General Engineering Requirements

- This repository uses a src-layout. Always run code and tests against local sources: `PYTHONPATH=src` (a package installed with `pip install` may be outdated).
- Before committing, run the following gates in order; all must pass:
  1. Targeted unit tests: `PYTHONPATH=src pytest -q <relevant tests>`
  2. Full non-remote test suite: `PYTHONPATH=src pytest -q tests/ -k 'not remote' --ignore=tests/integration`
  3. Style checks: `ruff check src/ tests/` and `ruff format --check src/ tests/`
  4. Type checks: `mypy src/modelscope_hub/`
- Never commit secrets. `~/.modelscope/credentials/` is private (it contains a pickled cookie jar and `m_session_id`); never expose its contents in logs or reports.
- When changing the default value of a configuration constant, update the corresponding tests for its default value and registration table, as well as the environment-variable table in `README.md`.
- Local SDK state must use lightweight files, without embedded database modules or packages. Keep recovery hints sparse and advisory: verify actual files against available size/hash metadata before trusting completion or stale failure records. Missing, corrupt, or unwritable hints must not imply success or prevent valid files from being reused. Each download task must reuse one hidden JSON for sparse recovery hints and its latest summary, without persistent per-call report or result files; healthy files must not trigger per-file state rewrites.

---

## 2. Controlled Scope: `upload_folder` Path

Changing any of the following files or symbols constitutes a change to the upload path and requires the mandatory simulation regression in Section 5:

- `src/modelscope_hub/_upload.py`
  - Batching: `_calculate_adaptive_batch_size`, `_plan_commit_batches`, `_estimate_commit_operation_bytes`, `_plan_result_batches`
  - Routing: `_is_lfs`, `_is_inline_metadata`, `_upload_mode`
  - Main flow and recovery: `upload_folder`, `_retry_failed_commits`, `_retry_failed_files_react`, `_retry_failed_simple`, `_commit_with_retry`
  - Failure classification: `classify_error`, `_ErrorCategory`, `_is_retryable_commit_error`
  - Preflight validation: `_warn_advisory_upload_limits`, `_prepare_upload_folder`
  - State tracking: `UploadTracker` (`begin_attempt` / `mark_*`)
- `src/modelscope_hub/errors.py`: `raise_for_status`, `_COMMIT_RETRYABLE_BUSINESS_CODES`, and HTTP/business-code mappings
- `src/modelscope_hub/_legacy_api.py`: `create_commit`, including the HTTP 200 / `Success:false` check
- `src/modelscope_hub/constants.py`: upload-related constants such as `UPLOAD_*` and `COMMIT_MAX_ACTIONS_PER_REQUEST`

---

## 3. Commit Batching Semantics That Must Be Preserved

Only files that are **pending** (not `COMMITTED`) for the current run are planned into batches. This prevents a resumed upload from turning a small number of remaining files into multiple tiny commits.

Apply the following three constraints; start a new batch as soon as any constraint is reached:

1. **Operation target (primary constraint):** `UPLOAD_COMMIT_BATCH_MAX_OPERATIONS`, default `256`. `_calculate_adaptive_batch_size` must return `min(target, 2000, pending_count)`—for fewer files, it converges to the pending count and by default never exceeds `256`.
2. **Request-body budget (secondary constraint):** `UPLOAD_COMMIT_MAX_INLINE_BYTES`, default `8 MiB`. Count only normal files, using base64 expansion (`×4/3`) plus JSON overhead. For LFS files, count only pointer metadata (about `370 B/op`); **never include blob payload bytes**.
3. **Server hard limit:** `COMMIT_MAX_ACTIONS_PER_REQUEST=2000`. Clamp to this limit in every case; it must never be exceeded.

The following behavior must also be preserved:

- **Tail-batch rebalancing:** when `last_batch × 4 < previous_batch` and both rebalanced halves meet all constraints, rebalance the final two batches (for example, `256/32 → 144/144`). Do not add commits, and preserve deterministic path order.
- **Oversized inline fallback:** if one inline file alone exceeds the budget, place it in its own batch and emit a warning; the upload must not stall.
- **Recovery uses the same planner:** `_retry_failed_commits`, ReAct recovery, and simple retry must all use `_plan_result_batches` (and therefore the same `_plan_commit_batches`).
- **`delete_files` batches independently:** split only at the hard limit of `2000`, not at `256`.
- **`upload_file`:** one file and one commit per call; no batching and no tracker.

### Routing Rules: LFS vs. Inline

- Files matched by `_is_inline_metadata` (such as `README.md`, `.gitattributes`, `.gitignore`, and `config*.json`) are **always inline**.
- Otherwise, `size > UPLOAD_LFS_FORCE_THRESHOLD_BYTES` (default **64 KiB**) routes to LFS.
- Otherwise, files matching `DATASET_LFS_SUFFIX` or `MODEL_LFS_SUFFIX` route to LFS.
- All other files are normal inline files.

---

## 4. Failure and Retry Model That Must Be Preserved

- Failures have three categories, and **all of them must be included in final statistics**:
  - `retry_failed_files`: transport failure → ReAct recovery (parallel or serial retries, backoff, then single-file handling).
  - `retry_commit_batches`: retryable commit failure → **commit-only recovery** (do not re-upload blobs).
  - `terminal_failures`: non-retryable in the current run; record as `FAILED` in the tracker and include in the final `Failed` count.
- **Retryability in commit context** (`_is_retryable_commit_error` / `classify_error(commit_context=True)`):
  - Business code `10030000001` ("branch changed, please retry" / "commit rejected by repository policy") is retryable only at a `/commit/` endpoint; `raise_for_status` maps it to `ServerError`.
  - During commit-only recovery, policy-related rejections recursively split batches in half by file count, down to individual files.
  - Genuine parameter errors (for example, an invalid-path E3021, a 400 without a policy code, or 401/403/404) remain **non-retryable**.
- **HTTP 200 with `Success:false`:** `LegacyClient.create_commit` must raise `APIError` and enter the unified failure path; false successes are prohibited.
- **Abort after consecutive failures:** after `UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES` (default `3`) batches fail completely in succession, the main flow must raise `RuntimeError`. The tracker must already be persisted so that the upload can be resumed.
- **Manual rerun semantics (mandatory):** regardless of the prior error, skip only `COMMITTED` entries. `FAILED`, `UPLOADED`, and missing entries must re-enter the full upload-and-commit path; `begin_attempt` clears the previous `error_type`. A terminal failure must raise `StorageError` and suppress synchronous deletion.
- **Advisory preflight limits:** capacity-related thresholds (total file count, files per directory, individual file size, and total normal-file bytes) must only warn and never block. Structural errors (an empty `repo_id`, empty input, a nonexistent or out-of-bound path, invalid `repo_type`, or a file modified during upload), together with the server's hard limit of `2000`, remain enforced constraints.

---

## 5. Mandatory Simulation Regression for Upload-Path Changes

> Rule: **Before committing any change within the scope in Section 2, perform a scenario-based simulation review of every case below and include an Upload Simulation Regression Report in the delivery notes.** A simulation is a logical derivation from the code's current strategy and must match the implementation. If a conclusion changes because of the current change, explicitly document the before/after difference and its cause.

### 5.1 Required Dataset-Scale Scenarios

| ID | Dataset characteristics | Required simulated conclusions |
|---|---|---|
| D1 | 1 TB / 2 million files, from tens of KiB to tens of MiB | Routing distribution, blob PUT order of magnitude, presign request count, commit-batch order of magnitude, rate-limit risk, memory and hashing overhead |
| D2 | 50 GB / 20 thousand files | Nearly all LFS, presign count, commit-batch count (including tail-batch rebalancing), dominant costs |
| D3 | 5 GB / 2 thousand files | Commit-batch count and tail behavior (including whether rebalancing applies), blob-dominated cost |
| D4 | 200 GB / 10 files | All LFS, **one commit**, blob-transfer dominated, no capacity warning |
| DX | 1,000 files × 1 GB | All LFS, the 8 MiB budget does not apply, commit batches (`256×3+232=4`), approximately 1,000 PUTs |

For every scenario, derive at least: the LFS/normal routing result, the `_calculate_adaptive_batch_size` target, `_plan_commit_batches` count plus operations and estimated body size per batch, blob PUT count, presign request count, advisory warnings triggered, and resume behavior.

### 5.2 Required Error-Type Scenarios

For each error below, describe its classification, whether it is automatically retried in the current run, its handling path, and its final outcome:

- Transport-layer network failure or timeout (`E1020` / `E1001`, blob stage)
- Commit 5xx (`E1002`)
- 429 rate limiting (`E1021`), including `Retry-After` and the configured limit
- 409 branch changed / 400 repository policy (business code `10030000001`, `/commit/`)
- HTTP 200 + `Success:false`
- Genuine parameter error (invalid-path E3021 without a policy code)
- Authentication, authorization, or nonexistent resource (`E3001` / `E3002` / `E3020`)
- File modified during upload (size mismatch)
- Shared blob owner failure (replicas are deferred)
- Three consecutive batches failing completely (protective abort and resumable rerun)

### 5.3 Upload Simulation Regression Report Template

```
# upload_folder Simulation Regression Report
- Change summary / affected Section 2 symbols:
- Baseline configuration snapshot: LFS threshold / 256 / 8 MiB / 2000 / recovery switches / consecutive-failure threshold
- Dataset scenario derivations: D1 / D2 / D3 / D4 / DX (routing, blob PUTs, presigning, commit batches with operations and body sizes, warnings, resume behavior)
- Error-type behavior: each item (classification / automatic retry / handling path / final outcome)
- Difference from the previous version: behavior changed by this modification and the reason (write "None" if there is no difference)
- Automated validation: pytest / ruff / mypy results
- Conclusion: compliance with the semantics in Sections 3 and 4; risks and tuning recommendations
```

### 5.4 Required Automated Validation (Run in Parallel with the Simulation)

- `PYTHONPATH=src pytest -q tests/test_upload_batching.py tests/test_commit_action_limit.py tests/test_errors_openapi_codes.py tests/test_upload_config.py tests/test_upload_lfs_gate.py tests/test_upload_blob_dedup.py`
- If error mappings changed, additionally run `tests/test_login_failure_paths.py`.
- The full non-remote suite, ruff, and mypy must pass.

---

## 6. Change Checklist

After changing the upload path, check whether the following must be updated as well:

- `tests/`: assertions for batching, errors, configuration, deduplication, LFS gating, and related behavior—especially assertions for default values.
- The upload environment-variable table and semantic documentation in `README.md`.
- `__all__` exports and compatibility aliases in `src/modelscope_hub/compat/` and `constants.py`.
- This file, if new constants, error codes, or scenarios were introduced.

---

## 7. User-Agent (UA) Requirements

- Every SDK HTTP request must include a `User-Agent` starting with `modelscope_hub/<version>`. Use `utils.build_user_agent` / `append_user_agent` and enforce coverage at client send boundaries across OpenAPI, Legacy, Agent, both sessions, anonymous requests, retries, and probes.
- Custom UA information must be appended, never replacing the SDK identifier. Handle header names case-insensitively; empty values must not remove the SDK UA. Propagate CLI / compat markers (`invoked_by/cli` / `invoked_by/compat`) to derived clients and download requests.
- Prefer a persisted `session_id`; fall back to a process-stable ID when storage is not writable. Telemetry persistence failures must not block business requests. UA must not contain tokens, cookies, or other credentials.
- Changes to HTTP transport, client construction, or delegating entry points must verify these invariants against actual `PreparedRequest` objects; checking session default headers alone is insufficient.

---

## 8. vLLM Compatibility Contract

- **Cache paths:** Locate a repository only through the path returned by `snapshot_download()`, `snapshot_download(..., local_files_only=True)`, or `resolve_cached_snapshot()`; never append a repository ID to a legacy helper root such as `get_model_cache_root()`. Keep legacy helper return values unchanged, and keep both standard and legacy cache layouts reusable.
- **No runtime patching or Hugging Face dependency:** modelscope-hub must not import, patch, or hook vLLM or other consumers, and must not depend on `huggingface_hub` or its settings. Hugging Face API semantics, such as exception types for missing files, belong to the SDK compatibility layer `modelscope.utils.hf_util`.
- **Legacy path link:** For releases that still append IDs to the legacy model root, the legacy-compatible `snapshot_download()` keeps `<legacy model root>/<owner>/<name>` linked to the directory it returns (the snapshot or a reused legacy cache). It applies to models in the default or `MODELSCOPE_CACHE` root, is best-effort, replaces only its own links into the repository's snapshots, never real directories or other links, and is skipped for offline resolution. Legacy-cache detection, scanning, and cleanup treat the link as an alias, never as a legacy repository. Remove it once supported vLLM releases resolve paths through the SDK API.
- **Arguments and exceptions:** Propagate `revision`, `cache_dir`, and `local_files_only` through every layer (shim → compat → implementation). An empty revision at a compatibility entry point selects the default revision; an explicit revision missing from the standard snapshots must not fall back to the default. Shim modules must keep exporting the exceptions callers import from them, such as `InvalidParameter` from `modelscope.hub.snapshot_download`.
- **Offline:** `local_files_only=True` resolution performs no network access, credential initialization, or writes, and raises `CacheNotFound` when the cache is missing.
- **Regression:** Changes affecting this integration must be verified against a real vLLM installation: online multimodal serving by repository ID, offline serving, and legacy caches. Unit tests mock only the lowest-level dependency and verify network attempts and disk writes; high-level mocks alone do not prove compatibility.

---

## 9. Download-Count Contract (`snapshot-identifier`)

### 9.1 Task Granularity and Implementation Boundary

- `snapshot-identifier` identifies one logical snapshot task, not a file, HTTP request, thread, process, or UA session. Generate it once at the `download_repo` / `download_repo_report` entry point and pass it down explicitly; header builders only read it and never generate a UUID.
- Every file, worker, transport retry, integrity retry, Range resume, parallel part, and route probe of one task must reuse that identifier. Reuse inside a single file-transfer loop does not prove repository-wide granularity; the outer integrity-retry chain must carry it too.
- Standalone `download_file`, compat single-file calls, an explicit file list in normal CLI mode, and kernels `hf_hub_download` must not send the header. In CLI report mode the file list is a snapshot filter and still shares the task identifier. The HTTP layer must never infer the task type.
- Never store the identifier in a mutable shared field on `DownloadManager` or `HubApi`; concurrent or successive tasks on one instance must stay isolated. A non-blank caller-supplied identifier must not be overwritten, replaced by the UA `session_id`, derived from file paths, merged with `X-Request-ID`, or rotated over time.
- Return a local file that passes the current size/hash check before building any download request; metadata compatibility paths are cache-first too. Never send requests only to increment counters, and never block or fake success on counting failures.
- Report mode reuses `snapshot: {id, open}` in the existing hidden JSON: an unfinished task reuses the ID, and a new ID starts after `COMPLETE`. Healthy files are not persisted individually and no per-file writes are added. A corrupt, missing, or unwritable hint may only reduce recovery/deduplication and never replaces file-integrity validation.
- Strict mode treats each call as a new task; report-mode resume must distinguish new from resumed tasks. For unpersisted boundaries such as a hard kill right after a `COMPLETE` tracker, state the possible over-count; never claim strict deduplication under arbitrary crashes.

### 9.2 Server-Side Counting and Evidence Boundary

- Scope is the dataset file endpoint. The deduplication key is dataset ID + identifier and excludes user, IP, file, revision, and UA. A qualifying request with a non-blank new identifier counts +1 for the dataset; a missing or blank identifier accrues fractionally against the server denominator N. The SDK never submits weights and never assumes N=1000. `Source=SDK` and UA are channel attribution only; they never replace or correct the snapshot identifier, and integrations must confirm the target protocol version.
- HEAD, `View=true`, and exactly root-level `README.md`, `dataset_infos.json`, and `.gitattributes` are neither counted nor deduplicated. Same-named files in nested directories do not inherit the root-path exclusion.
- The TTL is 7200 s from first insertion and repeats do not refresh it, so reusing an ID after two hours may count again. The +1 occurs on the first qualifying request, not at completion. A 302 does not prove full receipt, cancellation or failure has no rollback protocol, and asynchronous deduplication is not atomic with the DB increment. Never claim exactly-once.
- Snapshots with include/exclude filters still count as one task. Model, skill archive, explicit `download/increase`, and other download paths must be verified separately; dataset semantics must not be generalized. Model counts remain 0 for both SDK generations and are recorded as unconfirmed only.
- For every scenario, take a stable baseline, then observe asynchronous persistence within a stability window while verifying actual request headers, request count, unique-ID count, and integer/fractional delta. When an expected value exists, assert equality with the measured delta; a broad oracle range is not sufficient.
- Legacy `Downloads` and OpenAPI `downloads` must not be assumed equivalent. Probe fractional conversion with independent requests at integer multiples of N and record both readings. Without action logs, SQL, or deployment metadata, state the evidence gap; never treat a page integer or a log-line count as the exact DB value.
- Inspect the actual `PreparedRequest` with case-insensitive header lookup and record only required fields plus credential presence. Never persist tokens, cookie values, or presigned URLs. A WAF cookie is not a login credential; anonymous assertions must distinguish `Authorization` / `m_session_id` from ordinary cookies.

### 9.3 Mandatory Regression Matrix

- Changes touching task context, request headers, recovery state, or delegating entry points must cover: inline+LFS repositories, multiple workers, sequential and same-instance concurrent tasks, Range/disconnect/integrity retries, single file, filters, full cache hit, root metadata plus same-named nested files, report-mode cancellation / cross-process resume / a new task after `COMPLETE`, and a corrupt or unwritable tracker.
- Offline tests mock only the lowest-level dependency and capture real HTTP requests; a high-level `snapshot_download` mock cannot prove argument, UA, task-ID, or compatibility propagation. Online runs use isolated local output, compare downloaded size/hash against the server-side file listing, and never substitute client-side upload routing for server echo.
- Online verification pins the HTTPS endpoint and an authorized owner and prefers registered private test repositories. Public visibility is opened only in an approved whitelist window and restored to private in `finally`. Creation, upload, visibility change, and deletion are never self-authorized by "verification" needs; deletion requires prior approval of the exact list plus hard scope checks.
- Preserve unmodified baseline and failure request evidence. When a framework assertion is found incorrect, record why and re-judge or rerun; never turn a failure into a pass by relaxing SDK invariants. Reproduce pre-existing dependency/API failures on the unmodified baseline, then isolate the version and verify, without folding them into the current fix.

### 9.4 `modelscope` and Hub Dual-Package Compatibility

- "Latest version" means the official stable release: verify its dependency constraint and record the wheel SHA-256, both package versions, and the actual import path. A local `+main` tree, a stale editable install, or egg-info is not evidence of a release. Use an isolated workspace environment with Hub pointing at local `src`; never replace the user's global environment.
- Verify from real `modelscope` entry points: snapshot, dataset snapshot, model and dataset single file, `HubApi.download_model`, and the branded CLI. Preserve historical positional arguments, `revision`, `cache_dir`, `local_dir`, token and endpoint propagation, `str` return values, exceptions, and the UA markers `invoked_by/compat` / `invoked_by/cli`.
- Cover online download, cache reuse, and `local_files_only`. Offline must not only return the correct path but also prove no network attempt, credential initialization, or disk write. Report parameters unsupported by a shim are recorded as capability boundaries and never faked by dynamic patching.
- Never deliver a download-count fix by modifying consumers or vLLM, adding a Hugging Face dependency, or runtime hooks; optional installs serve isolated regression only. A change that affects the Section 8 integration still requires its real-vLLM verification.
