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

## 9. 下载计数规约（`snapshot-identifier`）

### 9.1 任务粒度与实现边界
- `snapshot-identifier` 表示一次逻辑快照任务，不是文件、HTTP 请求、线程、进程或 UA 会话的标识。只能在 `download_repo` / `download_repo_report` 的任务入口生成一次，并显式向下透传；请求头构建器只读取标识，不生成 UUID。
- 同一任务的所有文件、并发 worker、传输重试、完整性重试、Range 续传、并行分段及路由探测必须复用同一标识。一次单文件传输循环内复用标识，不足以证明整库粒度正确；完整性重试的外层调用链也必须覆盖。
- 独立 `download_file`、compat 单文件、CLI 普通模式的显式文件列表及 kernels `hf_hub_download` 不发送该头。CLI 报告模式的文件列表作为快照过滤条件，仍共用任务标识；不得由 HTTP 层猜测任务类型。
- 禁止把任务标识存成 `DownloadManager` / `HubApi` 的可变共享字段；同一实例并发或顺序发起的新任务必须相互隔离。非空白调用方标识不得被覆盖，不得替换为 UA `session_id`、拼接文件路径、合并 `X-Request-ID` 或按时间轮换。
- 本地文件通过当前大小/可用哈希校验后，应在构建下载请求之前返回；元文件兼容路径同样必须缓存优先。不发送补计数请求，不因统计失败阻塞或伪造下载成功。
- 报告模式复用原隐藏 JSON 的 `snapshot: {id, open}`：未结束任务复用 ID，`COMPLETE` 后新建 ID；健康文件不逐条持久化、不增加逐文件写盘。损坏、缺失或不可写的提示只能降低恢复/去重能力，不能代替文件完整性判定。
- 严格模式每次调用是新任务；报告模式恢复必须区分新任务与旧任务。对已完成 tracker 后的新任务硬中断等未持久化边界，应说明可能多计，不承诺任意崩溃下严格去重。

### 9.2 服务端计数与证据边界
- 本规约针对数据集文件入口：去重键是“数据集 ID + 标识”，不含用户、IP、文件、revision 或 UA。合格请求的非空白新标识计整库 +1；无标识/空白标识按服务端分母 N 折算。SDK 不自行提交权重，不把所有分母都假设为 1000。`Source=SDK` 与 UA 仅用于渠道归因，不能替代或修正快照标识；联调须核验目标服务协议版本。
- HEAD、`View=true` 及根路径精确匹配的 `README.md`、`dataset_infos.json`、`.gitattributes` 不计数、不占用去重标识；嵌套目录同名文件不可套用根路径排除规则。
- TTL 为首次占键后的 7200 秒，重复请求不刷新 TTL；超过两小时复用原 ID 仍可能再次 +1。整库 +1 发生在首个合格请求，而非完成时；302 不证明内容已接收完整，取消/失败没有回滚协议，异步去重与 DB 增量非原子，禁止宣称 exactly-once。
- include/exclude 的快照仍按一次任务计数。模型、skill archive、显式 `download/increase` 及其他下载链路须分别核验，不推广数据集统计语义；模型计数新旧均为 0，只能记为统计未证实。
- 每个场景先取稳定基线，再观察异步落库与稳定窗口，同时核验真实请求头、请求数、唯一 ID 数和整数/小数增量；有明确预期值时，必须断言实测增量等于预期，不能仅接受宽泛 oracle 区间。
- legacy `Downloads` 与 OpenAPI `downloads` 的展示口径不得先验视为等价。使用 N 的整数倍独立请求检验小数折算，记录两接口差异；未读取动作日志、SQL 或部署元数据时，必须明确证据缺口，不把页面整数或动作日志条数当成 DB 精确值。
- 抓取实际 `PreparedRequest`，大小写不敏感地核验头；仅记录必要字段及鉴权存在性，不保存 token、Cookie 值或签名 URL。WAF Cookie 不等于登录凭证，匿名断言应区分 Authorization / `m_session_id` 与普通 Cookie。

### 9.3 必须保留的回归矩阵
- 涉及任务上下文、请求头、恢复状态或委托入口的变更，必须覆盖：整库 inline+LFS、多 worker、顺序与同实例并发任务、Range/断连重试/完整性重试、单文件、过滤、全缓存命中、根元文件与嵌套同名文件、报告模式取消/跨进程恢复/完成后新任务、损坏/不可写 tracker。
- 离线测试只模拟最底层依赖，抓取实际 HTTP 请求；高层 `snapshot_download` mock 不能证明参数、UA、任务 ID 或兼容性透传正确。线上使用独立本地输出，核对下载大小/哈希和服务端实际文件列表，不用客户端上传路由预测代替服务端回显。
- 线上验证显式固定 HTTPS 站点与授权 owner，优先复用登记的 private 测试仓库；public 只在已批准的白名单窗口开放，并在 finally 恢复 private。创建、上传、改可见性、删除不能由“验证”要求自行扩权；删除须先审批具体清单并硬校验作用域。
- 保留未经改写的基线/失败请求证据；发现测试框架误判时记录原因并重判或重新运行，禁止通过放宽 SDK 不变量把失败变为成功。依赖/API 的既有失败应在未修改基线复现，再隔离版本验证，不混入本轮无关修复。

### 9.4 `modelscope` 与 Hub 双包兼容验收
- “最新版本”必须核验官方稳定发布包及依赖约束，记录 wheel SHA-256、两包版本和实际导入路径；本地 `+main`、旧 editable 安装或 egg-info 不能充当最新发布版证据。使用工作区隔离环境，Hub 指向本地 `src`，不替换用户全局环境。
- 从真实 `modelscope` 入口验证 snapshot/dataset snapshot、模型/数据集单文件、`HubApi.download_model` 与品牌 CLI；保留历史 positional 参数、`revision`、`cache_dir`、`local_dir`、token/endpoint 透传、str 返回值、异常和 UA `invoked_by/compat` / `invoked_by/cli`。
- 必须覆盖线上下载、缓存复用和 `local_files_only`；离线不仅要返回正确路径，还必须确认无网络尝试、凭证初始化或写盘。shim 不支持的新报告参数应记录能力边界，不能动态打补丁伪装支持。
- 不通过修改消费者/vLLM、新增 Hugging Face 依赖或运行时 hook 交付下载计数修复；安装可选依赖仅为隔离回归使用。修改影响第 8 节集成时仍执行其真实 vLLM 验证要求。
