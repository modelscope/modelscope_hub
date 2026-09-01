<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<div align="center">

The official Python SDK & CLI for [ModelScope Hub](https://modelscope.cn) — download, upload, and manage AI assets from one unified interface.

[![PyPI](https://img.shields.io/pypi/v/modelscope-hub)](https://pypi.org/project/modelscope-hub/)
[![Python](https://img.shields.io/pypi/pyversions/modelscope-hub)](https://pypi.org/project/modelscope-hub/)
[![license](https://img.shields.io/github/license/modelscope/modelscope_hub.svg)](https://github.com/modelscope/modelscope_hub/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/modelscope_hub.svg)](https://github.com/modelscope/modelscope_hub/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/modelscope_hub.svg)](https://github.com/modelscope/modelscope_hub/pulls)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/modelscope_hub)](https://github.com/modelscope/modelscope_hub/commits/main)

[modelscope.cn](https://modelscope.cn) | [modelscope.ai](https://modelscope.ai)

</div>

## Why modelscope-hub?

`modelscope-hub` connects your code to the [ModelScope](https://modelscope.cn) ecosystem — models, datasets, Studio spaces, skills, and MCP servers — through a single `HubApi` class or the `ms-hub` CLI.

- **Unified repo interface** — one set of methods for models, datasets, studios, skills, and MCP servers
- **OpenAPI-first** — built on the ModelScope OpenAPI surface with transparent legacy fallback
- **Production-grade downloads** — HTTP Range resume, parallel range download for large files, per-file retry with backoff, SHA256 integrity checks, file lock for multiprocess safety, offline mode, progress callbacks, and intra-cloud acceleration
- **Full lifecycle CLI** — download, upload, deploy, manage secrets, inspect cache — all from the terminal
- **Deep ecosystem integration** — seamless access to 100K+ models and datasets on ModelScope Hub; works with the `modelscope` training framework, Studio deployment platform, and MCP server infrastructure

---

## News

**v0.4.0** (2026-09-01)
- **Feature**: full MCP/Studios OpenAPI coverage: Studio lists, variables and configuration options; hosted MCP discovery; protected visibility and runtime metadata; read-only tokens can log in and rejected writes name the required tier
- **Fix**: Studio compat calls no longer leak connection options or API tokens, or drop cover images; errors distinguish permission, quota and conflicts; anonymous Studio info and log pagination work
- **Enhance**: MCP verb negotiation and Studio-owner listings adapt to endpoint behaviour
- **Quality**: the vendored OpenAPI spec and operation registry flag unimplemented published MCP/Studios endpoints

**v0.3.1** (2026-09-01)
- **Refactor**: upload environment variables now state their units and semantics; deprecated names remain supported with warnings
- **Fix**: logout works; download locks avoid same-basename contention; `whoami` and legacy responses normalise pre-production field names

**v0.3.0** (2026-08-19)
- **Feature**: this package owns all four console scripts (`modelscope`, `ms`, `modelscope-hub`, `ms-hub`), adds `HubApi.get_current_username()`, and supports both agent visibility wire formats
- **Fix**: large `repo/files` listings are re-enumerated when the server silently truncates results

<details>
<summary>Older releases</summary>

**v0.2.0** (2026-08-01)
- **Breaking**: `HubApi.login()` now raises `NetworkError` / `ServerError` / `RequestTimeoutError` for non-authentication failures instead of always raising `AuthenticationError` — widen `except AuthenticationError` to `HubError` if you catch *any* login failure
- **Fix**: network, timeout and 5xx errors during login are no longer misreported as an invalid token; a failed login no longer deletes stored credentials; bare and upper-case endpoints are accepted (e.g. `--endpoint modelscope.ai`)
- **Enhancement**: a rejected login probes the peer ModelScope site and suggests `--endpoint` when the token is valid there; `--verbose` prints the full error cause chain

**v0.1.9** (2026-07-31)
- **Fix**: two stale unit tests that failed in downstream distro sandboxes ([#46](https://github.com/modelscope/modelscope_hub/issues/46), NixOS): align the `mcp deploy` call-shape assertion and the `parse_timestamp` timezone-normalization contract
- **Feature**: `ms-hub mcp deploy` prints the operational URL after deployment; new `--auth-check` and repeatable `--env KEY=VALUE` options; `--transport-type` now validates against `sse`/`streamable_http`
- **CI**: new `citest` workflow — mock-mode test suite (no credentials/network, mirrors distro build sandboxes) on Python 3.10/3.12/3.14 plus ruff/mypy hard gates; releases now require a green test gate before publishing; added a `pre-commit` config
- **Quality**: ruff & mypy debt cleared to zero; lint/type-check targets aligned to the supported Python floor (3.10)

**v0.1.8** (2026-07-21)
- **Feature**: `ms-hub agent` raw file transfer (download/upload/list) for remote agent repos; visibility support for agent hub; cache checksum verification (`ms-hub cache verify`)
- **Fix**: forward `progress_callbacks` through `HubApi.download_repo` so custom download-progress callbacks work end-to-end; harden legacy (pre-1.38) cache auto-detection (reuse existing `{cache}/models/...` and default `{cache}/hub/models/...` layouts); normal (non-LFS) file upload
- **Packaging**: rename console scripts to `modelscope-hub` / `ms-hub` to avoid a file conflict with the `modelscope` package (e.g. FreeBSD pkg)

**v0.1.7** (2026-07-07)
- **Feature**: intra-/inter-region cloud download acceleration, with a source marker in the progress bar
- **Fix**: align `snapshot_download` cache path with the CLI; add legacy cache fallback
- **Refactor**: inter-region config via env var only (removed the `--inter-regions` CLI arg); cache the region probe

**v0.1.6** (2026-07-03)
- **Refactor**: replace the extra-field whitelist with a reserved-field blocklist for more permissive param passthrough

**v0.1.5** (2026-06-30)
- **Fix**: adaptive commit batch size for uploads

**v0.1.4** (2026-06-26)
- **Feature**: `gated_mode` parameter for `create_repo`; `ms-hub create --gated/--no-gated` flags
- **Refactor**: unify visibility / gated_mode semantics in the SDK layer
- **Fix**: `create_repo` extra-kwargs whitelist + type validation; correct visibility mapping (`private` bool is authoritative)

**v0.1.3** (2026-06-23)
- **Feature**: add `AlreadyExistsError` (E3026) and fix the `exist_ok` mechanism; align `list_repos`/`RepoInfo` with the OpenAPI response format
- **Fix**: `clear-cache` supports all cache layouts (standard/flat/legacy); add `last_modified` mapping and `to_dict()` for `RepoInfo`/`PagedResult`

**v0.1.2** (2026-06-23)
- **Fix**: unify `list_datasets`/`get_dataset` return format and align parameters

**v0.1.1** (2026-06-22)
- **Fix**: legacy API for msdatasets loading

**v0.1.0** (2026-06-18)
- **Feature**: Configurable upload failure thresholds (consecutive failures & total wait time)
- **Fix**: compatibility && error handling
  - Legacy cache path compatibility — compat layer preserves flat `{cache_dir}/{owner}/{name}/` layout
  - Type safety in legacy path resolution (handle both `str` and `Path` config values)
  - Improved error handling and response normalization in compatibility APIs

**v0.0.9** (2026-06-12)
- **Feature**: `get_model` support `revision`; expanded param passthrough for repo/model ops
- **Fix**: Pattern normalization accepts iterable inputs (tuple, etc.); `parse_timestamp` robust timezone conversion for ISO 8601, floats, milliseconds

**v0.0.8** (2026-06-10)
- **Feature**: `ms-hub list --all` auto-pagination; `ms-hub create --skill-file` zip upload; `ms-hub list --envs`
- **Fix**: Download per-file lock & stale detection & atomic merge; `--disable-tqdm` for folder upload
- **Security**: Redact tokens from git/API error output
- **Refactor**: Centralize env var registry; unify `MODELSCOPE_DOMAIN` → `MODELSCOPE_ENDPOINT`

**v0.0.5** (2026-06-05)
- Fix `list_repos` pagination and dataset visibility issues
- OpenAPI spec alignment: pagination limits, retry, auth, request body

**v0.0.4** (2026-06-05)
- Flatten CLI to top-level commands (`ms-hub create/info/list/delete`)
- Migrate credentials to `~/.modelscope/credentials/`
- Fix dataset/skill download, blob upload auth, error code refactor

</details>

---

## Installation

```bash
pip install modelscope-hub
```

Requires Python 3.10+. Lightweight — only `requests`, `tqdm`, `filelock`, `urllib3`.

### Command names

This package installs every ModelScope console script, and all four are the same
program — pick whichever reads best in your shell:

| Command | Use it for |
| --- | --- |
| `modelscope` | the primary, brand-level command |
| `ms` | short form of `modelscope` |
| `modelscope-hub` | explicit hub-only invocation |
| `ms-hub` | short form of `modelscope-hub` |

Owning all four in one distribution is deliberate: when the umbrella
[`modelscope`](https://github.com/modelscope/modelscope) SDK also declared
`modelscope` / `ms`, installing or upgrading either package could overwrite or
delete the other's scripts and leave you with no CLI at all. OS packagers (such
as FreeBSD pkg) likewise refuse to let two ports own the same path. The SDK now
contributes its extra commands as plugins instead.

So `modelscope` and `ms` work on a hub-only install too. Commands that belong to
the full SDK (`pipeline`, `server`, `studio`, `modelcard`, …) appear once you
also `pip install modelscope`; until then `--help` says so. The two `*-hub`
aliases always report just this package, which is handy for scripts that must
pin the lightweight client:

```bash
ms-hub --version      # modelscope-hub X.Y.Z
modelscope --version  # modelscope A.B.C (modelscope-hub X.Y.Z)
```

---

## Quick Start

### Authenticate

```bash
ms-hub login
# or pass a token directly
ms-hub login --token $MODELSCOPE_API_TOKEN
```

Get your token at [modelscope.cn/my/access/token](https://modelscope.cn/my/access/token) or [modelscope.ai/my/access/token](https://modelscope.ai/my/access/token).

```python
from modelscope_hub import HubApi

api = HubApi(token="your-token")
user = api.whoami()
print(user.username)
```

### Download

```bash
# Full snapshot
ms-hub download Qwen/Qwen3-0.6B

# Single file
ms-hub download Qwen/Qwen3-0.6B config.json

# With filters
ms-hub download Qwen/Qwen3-0.6B --include "*.safetensors" --exclude "*.bin"

# Directly into a local directory (bypasses cache)
ms-hub download Qwen/Qwen3-0.6B --local-dir ./my-model
```

```python
path = api.download_file("Qwen/Qwen3-0.6B", "model", "config.json")

snapshot = api.download_repo(
    "Qwen/Qwen3-0.6B", "model",
    allow_patterns=["*.safetensors", "*.json"],
    max_workers=8,
)

# Offline mode — return cached path without network access
path = api.download_file("Qwen/Qwen3-0.6B", "model", "config.json", local_files_only=True)
```

### Upload

```bash
ms-hub upload my-org/my-model ./weights.safetensors
ms-hub upload my-org/my-model ./output --repo-type model --commit-message "add weights"
```

```python
api.upload_file("my-org/my-model", "model", "./weights.safetensors", "weights.safetensors")
api.upload_folder("my-org/my-model", "model", "./output", path_in_repo="")
```

### Create a Repository

```bash
ms-hub create my-org/my-model --repo-type model --visibility private
```

```python
api.create_repo("my-org/my-model", "model", visibility="private", license="apache-2.0")
```

### Deploy a Studio

```bash
ms-hub deploy my-org/chat-demo --repo-type studio
ms-hub logs my-org/chat-demo --log-type run
ms-hub stop my-org/chat-demo --repo-type studio
```

```python
api.deploy_repo("my-org/chat-demo", "studio")
api.get_repo_logs("my-org/chat-demo", log_type="run")
api.stop_repo("my-org/chat-demo", "studio")
```

---

## CLI Reference

The CLI is available as both `ms-hub` and `modelscope-hub`.

**Global options** (placed before or after the subcommand):

| Option | Description |
|--------|-------------|
| `--token TOKEN` | API token (overrides env and persisted token) |
| `--endpoint URL` | API endpoint (default: `https://modelscope.cn`) |
| `-v, --verbose` | Enable DEBUG logging (global only) |
| `-V, --version` | Print version and exit (global only) |

> `--token` and `--endpoint` can be placed either before or after the subcommand:
> `ms-hub --token xxx download ...` and `ms-hub download ... --token xxx` are equivalent.

### `ms-hub login`

Authenticate and persist your token locally.

```bash
ms-hub login                          # interactive prompt
ms-hub login --token $MY_TOKEN        # non-interactive
```

| Option | Description |
|--------|-------------|
| `--token TOKEN` | API token; prompted interactively if omitted |

#### Token permission levels

Tokens are issued with one of three permission levels, set when you create the
token at [modelscope.cn/my/myaccesstoken](https://modelscope.cn/my/myaccesstoken):

| Level | Can do |
|-------|--------|
| `read` | Browse and download; list your Studios, secrets and variables |
| `write` | Everything above, plus create/upload/deploy and change settings |
| `admin` | Everything above, plus administrative operations |

What this means in practice:

- **A read-only token can log in.** `ms-hub login` succeeds and prints a warning
  that git and session credentials were not issued, so pushes and uploads will
  need a `write` token. Downloads and browsing work normally.
- **A rejected write is explicit.** Attempting a write with a read-only token
  fails with `[E3002] Permission denied` and a suggestion naming the level the
  operation requires, rather than leaving you to guess.
- **Quota exhaustion is not a permission problem.** The service reports both as
  HTTP 403; the SDK separates them into `PermissionDeniedError` and
  `QuotaExceededError` (the latter is never retried, since retrying an exhausted
  quota only delays the error).
- **Public reads never fail because of a token.** For public endpoints the SDK
  retries once without credentials, so a read-scoped or stale token cannot hide
  data any anonymous caller can see.

> The SDK cannot tell you which level *your* token has: the API does not report
> it. Check the token settings page if you are unsure.

### `ms-hub whoami`

Show the user associated with the current token.

```bash
ms-hub whoami
ms-hub whoami --token $MY_TOKEN   # check a specific token without logging in
```

### `ms-hub download`

Download a single file or a full repository snapshot.

```bash
ms-hub download Qwen/Qwen3-0.6B                                  # full snapshot
ms-hub download Qwen/Qwen3-0.6B config.json                      # single file
ms-hub download Qwen/Qwen3-0.6B --include "*.safetensors"         # filter by glob
ms-hub download Qwen/Qwen3-0.6B --local-dir ./out --max-workers 8 # direct download
ms-hub download my-org/my-data --repo-type dataset --revision v2   # dataset at tag
```

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `repo_id` | yes | Repository identifier (`owner/name`) |
| `files...` | no | Specific file paths; omit for full snapshot |
| `--repo-type {model,dataset}` | no | Default: `model` |
| `--revision REV` | no | Branch, tag, or commit hash (default: `master`) |
| `--local-dir DIR` | no | Download directly here (bypasses cache layout) |
| `--cache-dir DIR` | no | Override default cache directory |
| `--include GLOB...` | no | Only download matching files; repeatable |
| `--exclude GLOB...` | no | Skip matching files; repeatable |
| `--max-workers N` | no | Parallel download threads (default: `4`) |
| `--force` | no | Re-download even if cached |

<details>
<summary>Advanced examples</summary>

```bash
# Download multiple specific files at once
ms-hub download Qwen/Qwen3-0.6B config.json tokenizer.json generation_config.json

# Download only safetensors, skip GGUF and bin weights
ms-hub download Qwen/Qwen3-0.6B --include "*.safetensors" --exclude "*.bin" "*.gguf"

# Download a dataset at a specific tag into a local directory
ms-hub download my-org/my-data --repo-type dataset --revision v2 --local-dir ./data

# Use a custom cache directory and 8 parallel threads
ms-hub download Qwen/Qwen3-0.6B --cache-dir /data/hub-cache --max-workers 8

# Force re-download even if already cached
ms-hub download Qwen/Qwen3-0.6B config.json --force

# Download all skills from a collection (legacy flag)
ms-hub download --collection my-org/skill-collection

# Enable parallel range download for large files (env var)
MODELSCOPE_DOWNLOAD_PARALLELS=4 ms-hub download Qwen/Qwen3-0.6B

# Use the modelscope.ai endpoint (global option, before subcommand)
ms-hub --endpoint https://modelscope.ai download Qwen/Qwen3-0.6B
```

</details>

### `ms-hub upload`

Upload a file or folder to a repository.

```bash
ms-hub upload my-org/my-model ./weights.safetensors                       # single file
ms-hub upload my-org/my-model ./output models/ --repo-type model          # folder → subdir
ms-hub upload my-org/my-model . --include "*.py" --commit-message "code"  # filtered folder
```

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `repo_id` | yes | Repository identifier |
| `local_path` | no | Local file or folder (default: inferred from repo name) |
| `path_in_repo` | no | Destination path inside the repo |
| `--repo-type {model,dataset}` | no | Default: `model` |
| `--revision REV` | no | Target branch (default: `master`) |
| `--commit-message MSG` | no | Commit message |
| `--commit-description DESC` | no | Extended commit description |
| `--include GLOB...` | no | Include filter for folder mode; repeatable |
| `--exclude GLOB...` | no | Exclude filter for folder mode; repeatable |
| `--max-workers N` | no | Parallel upload threads |
| `--use-cache / --no-cache` | no | Enable/disable resumable upload cache (default: on) |
| `--disable-tqdm` | no | Disable progress bars |

<details>
<summary>Advanced examples</summary>

```bash
# Upload a single file with a custom commit message
ms-hub upload my-org/my-model ./weights.safetensors --commit-message "add fp16 weights"

# Upload a folder into a subdirectory of the repo
ms-hub upload my-org/my-model ./output models/ --repo-type model

# Upload only Python files from the current directory
ms-hub upload my-org/my-model . --include "*.py" --commit-message "update code"

# Upload only safetensors, skip checkpoints
ms-hub upload my-org/my-model ./output --include "*.safetensors" --exclude "*.ckpt" "*.bin"

# Upload to a dataset repo on a specific branch
ms-hub upload my-org/my-data ./data --repo-type dataset --revision dev

# Upload with extended commit description
ms-hub upload my-org/my-model ./weights.safetensors \
  --commit-message "v2 weights" \
  --commit-description "Retrained with extended dataset, 3 epochs, lr=2e-5"

# Resumable upload: interrupted uploads resume automatically via cache
ms-hub upload my-org/my-model ./large-folder
# If interrupted, just re-run the same command — already uploaded files are skipped

# Disable upload cache (no resume, fresh upload every time)
ms-hub upload my-org/my-model ./output --no-cache

# Disable progress bars (useful for CI/CD pipelines)
ms-hub upload my-org/my-model ./output --disable-tqdm
```

</details>

### `ms-hub create` / `ms-hub info` / `ms-hub list` / `ms-hub delete`

Repository management.

```bash
ms-hub create my-org/my-model --repo-type model --visibility private
ms-hub create my-org/demo --repo-type studio --sdk-type gradio
ms-hub info my-org/my-model --repo-type model
ms-hub list --repo-type model --owner my-org --page-size 20
ms-hub list --repo-type studio --owner my-org
ms-hub delete my-org/my-model --repo-type model --yes
```

> **Deprecation notice:** `delete_repo` / `ms-hub delete` emits a `DeprecationWarning` — programmatic repo deletion is restricted for security reasons and will be restored once token-scoped auth is available. Use the [web console](https://modelscope.cn) to delete repos.
>
> `delete_files` requires cookie-based session auth; API tokens may receive a 401 error.

<details>
<summary><code>ms-hub create</code> options</summary>

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `repo_id` | yes | Repository identifier |
| `--repo-type` | yes | `model`, `dataset`, `studio`, or `skill` |
| `--visibility` | no | `public`, `private` or `internal`. For Studios also `protected` (app public, code repository hidden) |
| `--license` | no | SPDX license identifier (e.g. `apache-2.0`) |
| `--chinese-name` | no | Display name in Chinese |
| `--description` | no | Repository description |
| `--exist-ok` | no | No error if repository already exists |
| `--sdk-type` | no | Studio SDK: `gradio`, `streamlit`, `docker`, `static` |
| `--sdk-version` | no | Studio SDK version |
| `--base-image` | no | Studio base Docker image |
| `--cover-image` | no | Studio cover image URL |
| `--hardware` | no | Studio hardware spec |

</details>

### `ms-hub deploy` / `ms-hub stop` / `ms-hub logs` / `ms-hub settings`

Manage Studio and MCP deployments.

```bash
ms-hub deploy my-org/chat-demo --repo-type studio
ms-hub logs my-org/chat-demo --log-type run --keyword ERROR --page-size 50
ms-hub settings my-org/chat-demo cpu=4 memory=8192
ms-hub stop my-org/chat-demo --repo-type studio
```

<details>
<summary>Options</summary>

| Command | `--repo-type` | Key Options |
|---------|---------------|-------------|
| `ms-hub deploy <repo_id>` | `{studio,mcp}` (default: `studio`) | — |
| `ms-hub stop <repo_id>` | `{studio,mcp}` (default: `studio`) | — |
| `ms-hub logs <repo_id>` | `{studio}` only | `--log-type {run,build}`, `--keyword`, `--page`, `--page-size` |
| `ms-hub settings <repo_id> key=val...` | `{studio,skill}` (default: `studio`) | Key-value pairs passed to backend |

> **Note:** `ms-hub logs` only supports Studio spaces. MCP server logs are not available via this command.
> `ms-hub settings` supports Studio and Skill repos; for MCP servers use `ms-hub mcp deploy` with configuration payload.

</details>

### `ms-hub secret` / `ms-hub studio variable`

Manage a Studio's environment variables. Two commands, because the disclosure
differs: a **secret**'s value is never returned by the API, a **variable**'s value
is publicly visible.

```bash
# Secrets — values are write-only
ms-hub secret add my-org/demo API_KEY sk-xxx
ms-hub secret list my-org/demo
ms-hub secret update my-org/demo API_KEY sk-new
ms-hub secret delete my-org/demo API_KEY --yes

# Plaintext variables — values are readable by anyone
ms-hub studio variable add my-org/demo MODEL_NAME Qwen2.5-7B
ms-hub studio variable list my-org/demo
ms-hub studio variable update my-org/demo MODEL_NAME Qwen2.5-14B
ms-hub studio variable delete my-org/demo MODEL_NAME --yes
```

<details>
<summary>Subcommands</summary>

| Subcommand | Arguments | Description |
|------------|-----------|-------------|
| `add` | `repo_id key value` | Add a new entry |
| `list` | `repo_id` | List entries (`secret` shows keys only; `variable` shows keys and values) |
| `update` | `repo_id key value` | Update a value |
| `delete` | `repo_id key [--yes]` | Delete an entry |

`ms-hub secret` accepts `--repo-type` (default: `studio`, currently the only supported type).

> **Put anything sensitive in a secret.** A plaintext variable's value is public.

</details>

### `ms-hub studio`

Browse Studio spaces and the runtime resources they can be configured with.

```bash
# Browse
ms-hub studio list --search chat --sort likes --page-size 20
ms-hub studio list --owner my-org --status all
ms-hub studio list --mcp-support --hardware-type xgpu

# Discover valid values for --hardware / --base-image / --sdk-version
ms-hub studio hardware --sdk-type gradio
ms-hub studio hardware --studio my-org/demo
ms-hub studio base-images
ms-hub studio sdk-versions --sdk-type gradio

# Runtime management (also available as the top-level deploy/stop/logs/settings)
ms-hub studio deploy my-org/demo
ms-hub studio logs my-org/demo --type build
ms-hub studio settings my-org/demo --visibility protected
```

<details>
<summary>Subcommands</summary>

| Subcommand | Arguments | Key Options |
|------------|-----------|-------------|
| `list` | — | `--search`, `--owner`, `--sort {default,last_modified,view_num,likes}`, `--status {running,all}`, `--mcp-support`/`--no-mcp-support`, `--hardware-type {xgpu,amd}`, `--page`, `--page-size` |
| `hardware` | — | `--sdk-type`, `--studio owner/name` |
| `base-images` | — | — |
| `sdk-versions` | — | `--sdk-type` (default `gradio`; only gradio publishes versions) |
| `deploy` / `stop` | `studio_id` | — |
| `logs` | `studio_id` | `--type {run,build}`, `--keyword`, `--page`, `--page-size` (max 500) |
| `settings` | `studio_id [key=val...]` | `--display-name`, `--description`, `--license`, `--cover-image`, `--sdk-type`, `--sdk-version`, `--base-image`, `--hardware`, `--visibility`, `--private`/`--public` |
| `secret` | see above | — |
| `variable` | see above | — |

`--visibility` accepts three values: `public` (code and app public), `protected`
(app public, code repository hidden) and `private`. `--private` / `--public`
remain as shorthands.

Paid hardware tiers are selected as `--hardware paid/<instance_type>`; run
`ms-hub studio hardware` to see the available identifiers.

`--status` defaults to `all` whenever `--owner` is given, so listing your own
spaces includes stopped ones. Without `--owner` the server's own default applies
(running spaces only).

</details>

### `ms-hub mcp`

Manage MCP (Model Context Protocol) servers.

```bash
ms-hub mcp list --search weather --page-size 10
ms-hub mcp list --hosted            # only your own hosted servers, with live URLs
ms-hub mcp info my-org/weather-mcp
ms-hub mcp deploy my-org/weather-mcp
ms-hub mcp undeploy my-org/weather-mcp
```

<details>
<summary>Subcommands</summary>

| Subcommand | Arguments | Key Options |
|------------|-----------|-------------|
| `list` | — | `--search`, `--hosted`, `--page`, `--page-size` |
| `info` | `server_id` | — |
| `deploy` | `server_id` | `--transport-type`, `--expiration-minutes`, `--auth-check`, `--env KEY=VALUE` |
| `undeploy` | `server_id` | — |

`--hosted` lists the servers you currently have hosted along with their
`operational_urls`. It takes no search or paging options, because the underlying
endpoint accepts none.

> Discovery is capped at `page * page_size <= 100` by the service.

</details>

### `ms-hub cache`

Inspect and clean the local download cache.

```bash
ms-hub cache scan
ms-hub cache scan --cache-dir /data/cache
ms-hub cache verify Qwen/Qwen3-0.6B
ms-hub cache verify Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
ms-hub cache clear --repo-type model --yes
ms-hub cache clear --repo-id my-org/old-model --repo-type model --yes
```

<details>
<summary>Options</summary>

| Subcommand | Key Options |
|------------|-------------|
| `scan` | `--cache-dir DIR` |
| `verify REPO_ID` | `--repo-type`, `--revision`, `--cache-dir`, `--local-dir`, `--fail-on-missing-files`, `--fail-on-extra-files` |
| `clear` | `--repo-type`, `--repo-id`, `--cache-dir`, `--yes` |

</details>

### `ms-hub agent`

Low-level raw file transfer for remote agent repositories: `download`, `upload`, `list`. This command transfers files as-is, with **no framework awareness**.

```bash
ms-hub agent download -r user/my-agent --local-dir ./my-agent   # download raw files
ms-hub agent upload   -r user/my-agent --local-dir ./my-agent   # upload raw
```

> **Framework-aware operations** (cross-framework `convert`, `watch`/bidirectional sync, `status`, `backups`, `restore`, `stop`) live in **[modelscope-agent](https://github.com/modelscope/ms-agent)** — use `ms-agent agent ...` instead. For example, to download and convert in one step: `ms-agent agent download -f qoder -r user/my-agent --target-framework qwenpaw`.

<details>
<summary>Subcommands</summary>

#### `ms-hub agent download`

Download all files of a remote agent repository to a local directory (raw, no conversion).

```bash
ms-hub agent download -r user/my-agent
ms-hub agent download -r user/my-agent --local-dir ./my-agent --revision master
```

| Option | Required | Description |
|--------|----------|-------------|
| `-r, --repo REPO` | yes | Remote repo identifier (`owner/name`) |
| `--local-dir DIR` | no | Destination directory (default: `./<repo-name>` under CWD) |
| `--revision REV` | no | Repository revision (default: `master`) |

#### `ms-hub agent upload`

Upload files from a local path (file or directory) to a remote agent repository (raw, no conversion). Creates the repo if it does not exist.

```bash
ms-hub agent upload -r user/my-agent --local-dir ./my-agent
ms-hub agent upload -r user/my-agent --local-dir ./my-agent --dry-run
```

| Option | Required | Description |
|--------|----------|-------------|
| `-r, --repo REPO` | yes | Remote repo identifier (`owner/name`) |
| `--local-dir DIR` | no | Source path (file or directory) to upload (default: CWD) |
| `--revision REV` | no | Repository revision (default: `master`) |
| `--dry-run` | no | List files that would be uploaded without uploading |

</details>

---

## SDK API Overview

All operations go through a single entry point:

```python
from modelscope_hub import HubApi

# Connect to modelscope.cn (default)
api = HubApi(token="...")

# Or connect to modelscope.ai
api = HubApi(token="...", endpoint="https://modelscope.ai")
```

<details>
<summary><strong>Full method reference</strong></summary>

| Category | Method | Description |
|----------|--------|-------------|
| **Auth** | `login(token)` | Persist and verify token |
| | `logout()` | Clear stored credentials |
| | `whoami()` | Get current user info |
| **Repo** | `create_repo(repo_id, repo_type, ...)` | Create a repository |
| | `get_repo(repo_id, repo_type)` | Get repository metadata |
| | `list_repos(repo_type, ...)` | Paginated listing (`model`, `dataset`, `studio`, `skill`, `mcp`) |
| | `delete_repo(repo_id, repo_type)` | Delete a repository *(deprecated — see note below)* |
| | `repo_exists(repo_id, repo_type)` | Check existence |
| **Files** | `upload_file(repo_id, repo_type, local, remote)` | Upload a single file |
| | `upload_folder(repo_id, repo_type, folder, ...)` | Upload a directory |
| | `download_file(repo_id, repo_type, file, ...)` | Download a single file (with retry, resume, offline mode) |
| | `download_repo(repo_id, repo_type, ...)` | Download full snapshot (parallel, file lock, progress callbacks) |
| | `list_repo_files(repo_id, repo_type)` | List files in a repo |
| | `delete_files(repo_id, repo_type, paths)` | Remove files *(cookie-auth only)* |
| **Version** | `list_repo_revisions(repo_id, repo_type)` | List branches and tags |
| | `create_repo_tag(repo_id, repo_type, tag)` | Create a tag |
| **Deploy** | `deploy_repo(repo_id, repo_type)` | Deploy Studio or MCP |
| | `stop_repo(repo_id, repo_type)` | Stop deployment |
| | `get_repo_logs(repo_id, ...)` | Fetch logs |
| | `update_repo_settings(repo_id, repo_type, ...)` | Update settings |
| **Secrets** | `add_secret(repo_id, key, value)` | Add a secret |
| | `list_secrets(repo_id)` | List secret keys (values are never returned) |
| | `update_secret(repo_id, key, value)` | Update a secret |
| | `delete_secret(repo_id, key)` | Delete a secret |
| **Variables** | `add_variable(repo_id, key, value)` | Add a plaintext variable *(value is public)* |
| | `list_variables(repo_id)` | List variables with their values |
| | `update_variable(repo_id, key, value)` | Update a variable |
| | `delete_variable(repo_id, key)` | Delete a variable |
| **Studio resources** | `list_studio_hardware(...)` | Hardware tiers a Studio can run on |
| | `list_studio_base_images()` | Available base images |
| | `list_studio_sdk_versions(...)` | Available SDK versions |
| **MCP** | `list_mcp_servers(...)` | List available MCP servers |
| | `list_operational_mcp_servers()` | List your hosted servers, with live URLs |
| | `get_mcp_server(server_id)` | Get server details |
| | `deploy_mcp_server(server_id)` | Deploy an MCP server |
| | `undeploy_mcp_server(server_id)` | Undeploy an MCP server |
| **Cache** | `scan_cache(cache_dir)` | Inspect local cache |
| | `clear_cache(cache_dir, ...)` | Free disk space |

> **Deletion restrictions:**
> - `delete_repo` is deprecated for security reasons (emits `DeprecationWarning`). Will be restored with token-scoped auth. Use the [web console](https://modelscope.cn) instead.
> - `delete_files` requires cookie-based session auth; API tokens (`ms-...`) may receive a 401 error.

> **Token permission levels:** every write method needs a token issued with
> `write` permission or higher. A rejected write raises `PermissionDeniedError`
> whose `suggestion` names the required level; an exhausted quota raises
> `QuotaExceededError` instead. See [Token permission levels](#token-permission-levels).

</details>

---

## Ecosystem Integration

`modelscope-hub` is the hub connectivity layer for the ModelScope ecosystem:

```
┌────────────────────────────────────────────────┐
│              ModelScope Platform                │
│   modelscope.cn  ·  modelscope.ai              │
│                                                │
│  Models · Datasets · Studios · Skills · MCP    │
└───────────────────┬────────────────────────────┘
                    │  OpenAPI / Legacy API
                    ▼
            ┌───────────────┐
            │ modelscope-hub│  ← this library
            │  SDK  +  CLI  │
            └───┬───────┬───┘
                │       │
        ┌───────┘       └────────┐
        ▼                        ▼
  modelscope framework    your application
  (training · eval)       (inference · deploy)
```

- **Browse & discover** — search 100K+ models and datasets via `list_repos` / `ms-hub repo list`
- **Download & cache** — pull model weights, tokenizer configs, or entire datasets into a managed cache or a local directory; supports offline mode via `local_files_only`
- **Train & fine-tune** — use with the [modelscope](https://github.com/modelscope/modelscope) framework: train locally, then push results back
- **Deploy** — launch a Studio space or MCP server directly from the CLI or SDK
- **Automate** — integrate into CI/CD pipelines with environment-variable auth and `--yes` flags for non-interactive operation

---

## Configuration

Run `ms-hub list --envs` to see all configurable environment variables with their current values.

Token is persisted locally after `ms-hub login` and auto-loaded in subsequent sessions.

<details>
<summary>Environment variables</summary>

**Core:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_API_TOKEN` | — | API authentication token |
| `MODELSCOPE_ENDPOINT` | `https://modelscope.cn` | API endpoint URL |
| `MODELSCOPE_CACHE` | `~/.cache/modelscope` | Local cache directory |
| `MODELSCOPE_HOME` | `~/.modelscope` | SDK config directory |
| `MODELSCOPE_PREFER_AI_SITE` | `false` | Prefer `modelscope.ai` over `modelscope.cn` |

**Network:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_API_TIMEOUT` | `60` | HTTP request timeout (seconds) |
| `MODELSCOPE_API_CONNECT_TIMEOUT` | `10` | HTTP connect timeout (seconds) |
| `MODELSCOPE_API_MAX_RETRIES` | `5` | Max retry attempts for transient failures |

**Download:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS` | `1` | Parallel range-download streams |
| `MODELSCOPE_DOWNLOAD_PARALLEL_THRESHOLD_MB` | `500` | Parallel download threshold (MB) |
| `MODELSCOPE_DOWNLOAD_CHUNK_SIZE_MB` | `1` | Streaming chunk size (MB) |
| `MODELSCOPE_DOWNLOAD_PART_SIZE_MB` | `160` | Parallel range chunk size (MB) |
| `MODELSCOPE_DOWNLOAD_MAX_RETRIES` | `5` | Per-file download retry count |
| `MODELSCOPE_DOWNLOAD_TIMEOUT` | `60` | Per-file download timeout (seconds) |
| `MODELSCOPE_DOWNLOAD_FILE_LOCK` | `true` | File lock for multiprocess download safety |
| `MODELSCOPE_DOWNLOAD_INTRA_CLOUD` | `true` | Alibaba cloud intra-cloud acceleration |
| `MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION` | (auto) | Override intra-cloud region ID |
| `MODELSCOPE_DOWNLOAD_INTER_CLOUD_REGIONS` | | Comma-separated peer regions for cross-region internal acceleration (e.g. `cn-hangzhou,cn-zhangjiakou`) |

**Upload:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_UPLOAD_MAX_CONCURRENT_WORKERS` | `min(8, cpu+4)` | Default parallel worker threads |
| `MODELSCOPE_UPLOAD_CACHE_ENABLED` | `true` | Enable resumable upload cache |
| `MODELSCOPE_UPLOAD_IGNORE_FILE_PATTERN` | — | File pattern excluded by legacy `push_to_hub` uploads |
| `MODELSCOPE_UPLOAD_MAX_FILE_SIZE_MB` | `102400` | Max single file size (MB, default 100 GB) |
| `MODELSCOPE_UPLOAD_MAX_FILE_COUNT` | `100000` | Max total files per upload |
| `MODELSCOPE_UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS` | `30` | Blob upload connect timeout (seconds) |
| `MODELSCOPE_UPLOAD_BLOB_READ_TIMEOUT_SECONDS` | `3600` | Blob upload read timeout (seconds) |

**Logging:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELSCOPE_LOG_LEVEL` | `INFO` | SDK log level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |
| `MODELSCOPE_NO_DEPRECATION_WARNINGS` | — | Suppress deprecation warnings |

> Old variable names (e.g. `API_TIMEOUT`, `DOWNLOAD_RETRY_TIMES`, `UPLOAD_USE_CACHE`) are
> deprecated and remain temporarily supported. They emit a `FutureWarning` and will be removed
> in a future version. Run `ms-hub list --envs` to see which deprecated names are active in your
> environment.

</details>

---

## Backward Compatibility

`modelscope-hub` provides a compatibility layer for code written against the old `modelscope.hub` API surface. The old SDK can delegate directly to `modelscope_hub.compat`:

```python
from modelscope_hub.compat import snapshot_download, model_file_download
from modelscope_hub.compat import LegacyHubApi as HubApi
```

All legacy parameter names (`allow_file_pattern`, `ignore_file_pattern`, `cookies`, etc.) are accepted and mapped to the new implementation.

---

## Development

```bash
git clone https://github.com/modelscope/modelscope_hub.git
cd modelscope_hub
make install    # pip install -e ".[dev]"
make test       # unit tests (no network)
make lint       # ruff check
make typecheck  # mypy
```

See `make help` for all available targets.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@Misc{modelscope-hub,
  title = {modelscope-hub: The official Python client to connect with ModelScope Hub.},
  author = {The ModelScope Team},
  howpublished = {\url{https://github.com/modelscope/modelscope_hub}},
  year = {2026}
}
```
