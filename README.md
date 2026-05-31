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

`modelscope-hub` connects your code to the [ModelScope](https://modelscope.cn) ecosystem — models, datasets, Studio spaces, skills, and MCP servers — through a single `HubApi` class or the `ms` CLI.

- **Unified repo interface** — one set of methods for models, datasets, studios, skills, and MCP servers
- **OpenAPI-first** — built on the ModelScope OpenAPI surface with transparent legacy fallback
- **Resumable downloads** — HTTP Range resume, parallel threads, SHA256 integrity checks
- **Full lifecycle CLI** — download, upload, deploy, manage secrets, inspect cache — all from the terminal
- **Deep ecosystem integration** — seamless access to 100K+ models and datasets on ModelScope Hub; works with the `modelscope` training framework, Studio deployment platform, and MCP server infrastructure

---

## Installation

```bash
pip install modelscope-hub
```

Requires Python 3.10+. Lightweight — only `requests`, `tqdm`, `filelock`, `urllib3`.

---

## Quick Start

### Authenticate

```bash
ms login
# or pass a token directly
ms login --token $MODELSCOPE_API_TOKEN
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
ms download Qwen/Qwen3-0.6B

# Single file
ms download Qwen/Qwen3-0.6B config.json

# With filters
ms download Qwen/Qwen3-0.6B --include "*.safetensors" --exclude "*.bin"

# Directly into a local directory (bypasses cache)
ms download Qwen/Qwen3-0.6B --local-dir ./my-model
```

```python
path = api.download_file("Qwen/Qwen3-0.6B", "model", "config.json")

snapshot = api.download_repo(
    "Qwen/Qwen3-0.6B", "model",
    allow_patterns=["*.safetensors", "*.json"],
    max_workers=8,
)
```

### Upload

```bash
ms upload my-org/my-model ./weights.safetensors
ms upload my-org/my-model ./output --repo-type model --commit-message "add weights"
```

```python
api.upload_file("my-org/my-model", "model", "./weights.safetensors", "weights.safetensors")
api.upload_folder("my-org/my-model", "model", "./output", path_in_repo="")
```

### Create a Repository

```bash
ms repo create my-org/my-model --repo-type model --visibility private
```

```python
api.create_repo("my-org/my-model", "model", visibility="private", license="apache-2.0")
```

### Deploy a Studio

```bash
ms deploy my-org/chat-demo --repo-type studio
ms logs my-org/chat-demo --log-type runtime
ms stop my-org/chat-demo --repo-type studio
```

```python
api.deploy_repo("my-org/chat-demo", "studio")
api.get_repo_logs("my-org/chat-demo", log_type="runtime")
api.stop_repo("my-org/chat-demo", "studio")
```

---

## CLI Reference

The CLI is available as both `ms` and `modelscope`.

**Global options** (placed before the subcommand):

| Option | Description |
|--------|-------------|
| `--token TOKEN` | API token (overrides env and persisted token) |
| `--endpoint URL` | API endpoint (default: `https://modelscope.cn`) |
| `-v, --verbose` | Enable DEBUG logging |
| `-V, --version` | Print version and exit |

### `ms login`

Authenticate and persist your token locally.

```bash
ms login                          # interactive prompt
ms login --token $MY_TOKEN        # non-interactive
```

| Option | Description |
|--------|-------------|
| `--token TOKEN` | API token; prompted interactively if omitted |

### `ms whoami`

Show the user associated with the current token.

```bash
ms whoami
```

### `ms download`

Download a single file or a full repository snapshot.

```bash
ms download Qwen/Qwen3-0.6B                                  # full snapshot
ms download Qwen/Qwen3-0.6B config.json                      # single file
ms download Qwen/Qwen3-0.6B --include "*.safetensors"         # filter by glob
ms download Qwen/Qwen3-0.6B --local-dir ./out --max-workers 8 # direct download
ms download my-org/my-data --repo-type dataset --revision v2   # dataset at tag
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

### `ms upload`

Upload a file or folder to a repository.

```bash
ms upload my-org/my-model ./weights.safetensors                       # single file
ms upload my-org/my-model ./output models/ --repo-type model          # folder → subdir
ms upload my-org/my-model . --include "*.py" --commit-message "code"  # filtered folder
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

### `ms repo`

Repository management (create, info, list, delete).

```bash
ms repo create my-org/my-model --repo-type model --visibility private
ms repo create my-org/demo --repo-type studio --sdk-type gradio
ms repo info my-org/my-model --repo-type model
ms repo list --repo-type model --owner my-org --page-size 20
ms repo delete my-org/my-model --repo-type model --yes
```

<details>
<summary><code>ms repo create</code> options</summary>

| Argument / Option | Required | Description |
|-------------------|----------|-------------|
| `repo_id` | yes | Repository identifier |
| `--repo-type` | yes | `model`, `dataset`, `studio`, `skill`, or `mcp` |
| `--visibility` | no | `public`, `private`, or `internal` |
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

### `ms deploy` / `ms stop` / `ms logs` / `ms settings`

Manage Studio and MCP deployments.

```bash
ms deploy my-org/chat-demo --repo-type studio
ms logs my-org/chat-demo --log-type runtime --keyword ERROR --page-size 50
ms settings my-org/chat-demo cpu=4 memory=8192
ms stop my-org/chat-demo --repo-type studio
```

<details>
<summary>Options</summary>

| Command | Key Options |
|---------|-------------|
| `ms deploy <repo_id>` | `--repo-type {studio,mcp}` |
| `ms stop <repo_id>` | `--repo-type {studio,mcp}` |
| `ms logs <repo_id>` | `--log-type {runtime,build}`, `--keyword`, `--page`, `--page-size` |
| `ms settings <repo_id> key=val...` | Key-value pairs passed to backend |

</details>

### `ms secret`

Manage secrets for Studio spaces.

```bash
ms secret add my-org/demo API_KEY sk-xxx
ms secret list my-org/demo
ms secret update my-org/demo API_KEY sk-new
ms secret delete my-org/demo API_KEY --yes
```

<details>
<summary>Subcommands</summary>

| Subcommand | Arguments | Description |
|------------|-----------|-------------|
| `add` | `repo_id key value` | Add a new secret |
| `list` | `repo_id` | List all secret keys |
| `update` | `repo_id key value` | Update a secret value |
| `delete` | `repo_id key [--yes]` | Delete a secret |

</details>

### `ms mcp`

Manage MCP (Model Context Protocol) servers.

```bash
ms mcp list --search weather --page-size 10
ms mcp info my-org/weather-mcp
ms mcp deploy my-org/weather-mcp
ms mcp undeploy my-org/weather-mcp
```

<details>
<summary>Subcommands</summary>

| Subcommand | Arguments | Key Options |
|------------|-----------|-------------|
| `list` | — | `--search`, `--page`, `--page-size` |
| `info` | `server_id` | — |
| `deploy` | `server_id` | — |
| `undeploy` | `server_id` | — |

</details>

### `ms cache`

Inspect and clean the local download cache.

```bash
ms cache scan
ms cache scan --cache-dir /data/cache
ms cache clear --repo-type model --yes
ms cache clear --repo-id my-org/old-model --repo-type model --yes
```

<details>
<summary>Options</summary>

| Subcommand | Key Options |
|------------|-------------|
| `scan` | `--cache-dir DIR` |
| `clear` | `--repo-type`, `--repo-id`, `--cache-dir`, `--yes` |

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
| | `list_repos(repo_type, ...)` | Paginated listing |
| | `delete_repo(repo_id, repo_type)` | Delete a repository |
| | `repo_exists(repo_id, repo_type)` | Check existence |
| **Files** | `upload_file(repo_id, repo_type, local, remote)` | Upload a single file |
| | `upload_folder(repo_id, repo_type, folder, ...)` | Upload a directory |
| | `download_file(repo_id, repo_type, file, ...)` | Download a single file |
| | `download_repo(repo_id, repo_type, ...)` | Download full snapshot |
| | `list_repo_files(repo_id, repo_type)` | List files in a repo |
| | `delete_files(repo_id, repo_type, paths)` | Remove files |
| **Version** | `list_repo_revisions(repo_id, repo_type)` | List branches and tags |
| | `create_repo_tag(repo_id, repo_type, tag)` | Create a tag |
| **Deploy** | `deploy_repo(repo_id, repo_type)` | Deploy Studio or MCP |
| | `stop_repo(repo_id, repo_type)` | Stop deployment |
| | `get_repo_logs(repo_id, ...)` | Fetch logs |
| | `update_repo_settings(repo_id, repo_type, ...)` | Update settings |
| **Secrets** | `add_secret(repo_id, key, value)` | Add a secret |
| | `list_secrets(repo_id)` | List secrets |
| | `update_secret(repo_id, key, value)` | Update a secret |
| | `delete_secret(repo_id, key)` | Delete a secret |
| **MCP** | `list_mcp_servers(...)` | List available MCP servers |
| | `get_mcp_server(server_id)` | Get server details |
| | `deploy_mcp_server(server_id)` | Deploy an MCP server |
| | `undeploy_mcp_server(server_id)` | Undeploy an MCP server |
| **Cache** | `scan_cache(cache_dir)` | Inspect local cache |
| | `clear_cache(cache_dir, ...)` | Free disk space |

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

- **Browse & discover** — search 100K+ models and datasets via `list_repos` / `ms repo list`
- **Download & cache** — pull model weights, tokenizer configs, or entire datasets into a managed cache or a local directory
- **Train & fine-tune** — use with the [modelscope](https://github.com/modelscope/modelscope) framework: train locally, then push results back
- **Deploy** — launch a Studio space or MCP server directly from the CLI or SDK
- **Automate** — integrate into CI/CD pipelines with environment-variable auth and `--yes` flags for non-interactive operation

---

## Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `MODELSCOPE_API_TOKEN` | Default API token |
| `MODELSCOPE_ENDPOINT` | API endpoint (default: `https://modelscope.cn`) |
| `MODELSCOPE_CACHE` | Override cache directory |

Token is persisted locally after `ms login` and auto-loaded in subsequent sessions.

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
