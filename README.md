<p align="center">
  <img src="https://modelscope.cn/static/favicon.png" width="60" alt="ModelScope">
</p>

<h3 align="center">modelscope-hub</h3>

<p align="center">
  <em>The official Python SDK & CLI for <a href="https://modelscope.cn">ModelScope Hub</a> — download, upload, and manage AI assets from one unified interface.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/modelscope-hub/"><img alt="PyPI" src="https://img.shields.io/pypi/v/modelscope-hub"></a>
  <a href="https://pypi.org/project/modelscope-hub/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/modelscope-hub"></a>
  <a href="https://github.com/modelscope/modelscope_hub/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/modelscope/modelscope_hub"></a>
</p>

---

## Why modelscope-hub?

`modelscope-hub` is a lightweight, **OpenAPI-first** client that lets you interact with everything on [ModelScope Hub](https://modelscope.cn) — models, datasets, studios, skills, and MCP servers — through a single `HubApi` class or the `ms` CLI.

- **One interface for all repo types** — no separate `model_download` / `dataset_download` functions
- **OpenAPI-first with transparent fallback** — uses the modern OpenAPI surface, falls back to legacy endpoints seamlessly
- **Resumable downloads** — HTTP Range-based resume, parallel threads, SHA256 integrity checks
- **CLI that just works** — `ms download`, `ms upload`, `ms deploy` and more
- **Backward compatible** — drop-in `compat` module for the old `modelscope.hub` SDK

---

## Installation

```bash
pip install modelscope-hub
```

Requires Python 3.10+. No heavy dependencies — only `requests`, `tqdm`, `filelock`, `urllib3`.

---

## Quick Start

### Authenticate

```bash
ms login
# or pass a token directly
ms login --token $MODELSCOPE_API_TOKEN
```

```python
from modelscope_hub import HubApi

api = HubApi(token="your-token")
user = api.whoami()
print(user.username)
```

### Download a Model

```bash
# Full snapshot
ms download Qwen/Qwen2.5-7B-Instruct

# Single file
ms download Qwen/Qwen2.5-7B-Instruct config.json

# With filters
ms download Qwen/Qwen2.5-7B-Instruct --include "*.safetensors" --exclude "*.bin"

# To a specific directory
ms download Qwen/Qwen2.5-7B-Instruct --local-dir ./my-model
```

```python
# Download a single file
path = api.download_file("Qwen/Qwen2.5-7B-Instruct", "model", "config.json")

# Download full repo
snapshot = api.download_repo(
    "Qwen/Qwen2.5-7B-Instruct", "model",
    allow_patterns=["*.safetensors", "*.json"],
    max_workers=8,
)
```

### Upload

```bash
# Upload a file
ms upload my-org/my-model ./weights.safetensors

# Upload a folder
ms upload my-org/my-model ./output --repo-type model
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

---

## CLI Reference

The CLI is available as both `ms` and `modelscope`:

| Command | Description |
|---------|-------------|
| `ms login` | Authenticate and persist token |
| `ms whoami` | Show current user |
| `ms download <repo_id> [files...]` | Download files or snapshots |
| `ms upload <repo_id> <path>` | Upload file or folder |
| `ms repo create/info/list/delete` | Repository management |
| `ms deploy <repo_id>` | Deploy a Studio or MCP server |
| `ms stop <repo_id>` | Stop a running deployment |
| `ms logs <repo_id>` | Fetch runtime/build logs |
| `ms secret add/list/update/delete` | Manage Studio secrets |
| `ms mcp list/info/deploy/undeploy` | MCP server operations |
| `ms cache scan/clear` | Inspect or clean local cache |

Global options: `--token`, `--endpoint`, `-v` (verbose), `-V` (version).

---

## SDK API Overview

All operations go through a single entry point:

```python
from modelscope_hub import HubApi

api = HubApi(token="...", endpoint="https://modelscope.cn")
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
| **Version** | `list_repo_revisions(repo_id, repo_type)` | List branches/tags |
| | `create_repo_tag(repo_id, repo_type, tag)` | Create a tag |
| **Deploy** | `deploy_repo(repo_id, repo_type)` | Deploy Studio/MCP |
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

## Deployment & MCP

Deploy a Studio space or manage MCP servers directly:

```python
# Deploy a Studio
api.deploy_repo("my-org/chat-demo", "studio", payload={"instance_type": "GPU-A10"})

# List MCP servers
page = api.list_mcp_servers(search="weather")
for server in page.items:
    print(server["id"])

# Deploy an MCP server
api.deploy_mcp_server("my-org/weather-mcp")
```

```bash
ms deploy my-org/chat-demo --repo-type studio
ms mcp deploy my-org/weather-mcp
ms logs my-org/chat-demo
```

---

## Backward Compatibility

Migrating from the old `modelscope` SDK? The `compat` module provides drop-in replacements:

```python
# Old code (modelscope.hub)
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi

# New code (swap the import)
from modelscope_hub.compat import snapshot_download, LegacyHubApi as HubApi
```

The CLI also accepts legacy argument styles:

```bash
# These all work
ms download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./out
ms download Qwen/Qwen2.5-7B-Instruct --local-dir ./out
```

<details>
<summary><strong>Compat module exports</strong></summary>

| Symbol | Maps to |
|--------|---------|
| `snapshot_download(model_id, ...)` | `HubApi.download_repo()` |
| `dataset_snapshot_download(dataset_id, ...)` | `HubApi.download_repo()` |
| `model_file_download(model_id, file, ...)` | `HubApi.download_file()` |
| `dataset_file_download(dataset_id, file, ...)` | `HubApi.download_file()` |
| `LegacyHubApi` | Wraps `HubApi` with old method signatures |
| `REPO_TYPE_MODEL`, `REPO_TYPE_DATASET`, ... | String constants |
| `ModelVisibility_PUBLIC/PRIVATE/INTERNAL` | Integer constants |

</details>

---

## Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `MODELSCOPE_API_TOKEN` | Default API token |
| `MODELSCOPE_ENDPOINT` | Override API endpoint |
| `MODELSCOPE_CACHE_DIR` | Override cache directory |
| `MODELSCOPE_HUB_NO_DEPRECATION_WARNINGS` | Suppress legacy arg warnings |

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
