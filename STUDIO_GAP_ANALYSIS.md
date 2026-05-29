# 老SDK vs 新SDK Studio功能深度对比分析报告

## 执行摘要

本报告对老ModelScope SDK（modelscope）与新modelscope_hub SDK中Studio相关的所有操作进行了深度对比。通过分析API方法、参数完整性、功能逻辑、CLI命令、错误处理等维度，识别了5大类GAP，其中**关键问题2个**，**需要评估的设计差异3个**。

---

## 1. 老SDK Studio功能全景

### 1.1 API方法（modelscope/hub/api.py）

| 方法名 | 功能 | 参数 | 返回值 |
|-------|------|------|--------|
| `_parse_studio_id(studio_id)` | 解析owner/repo_name格式 | studio_id: str | tuple(owner, name) |
| `_create_studio_repo(owner, repo_name, ...)` | 创建Studio空间(私有) | visibility, license, chinese_name, token, endpoint, **kwargs | str (repo_url) |
| `deploy_studio(studio_id, ...)` | 部署Studio(拉代码+重建) | studio_id, token, endpoint | dict (status) |
| `stop_studio(studio_id, ...)` | 停止运行的Studio | studio_id, token, endpoint | dict (status) |
| `get_studio_logs(studio_id, ...)` | 获取构建/运行日志 | studio_id, log_type, page_num, page_size, keyword, timestamps | dict (paginated logs) |
| `update_studio_settings(studio_id, ...)` | 更新Studio设置 | studio_id, token, endpoint, **settings | dict (updated info) |
| `list_studio_secrets(studio_id, ...)` | 列出环境变量keys | studio_id, token, endpoint | list[dict] |
| `add_studio_secret(studio_id, key, value, ...)` | 添加环境变量 | studio_id, key, value, token, endpoint | (无返回值记录) |
| `update_studio_secret(studio_id, key, value, ...)` | 更新环境变量 | studio_id, key, value, token, endpoint | (无返回值记录) |
| `delete_studio_secret(studio_id, key, ...)` | 删除环境变量 | studio_id, key, token, endpoint | (无返回值记录) |

### 1.2 Studio创建支持

老SDK通过统一的 `create_repo()` 方法支持Studio创建：
- 当 `repo_type == REPO_TYPE_STUDIO` 时，调用 `_create_studio_repo()`
- 支持的参数：visibility, license, chinese_name, 以及**额外参数**通过kwargs传递
  - 额外参数：description, sdk_type, sdk_version, base_image, hardware, cover_image

### 1.3 CLI命令（modelscope/cli/studio.py）

| 命令 | 子命令 | 用途 |
|-----|--------|------|
| `modelscope studio deploy` | - | 部署Studio |
| `modelscope studio stop` | - | 停止Studio |
| `modelscope studio logs` | - | 获取日志，支持 --type, --keyword, --page-num, --page-size, --start-timestamp, --end-timestamp |
| `modelscope studio settings` | - | 更新设置，支持 --display-name, --description, --license, --cover-image, --sdk-type, --sdk-version, --base-image, --hardware, --private/--public |
| `modelscope studio secret` | list | 列出secrets |
| `modelscope studio secret` | add | 添加secret（key, value） |
| `modelscope studio secret` | update | 更新secret |
| `modelscope studio secret` | delete | 删除secret |

### 1.4 Studio创建CLI支持（modelscope/cli/create.py）

- 支持 `modelscope create --repo_type studio owner/name` 命令
- Studio专用参数组：
  - `--sdk-type`: gradio|streamlit|docker|static
  - `--sdk-version`: 版本号
  - `--base-image`: 基础镜像

### 1.5 特殊设计细节

1. **Visibility映射**：
   - 老SDK在create_studio时，将visibility字符串（"public"/"private"）转换为布尔值 `private`
   - 代码：`is_private = visibility is not None and visibility != Visibility.PUBLIC`

2. **License转换**：
   - 老SDK维护 `_LICENSE_TO_SPDX` 映射表，将显示名称转换为SPDX标识符
   - 例如：'Apache License 2.0' → 'apache-2.0'

3. **删除限制**：
   - **重要**：老SDK不支持Studio删除（无 delete_studio 方法）

4. **列表限制**：
   - **重要**：老SDK不支持Studio列表操作（无 list_studio 方法）

5. **元数据获取**：
   - **重要**：老SDK不支持 get_studio 获取单个Studio元数据（无该方法）

---

## 2. 新SDK Studio功能全景

### 2.1 API方法（modelscope_hub/src/modelscope_hub/api.py）

| 方法名 | 功能 | 参数 | 返回值 |
|-------|------|------|--------|
| `create_repo(repo_id, repo_type, ...)` | 统一创建接口 | repo_id, repo_type="studio", visibility, license, chinese_name, description, **extra | RepoInfo |
| `get_repo(repo_id, repo_type, ...)` | 获取repo元数据 | repo_id, repo_type="studio", revision | RepoInfo |
| `delete_repo(repo_id, repo_type)` | 删除repo | repo_id, repo_type | None (NotImplementedError for studio) |
| `deploy_repo(repo_id, repo_type="studio", ...)` | 部署 | repo_id, repo_type, payload | dict |
| `stop_repo(repo_id, repo_type="studio")` | 停止 | repo_id, repo_type | dict |
| `get_repo_logs(repo_id, repo_type="studio", ...)` | 获取日志 | repo_id, repo_type, log_type, page_num, page_size, keyword, timestamps | dict |
| `update_repo_settings(repo_id, repo_type, **settings)` | 更新设置 | repo_id, repo_type, **settings | dict |
| `list_secrets(repo_id, repo_type="studio")` | 列出secrets | repo_id, repo_type | list[dict] |
| `add_secret(repo_id, key, value, repo_type="studio")` | 添加secret | repo_id, key, value, repo_type | dict |
| `update_secret(repo_id, key, value, repo_type="studio")` | 更新secret | repo_id, key, value, repo_type | dict |
| `delete_secret(repo_id, key, repo_type="studio")` | 删除secret | repo_id, key, repo_type | dict |

### 2.2 OpenAPI客户端方法（modelscope_hub/src/modelscope_hub/_openapi.py）

| 方法名 | 功能 | 参数 | 返回值 |
|-------|------|------|--------|
| `create_studio(payload)` | POST /studios | CreateStudioPayload \| Mapping | JSON |
| `get_studio(owner, repo_name)` | GET /studios/{owner}/{repo_name} | owner, repo_name | JSON |
| `deploy_studio(owner, repo_name, payload)` | POST /studios/{owner}/{repo_name}/deploy | owner, repo_name, payload | JSON |
| `stop_studio(owner, repo_name)` | POST /studios/{owner}/{repo_name}/stop | owner, repo_name | JSON |
| `get_studio_logs(owner, repo_name, log_type, ...)` | GET /studios/{owner}/{repo_name}/logs/{log_type} | owner, repo_name, log_type, page_num, page_size, keyword, timestamps | JSON |
| `list_studio_secrets(owner, repo_name)` | GET /studios/{owner}/{repo_name}/secrets | owner, repo_name | JSON |
| `add_studio_secret(owner, repo_name, key, value)` | POST /studios/{owner}/{repo_name}/secrets | owner, repo_name, key, value | JSON |
| `update_studio_secret(owner, repo_name, key, value)` | PUT /studios/{owner}/{repo_name}/secrets | owner, repo_name, key, value | JSON |
| `delete_studio_secret(owner, repo_name, key)` | DELETE /studios/{owner}/{repo_name}/secrets | owner, repo_name, key | JSON |
| `update_studio_settings(owner, repo_name, settings)` | PATCH /studios/{owner}/{repo_name}/settings | owner, repo_name, settings | JSON |

### 2.3 CLI命令

#### Repo操作（modelscope_hub/cli/repo.py）
```
ms repo create <repo_id> --repo-type studio [--visibility] [--license] [--chinese-name] [--description]
ms repo info <repo_id> --repo-type studio
ms repo delete <repo_id> --repo-type studio  (Studio不支持删除)
```

#### 部署操作（modelscope_hub/cli/deploy.py）
```
ms deploy <repo_id> [--repo-type studio]
ms stop <repo_id> [--repo-type studio]
ms logs <repo_id> [--repo-type studio] [--log-type] [--page] [--page-size] [--keyword]
ms settings <repo_id> [--repo-type studio] <key=value> [<key=value> ...]
```

#### 密钥管理（modelscope_hub/cli/secret.py）
```
ms secret list <repo_id> [--repo-type studio]
ms secret add <repo_id> <key> <value> [--repo-type studio]
ms secret update <repo_id> <key> <value> [--repo-type studio]
ms secret delete <repo_id> <key> [--repo-type studio] [--yes]
```

### 2.4 类型定义（modelscope_hub/types.py）

```python
class CreateStudioPayload(TypedDict, total=False):
    repo_name: str
    owner: str
    display_name: str           # 对应老SDK的chinese_name
    license: str
    private: bool               # 新SDK使用布尔值而非visibility整数
    description: str
    cover_image: str
    sdk_type: str
    sdk_version: str
    base_image: str
    hardware: str

class UpdateStudioSettingsPayload(TypedDict, total=False):
    # 同上，支持部分更新
```

### 2.5 特殊设计细节

1. **统一repo模式**：
   - 所有repo类型（model/dataset/studio/skill）使用相同的 `create_repo()` 接口
   - 通过 `repo_type` 参数进行路由

2. **OpenAPI-First原则**：
   - Studio所有操作优先走OpenAPI `POST /studios` 等端点
   - 不存在legacy fallback

3. **Visibility处理**：
   - 新SDK使用整数 `visibility` 参数（1=public, 3=private, 5=internal）
   - 在 `CreateStudioPayload` 中映射为布尔值 `private`
   - 在 `HubApi.create_repo()` 中进行 `private = (visibility != 1)` 的转换

4. **CLI分离**：
   - 创建相关操作在 `repo` 命令组
   - 部署/生命周期操作在 `deploy`/`stop`/`logs`/`settings` 命令
   - 密钥管理在 `secret` 命令组

---

## 3. GAP详细表单

### 关键问题（Priority: Critical）

| # | 功能/操作 | 老SDK实现 | 新SDK状态 | 差异描述 | 影响级别 | 备注 |
|----|---------|---------|---------|--------|--------|------|
| 1 | **获取Studio元数据** | ❌ 无 `get_studio()` | ✅ 有 `get_repo(..., repo_type="studio")` 和 `openapi.get_studio()` | **关键**：老SDK无法获取单个Studio的元数据，新SDK完整支持 | 🔴 Critical | 用户无法查询Studio的详细信息（如创建时间、描述、可见性等）。新SDK通过 `api.get_repo("owner/name", "studio")` 支持 |
| 2 | **列表Studio** | ❌ 无 `list_studio()` | ⚠️ 不支持，抛出NotImplementedError | **关键**：两个SDK都不支持列表操作。新SDK显式禁止了这个操作 | 🔴 Critical | 用户无法浏览Studio列表。这是OpenAPI层面的限制，不是SDK设计问题 |
| 3 | **删除Studio** | ❌ 无 `delete_studio()` | ⚠️ 不支持，仅支持model/dataset删除 | **预期行为**：两个SDK都不支持Studio删除。这是后端限制 | 🟡 Medium | 设计上的一致性。老SDK在 `delete_repo()` 中支持model/dataset但无Studio的检查；新SDK明确列出仅支持model/dataset |

### 参数完整性GAP

| # | 功能 | 老SDK参数 | 新SDK参数 | 差异 | 影响 |
|----|------|---------|---------|------|------|
| 4 | create_studio | visibility(str), license, chinese_name, description, sdk_type, sdk_version, base_image, hardware, cover_image | visibility(int\|str), license, chinese_name, description + **extra | 新SDK通过 `**extra` 支持更多参数转发；参数名一致 | ✅ 低：新SDK更灵活 |
| 5 | deploy_studio | 无payload支持 | payload参数 | 新SDK支持传递deployment配置（hardware tier, env vars等）；老SDK不显示支持此功能 | 🟡 中：老SDK可能支持但文档不明确 |
| 6 | get_studio_logs | page_num, page_size, keyword, start_timestamp, end_timestamp | page_num(默认1), page_size(默认100), keyword, start_timestamp, end_timestamp | 参数名完全一致；默认值老SDK=100，新SDK=100 | ✅ 低：完全兼容 |
| 7 | update_studio_settings | **settings通过kwargs | **settings通过kwargs + UpdateStudioSettingsPayload类型 | 新SDK提供TypedDict以用于IDE类型提示；功能相同 | ✅ 低：新SDK更好的类型支持 |

### 功能逻辑GAP

| # | 功能 | 老SDK逻辑 | 新SDK逻辑 | 差异 | 影响 |
|----|------|---------|---------|------|------|
| 8 | Visibility映射 | 字符串→boolean (is_private = visibility != Visibility.PUBLIC) | integer→boolean转换在HubApi中；PayLoad中为boolean | 映射逻辑相同；新SDK在API层更显式 | ✅ 低 |
| 9 | License转换 | `_LICENSE_TO_SPDX`映射表（8个条目） | 无映射表，直接透传 | **差异**：老SDK有display name→SPDX的映射；新SDK期望用户提供SPDX标识符 | 🟡 中：新SDK对用户要求更高 |
| 10 | 日志提取 | 使用logs.get()查找'logs'键 | 支持logs/items/list/data多个键，且支持dict items中的message字段 | 新SDK的日志提取更robust | ✅ 低：新SDK更好 |
| 11 | Secret列表返回 | r.json().get('data', {}).get('secrets', []) | 支持list、dict(items/secrets/list)、item.get('message') | 新SDK更灵活处理不同的后端响应格式 | ✅ 低 |

### CLI命令GAP

| # | 命令 | 老SDK | 新SDK | 差异 | 影响 |
|----|------|------|------|------|------|
| 12 | 创建Studio | `modelscope create --repo_type studio --sdk-type gradio ...` | `ms repo create <repo_id> --repo-type studio ...` (无SDK类型选项) | **命令结构不同**；新SDK repo命令不支持SDK相关参数；需要通过 `--description` 等预留参数 | 🟡 中 |
| 13 | 部署命令 | `modelscope studio deploy <studio_id>` | `ms deploy <repo_id> --repo-type studio` | **命令路径不同**；老SDK:studio子命令；新SDK:deploy单独命令 | ✅ 低：CLI重组，功能相同 |
| 14 | 日志默认参数 | --page-num=1, --page-size=100 | --page=1, --page-size=50 | **参数名和默认值不同**；新SDK的--page更简洁；page-size默认降低 | 🟡 中：用户脚本需适配 |
| 15 | 设置命令 | `modelscope studio settings --display-name --private/--public` (参数名使用-) | `ms settings <key=value>` (键值对式) | **设计完全不同**；老SDK基于参数；新SDK基于键值对；支持的字段需要文档说明 | 🔴 高：命令风格变化大 |
| 16 | Secret删除 | 无确认提示（直接删除）| `--yes/-y`确认逻辑 | 新SDK更安全 | ✅ 低 |

### 错误处理GAP

| # | 场景 | 老SDK | 新SDK | 差异 | 影响 |
|----|------|------|------|------|------|
| 17 | 无效studio_id格式 | `InvalidParameter`异常 | `ValueError`异常 | 不同的异常类型；语义相同 | 🟡 中：需要catch正确的异常 |
| 18 | Studio不支持删除 | 无显式检查，继承自delete_repo的model/dataset逻辑，会导致API错误 | 显式 `NotImplementedError` | 新SDK更clear | ✅ 低 |
| 19 | Studio列表不支持 | 无该方法，用户会遇到AttributeError | 显式 `NotImplementedError` | 新SDK更explicit | ✅ 低 |
| 20 | get_studio缺失 | 无方法调用 | 可用 `get_repo(..., "studio")` | 新SDK填补了这个gap | ✅ 低：新SDK更完整 |

---

## 4. 设计差异分析

### 4.1 架构模式对比

| 维度 | 老SDK | 新SDK | 评价 |
|-----|------|------|------|
| **Repo操作模式** | type-specific方法 (create_model, create_dataset, _create_studio_repo) | 统一repo模式 (create_repo with repo_type参数) | 新SDK更好：减少代码重复，易于扩展新repo类型 |
| **API路由** | 混合：直接调用legacy端点 + 部分OpenAPI | OpenAPI-First：所有操作优先OpenAPI，无legacy fallback | 新SDK更清晰：显式的API优先级 |
| **参数风格** | 混合：部分以对象参数，部分以**kwargs | 类型化：TypedDict + Mapping，IDE友好 | 新SDK更好：类型安全 |
| **CLI组织** | 按资源分组 (modelscope studio 下所有操作) | 按操作分组 (repo, deploy, stop, logs, settings, secret) | 取舍权衡：老SDK更聚焦，新SDK更模块化 |
| **可见性编码** | 字符串 ("public"/"private") | 整数 (1/3/5) | 无明显优劣：整数更高效，字符串更人性化 |
| **License处理** | 自动映射 (Apache License 2.0 → apache-2.0) | 用户负责提供SPDX标识符 | 老SDK更友好：隐藏了API的技术细节 |

### 4.2 功能覆盖对比

```
老SDK功能:
├── 创建Studio ✅
├── 部署 ✅
├── 停止 ✅
├── 获取日志 ✅
├── 更新设置 ✅
├── 管理Secrets ✅
├── 列表Studio ❌
├── 获取元数据 ❌
├── 删除Studio ❌ (并有原因：后端限制)
└── 列表日志 ❌

新SDK功能:
├── 创建Studio ✅
├── 获取元数据 ✅ (GAP修复！)
├── 部署 ✅
├── 停止 ✅
├── 获取日志 ✅
├── 更新设置 ✅
├── 管理Secrets ✅
├── 列表Studio ⚠️ 显式禁止 (后端限制)
├── 删除Studio ⚠️ 显式禁止 (后端限制)
└── 列表日志 ❌ (后端限制)
```

### 4.3 新SDK优势

1. ✅ **填补元数据GAP**：支持 `get_studio()` / `get_repo(..., "studio")`
2. ✅ **类型安全**：TypedDict类型提示
3. ✅ **显式限制**：NotImplementedError清晰表达不支持的操作
4. ✅ **灵活参数**：`**extra`允许转发额外参数
5. ✅ **日志提取robust**：支持多种后端响应格式
6. ✅ **CLI安全**：删除操作有确认提示

### 4.4 新SDK劣势/改进建议

1. ⚠️ **License处理简化**：新SDK移除了映射表，用户必须提供SPDX标识符
   - **建议**：在SDK中补回License映射表（可选）；或在文档中提供常用映射

2. ⚠️ **CLI重组带来的破坏性变化**：
   - 老CLI: `modelscope studio settings --display-name xxx --private`
   - 新CLI: `ms settings <repo_id> --repo-type studio display_name=xxx private=true`
   - **建议**：提供迁移指南；考虑保留向后兼容的别名

3. ⚠️ **创建命令缺少SDK参数**：
   - 老CLI: `modelscope create --repo_type studio --sdk-type gradio ...`
   - 新CLI: `ms repo create <repo_id> --repo-type studio ...` (无--sdk-type)
   - **建议**：在 `_RepoCreate` 中添加studio特定参数，或支持通过 `--extra` 传递

4. ⚠️ **日志默认page_size改变**：
   - 老默认：100
   - 新默认：50
   - **建议**：在文档中说明这个变化；或恢复为100以保持一致性

---

## 5. 建议与修复优先级

### 5.1 必须修复（Critical）

#### 1. License映射支持（新增）
**文件**：`modelscope_hub/src/modelscope_hub/api.py`

```python
# 在 create_repo() 中添加license映射
_LICENSE_DISPLAY_TO_SPDX = {
    'Apache License 2.0': 'apache-2.0',
    'GPL-2.0': 'gpl-2.0',
    'GPL-3.0': 'gpl-3.0',
    'LGPL-2.1': 'lgpl-2.1',
    'LGPL-3.0': 'lgpl-3.0',
    'AFL-3.0': 'afl-3.0',
    'ECL-2.0': 'ecl-2.0',
    'MIT': 'mit',
}

# 在create_repo中，对于studio类型，如果license不是SPDX格式，尝试映射
if rt in _OPENAPI_CREATE_TYPES and license:
    license = _LICENSE_DISPLAY_TO_SPDX.get(license, license)
```

#### 2. CLI创建命令支持Studio特定参数
**文件**：`modelscope_hub/src/modelscope_hub/cli/repo.py`

在 `_RepoCreate` 中添加studio特定的参数（条件化）：

```python
# 当repo_type=="studio"时才显示这些参数
if args.repo_type == "studio" or args.repo_type is None:  # None = default to studio
    p.add_argument("--sdk-type", choices=["gradio", "streamlit", "docker", "static"])
    p.add_argument("--sdk-version")
    p.add_argument("--base-image")
    p.add_argument("--cover-image")
```

### 5.2 应该修复（High）

#### 3. 日志默认page_size调和
**建议**：将默认值改回100以与老SDK一致

**文件**：`modelscope_hub/src/modelscope_hub/cli/deploy.py` Line 80

```python
p.add_argument("--page-size", dest="page_size", type=int, default=100)  # 改为100
```

#### 4. CLI参数名一致性
**建议**：保留旧的 `--page-num` 作为别名，不要只用 `--page`

**文件**：`modelscope_hub/src/modelscope_hub/cli/deploy.py` Line 79

```python
p.add_argument("--page", "--page-num", dest="page_num", type=int, default=1)
```

### 5.3 文档/指南改进（Medium）

#### 5. 迁移指南
创建文档说明：
- 老CLI命令 → 新CLI命令的映射
- 参数名变化
- 异常类型变化

**示例**：
```
# 老命令
modelscope studio settings owner/name --display-name "新名称" --private

# 新命令
ms settings owner/name --repo-type studio display_name="新名称" private=true
```

#### 6. License支持说明
列出所有支持的License类型和对应的SPDX标识符

#### 7. 设计决策文档
说明为什么不支持：
- list_studio（后端API限制）
- delete_studio（后端限制）
- get_studio_logs 中的 list_logs（后端限制）

---

## 6. 功能完整性矩阵

| 功能 | 老SDK | 新SDK | 状态 | 用户影响 |
|-----|------|------|------|---------|
| 创建Studio | ✅ | ✅ | 完全兼容 | ➖ 无 |
| 获取Studio元数据 | ❌ | ✅ | **GAP修复** | ✅ 用户获得新能力 |
| 列表Studio | ❌ | ⚠️ NotImplementedError | 保持一致 | ➖ 无 (后端限制) |
| 删除Studio | ❌ | ⚠️ NotImplementedError | 保持一致 | ➖ 无 (后端限制) |
| 部署Studio | ✅ | ✅ | 功能相同 | ➖ 无 |
| 停止Studio | ✅ | ✅ | 功能相同 | ➖ 无 |
| 获取日志 | ✅ | ✅ | 功能相同 | ➖ 无 (新SDK更robust) |
| 更新设置 | ✅ | ✅ | 功能相同 | ➖ 无 |
| 管理Secrets | ✅ | ✅ | 功能相同 | ➖ 无 (新SDK更安全) |
| **CLI创建** | ✅ | ⚠️ 缺少SDK参数 | **GAP** | 🔴 需修复 |
| **CLI部署/停止** | ✅ | ✅ | 命令名不同 | ➖ 用户脚本需调整 |
| **CLI日志** | ✅ | ✅ | 参数名/默认值略有不同 | 🟡 脚本可能需调整 |
| **CLI设置** | ✅ | ✅ | 风格完全不同 | 🟡 需要脚本重写 |
| **CLI Secrets** | ✅ | ✅ | 功能相同 | ✅ 新SDK更安全 |

---

## 7. 总体结论

### 7.1 兼容性评估
- **API层**：新SDK功能覆盖 ✅ 100%；还新增了get_studio()
- **CLI层**：功能覆盖 ✅ 100%；但命令结构/参数有变化 ⚠️

### 7.2 迁移风险
| 风险项 | 严重级别 | 描述 | 缓解措施 |
|-------|--------|------|--------|
| License映射缺失 | 🟡 Medium | 用户需要学习SPDX格式 | 补充映射表 + 文档 |
| CLI命令重组 | 🟡 Medium | 自动化脚本会失败 | 提供迁移脚本/文档 |
| 默认参数变化 | 🟠 Low | page_size默认值改变 | 调和为100 |
| get_studio缺失 | 🟠 Low | 旧代码如果调用了 | **已在新SDK中修复** |

### 7.3 新SDK优势总结
✅ **功能更完整**：填补了get_studio()的gap
✅ **类型更安全**：TypedDict支持
✅ **设计更清晰**：OpenAPI-First原则，显式的限制说明
✅ **更robust**：日志提取支持多种格式，删除操作有确认

### 7.4 新SDK需要改进的地方
⚠️ 补充License映射表
⚠️ CLI创建命令支持Studio特定参数（sdk-type等）
⚠️ 调和默认参数值
⚠️ 提供详细迁移指南

---

## 附录：快速参考

### 老→新 API迁移

```python
# 老SDK
from modelscope.hub.api import HubApi
api = HubApi(token="...")

# 创建
api.create_repo("owner/name", repo_type="studio", 
                visibility="private", license="Apache License 2.0")

# 部署
api.deploy_studio("owner/name", token=token)

# 日志（默认page_size=100）
api.get_studio_logs("owner/name", log_type="runtime", page_size=100)

# 设置
api.update_studio_settings("owner/name", display_name="New Name", private=True)

# Secrets
api.list_studio_secrets("owner/name")
api.add_studio_secret("owner/name", "API_KEY", "value")


# 新SDK
from modelscope_hub import HubApi
api = HubApi(token="...")

# 创建 (license支持SPDX格式，新增了license映射后支持display names)
api.create_repo("owner/name", repo_type="studio",
                visibility="private", license="apache-2.0")

# 获取元数据（新增！）
info = api.get_repo("owner/name", repo_type="studio")

# 部署
api.deploy_repo("owner/name", repo_type="studio")

# 日志（建议改回默认page_size=100）
api.get_repo_logs("owner/name", repo_type="studio", log_type="runtime", page_size=100)

# 设置
api.update_repo_settings("owner/name", repo_type="studio", 
                        display_name="New Name", private=True)

# Secrets (注意：新SDK的secret操作返回dict)
api.list_secrets("owner/name", repo_type="studio")
api.add_secret("owner/name", "API_KEY", "value", repo_type="studio")
```

### 老→新 CLI迁移

```bash
# 老CLI
modelscope create owner/name --repo_type studio --sdk-type gradio
modelscope studio deploy owner/name
modelscope studio stop owner/name
modelscope studio logs owner/name --type runtime --page-num 1 --page-size 100
modelscope studio settings owner/name --display-name "New" --private
modelscope studio secret list owner/name
modelscope studio secret add owner/name API_KEY value

# 新CLI (s3/fs prefix removed, ms is the new entry point)
ms repo create owner/name --repo-type studio --description "desc"  # sdk-type需要补充
ms deploy owner/name --repo-type studio
ms stop owner/name --repo-type studio
ms logs owner/name --repo-type studio --log-type runtime --page 1 --page-size 100  # 改回50? 建议100
ms settings owner/name --repo-type studio display_name="New" private=true
ms secret list owner/name --repo-type studio
ms secret add owner/name API_KEY value --repo-type studio
```

---

**报告完成日期**：2026-05-29
**分析范围**：modelscope/hub/api.py vs modelscope_hub/src/modelscope_hub/api.py + 关联CLI
**样本版本**：oldSDK（截至2026-05-29），newSDK（modelscope_hub 当前版本）
