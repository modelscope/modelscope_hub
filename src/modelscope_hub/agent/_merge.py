# Copyright (c) Alibaba, Inc. and its affiliates.
"""Section-level Markdown merge engine for cross-framework workspace migration."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger("agent")


@dataclass
class Section:
    """A markdown section: optional title (## heading) + body lines."""
    title: str  # e.g. "## Active Tasks", empty string for preamble
    body: str   # everything after the title line until the next ## heading


@dataclass
class MergeAction:
    """Describes what happened to a single file or section during merge."""
    path: str
    action: str   # 'import' | 'default' | 'merged' | 'skip'
    detail: str   # human-readable description


@dataclass
class MergeResult:
    """Result of merging a single file."""
    content: str
    actions: list[MergeAction] = field(default_factory=list)


@dataclass
class FullMergeResult:
    """Result of merging an entire resource set."""
    merged_files: dict[str, str] = field(default_factory=dict)
    actions: list[MergeAction] = field(default_factory=list)


class SectionMerger:
    """Markdown section-level merge engine.

    Splits markdown by ``## `` headings, diffs user content against source
    defaults, and produces a merged result using target defaults as the base.
    """

    # ---- Parsing ----

    @staticmethod
    def parse_sections(content: str) -> list[Section]:
        """Split markdown into sections by ``## `` headings."""
        lines = content.split("\n")
        sections: list[Section] = []
        current_title = ""
        current_lines: list[str] = []

        for line in lines:
            if re.match(r"^## ", line):
                sections.append(Section(
                    title=current_title,
                    body="\n".join(current_lines),
                ))
                current_title = line
                current_lines = []
            else:
                current_lines.append(line)

        sections.append(Section(
            title=current_title,
            body="\n".join(current_lines),
        ))
        return sections

    @staticmethod
    def sections_to_content(sections: list[Section]) -> str:
        """Reconstruct markdown from sections."""
        parts = []
        for sec in sections:
            if sec.title:
                parts.append(sec.title + "\n" + sec.body)
            else:
                parts.append(sec.body)
        return "\n".join(parts)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize whitespace for comparison."""
        return text.strip()

    # ---- Diffing ----

    def diff_sections(
        self,
        user_content: str,
        source_default: str,
    ) -> tuple[list[Section], list[Section], list[Section]]:
        """Compare user content against source default.

        Returns:
            (unchanged, modified, added) -- three lists of Section objects.
        """
        user_secs = self.parse_sections(user_content)
        default_secs = self.parse_sections(source_default)

        default_map: dict[str, str] = {}
        for sec in default_secs:
            if sec.title:
                default_map[sec.title] = self._normalize(sec.body)

        unchanged, modified, added = [], [], []

        for sec in user_secs:
            if not sec.title:
                default_preamble = ""
                for ds in default_secs:
                    if not ds.title:
                        default_preamble = self._normalize(ds.body)
                        break
                if self._normalize(sec.body) != default_preamble:
                    modified.append(sec)
                else:
                    unchanged.append(sec)
                continue

            if sec.title in default_map:
                if self._normalize(sec.body) == default_map[sec.title]:
                    unchanged.append(sec)
                else:
                    modified.append(sec)
            else:
                added.append(sec)

        return unchanged, modified, added

    # ---- Merging ----

    def merge(
        self,
        user_content: str,
        source_default: str,
        target_default: str,
    ) -> MergeResult:
        """Merge user content into target default using section-level diff.

        Strategy:
        1. Start with target_default sections as the base.
        2. For sections the user modified: replace the target section body.
        3. For sections the user added: append at the end.
        4. Unchanged sections keep the target default version.
        """
        unchanged, modified, added = self.diff_sections(user_content, source_default)
        target_secs = self.parse_sections(target_default)

        actions = []
        modified_titles = {sec.title for sec in modified}
        added_titles = {sec.title for sec in added}
        modified_map = {sec.title: sec for sec in modified}

        result_secs = []
        for tsec in target_secs:
            if tsec.title in modified_titles:
                result_secs.append(modified_map[tsec.title])
                actions.append(MergeAction(
                    path="", action="user_modified",
                    detail=f"Section '{tsec.title}' -- user modification preserved",
                ))
            elif not tsec.title and "" in modified_titles:
                preamble_sec = modified_map.get("")
                if preamble_sec:
                    result_secs.append(preamble_sec)
                    actions.append(MergeAction(
                        path="", action="user_modified",
                        detail="Preamble -- user modification preserved",
                    ))
                else:
                    result_secs.append(tsec)
            else:
                result_secs.append(tsec)
                if tsec.title:
                    actions.append(MergeAction(
                        path="", action="keep_default",
                        detail=f"Section '{tsec.title}' -- target default",
                    ))

        # Append user-added sections
        for sec in added:
            result_secs.append(sec)
            actions.append(MergeAction(
                path="", action="user_added",
                detail=f"Section '{sec.title}' -- user custom section added",
            ))

        content = self.sections_to_content(result_secs)

        user_changes = len(modified) + len(added)
        summary = f"target default + {user_changes} user customization(s)"

        return MergeResult(
            content=content,
            actions=[MergeAction(path="", action="merged", detail=summary)] + actions,
        )


class HeartbeatMerger(SectionMerger):
    """Specialized merger for HEARTBEAT.md with line-level task merging
    inside the ``## Active Tasks`` section."""

    ACTIVE_TASKS_TITLE = "## Active Tasks"

    def _extract_task_lines(self, body: str) -> list[str]:
        """Extract non-empty, non-comment lines from a section body."""
        lines = []
        for line in body.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("<!--") and not stripped.endswith("-->"):
                lines.append(line)
        return lines

    def merge(
        self,
        user_content: str,
        source_default: str,
        target_default: str,
    ) -> MergeResult:
        """Merge HEARTBEAT.md with line-level Active Tasks merging."""
        user_secs = self.parse_sections(user_content)
        default_secs = self.parse_sections(source_default)

        user_active_body = ""
        default_active_body = ""
        for sec in user_secs:
            if sec.title == self.ACTIVE_TASKS_TITLE:
                user_active_body = sec.body
                break
        for sec in default_secs:
            if sec.title == self.ACTIVE_TASKS_TITLE:
                default_active_body = sec.body
                break

        default_task_lines = set(l.strip() for l in self._extract_task_lines(default_active_body))
        user_task_lines = self._extract_task_lines(user_active_body)
        new_tasks = [l for l in user_task_lines if l.strip() not in default_task_lines]

        result = super().merge(user_content, source_default, target_default)

        if not new_tasks:
            return result

        result_secs = self.parse_sections(result.content)
        for i, sec in enumerate(result_secs):
            if sec.title == self.ACTIVE_TASKS_TITLE:
                body_lines = sec.body.split("\n")
                insert_idx = len(body_lines)
                for j, line in enumerate(body_lines):
                    if "<!--" in line and j > 0:
                        insert_idx = j
                        break
                for task in new_tasks:
                    body_lines.insert(insert_idx, task)
                    insert_idx += 1
                result_secs[i] = Section(
                    title=sec.title,
                    body="\n".join(body_lines),
                )
                break

        result.content = self.sections_to_content(result_secs)
        result.actions.append(MergeAction(
            path="", action="task_merged",
            detail=f"{len(new_tasks)} user task(s) merged into Active Tasks",
        ))
        return result


# ---- Full resource merge orchestrator ----

PRODUCT_FILE_CLASSES = {
    "nanobot": {
        "portable": frozenset([
            "SOUL.md", "USER.md", "memory/MEMORY.md", "memory/HISTORY.md",
        ]),
        "config": frozenset([
            "AGENTS.md", "HEARTBEAT.md", "TOOLS.md",
        ]),
        "heartbeat": "HEARTBEAT.md",
    },
    "openclaw": {
        "portable": frozenset([
            "SOUL.md", "USER.md", "IDENTITY.md", "MEMORY.md",
        ]),
        "config": frozenset([
            "AGENTS.md", "HEARTBEAT.md", "TOOLS.md", "BOOTSTRAP.md",
        ]),
        "heartbeat": "HEARTBEAT.md",
    },
    "hermes": {
        "portable": frozenset([
            "SOUL.md", "memories/USER.md",
        ]),
        "config": frozenset([]),
        "heartbeat": "",
    },
    "qwenpaw": {
        "portable": frozenset([
            "SOUL.md", "PROFILE.md", "MEMORY.md",
        ]),
        "config": frozenset([
            "AGENTS.md", "HEARTBEAT.md", "BOOTSTRAP.md",
        ]),
        "heartbeat": "HEARTBEAT.md",
    },
    "openhuman": {
        "portable": frozenset([
            "SOUL.md", "IDENTITY.md", "USER.md", "PROFILE.md", "MEMORY.md",
        ]),
        "config": frozenset([
            "HEARTBEAT.md",
        ]),
        "heartbeat": "HEARTBEAT.md",
    },
}

_DEFAULT_FILE_CLASS = {
    "portable": frozenset(["SOUL.md", "USER.md"]),
    "config": frozenset(["AGENTS.md", "HEARTBEAT.md", "TOOLS.md"]),
    "heartbeat": "HEARTBEAT.md",
}

_section_merger = SectionMerger()
_heartbeat_merger = HeartbeatMerger()


# ---- Cross-product path mapping ----
SEMANTIC_GROUPS = [
    {"nanobot": "SOUL.md", "openclaw": "SOUL.md", "hermes": "SOUL.md",
     "qwenpaw": "SOUL.md", "openhuman": "SOUL.md"},
    {"nanobot": "USER.md", "openclaw": "USER.md", "hermes": "memories/USER.md",
     "openhuman": "USER.md"},
    {"nanobot": "memory/MEMORY.md", "openclaw": "MEMORY.md",
     "qwenpaw": "MEMORY.md", "openhuman": "MEMORY.md"},
    {"openclaw": "IDENTITY.md", "openhuman": "IDENTITY.md"},
    {"qwenpaw": "PROFILE.md", "openhuman": "PROFILE.md"},
    {"nanobot": "AGENTS.md", "openclaw": "AGENTS.md", "qwenpaw": "AGENTS.md"},
    {"nanobot": "HEARTBEAT.md", "openclaw": "HEARTBEAT.md",
     "qwenpaw": "HEARTBEAT.md", "openhuman": "HEARTBEAT.md"},
    {"nanobot": "TOOLS.md", "openclaw": "TOOLS.md"},
    {"openclaw": "BOOTSTRAP.md", "qwenpaw": "BOOTSTRAP.md"},
    {"nanobot": "memory/HISTORY.md"},
]

_ALL_PRODUCTS = ["nanobot", "openclaw", "hermes", "qwenpaw", "openhuman"]


def _build_path_map():
    path_map = {}
    for group in SEMANTIC_GROUPS:
        for src_product, src_path in group.items():
            targets = {
                tgt_product: group.get(tgt_product)
                for tgt_product in _ALL_PRODUCTS
                if tgt_product != src_product
            }
            path_map[(src_product, src_path)] = targets
    return path_map


PATH_MAP = _build_path_map()

PRODUCT_KNOWN_FILES = {
    "nanobot": frozenset([
        "SOUL.md", "USER.md", "AGENTS.md", "TOOLS.md", "HEARTBEAT.md",
        "memory/MEMORY.md", "memory/HISTORY.md",
    ]),
    "openclaw": frozenset([
        "SOUL.md", "USER.md", "AGENTS.md", "TOOLS.md", "HEARTBEAT.md",
        "IDENTITY.md", "BOOTSTRAP.md", "MEMORY.md",
    ]),
    "hermes": frozenset([
        "SOUL.md", "memories/USER.md",
    ]),
    "qwenpaw": frozenset([
        "SOUL.md", "PROFILE.md", "AGENTS.md", "MEMORY.md", "HEARTBEAT.md",
        "BOOTSTRAP.md",
    ]),
    "openhuman": frozenset([
        "SOUL.md", "IDENTITY.md", "USER.md", "PROFILE.md", "MEMORY.md",
        "HEARTBEAT.md",
    ]),
}


def _resolve_target_path(source_product: str, source_path: str, target_product: str) -> str | None:
    """Resolve the target path for a source file in a cross-product migration."""
    if source_product == target_product:
        return source_path
    key = (source_product, source_path)
    if key in PATH_MAP:
        return PATH_MAP[key].get(target_product)
    return source_path


def _extract_user_diff_text(user_content: str, source_default: str) -> str:
    """Extract user customizations as a text block."""
    if not source_default.strip():
        return user_content.strip()

    user_lines = user_content.strip().split("\n")
    default_lines = source_default.strip().split("\n")
    default_set = set(l.strip() for l in default_lines)

    diff_lines = []
    for line in user_lines:
        if line.strip() and line.strip() not in default_set:
            diff_lines.append(line)

    return "\n".join(diff_lines).strip()


def _catch_all_file(product: str) -> str:
    """The file that receives mutually-exclusive cross-product content."""
    known = PRODUCT_KNOWN_FILES.get(product)
    if known is None:
        return "AGENTS.md"
    if "AGENTS.md" in known:
        return "AGENTS.md"
    return "SOUL.md"


def merge_resources(
    incoming: dict[str, str],
    source_product: str,
    target_product: str,
    source_defaults: dict[str, str],
    target_defaults: dict[str, str],
    existing_skills: list[str] | None = None,
) -> FullMergeResult:
    """Merge incoming resources into a target product workspace.

    Args:
        incoming: files from the share snapshot {rel_path: content}
        source_product: product the snapshot came from
        target_product: product the user wants to apply to
        source_defaults: default templates for source product
        target_defaults: default templates for target product
        existing_skills: list of skill dir names already on target
    """
    is_cross_product = source_product != target_product
    existing_skill_set = set(existing_skills or [])
    result = FullMergeResult()

    src_cls = PRODUCT_FILE_CLASSES.get(source_product, _DEFAULT_FILE_CLASS)
    tgt_cls = PRODUCT_FILE_CLASSES.get(target_product, _DEFAULT_FILE_CLASS)
    portable_files = src_cls["portable"] | tgt_cls["portable"]
    config_files = src_cls["config"] | tgt_cls["config"]
    heartbeat_file = tgt_cls.get("heartbeat", "")

    handled_target_paths = set()
    overflow_blocks: list[tuple[str, str]] = []

    for path, content in incoming.items():
        # Skills: direct import, skip if exists
        if path.startswith("skills/"):
            parts = path.split("/")
            skill_name = parts[1] if len(parts) > 1 else ""
            if skill_name in existing_skill_set:
                result.actions.append(MergeAction(
                    path=path, action="skip",
                    detail=f"Skill '{skill_name}' already exists on target, skipped",
                ))
                continue
            result.merged_files[path] = content
            result.actions.append(MergeAction(
                path=path, action="import",
                detail="Skill imported",
            ))
            continue

        target_path = _resolve_target_path(source_product, path, target_product)

        if target_path is None:
            if path.startswith("memory/") or path.startswith("memories/") or path.startswith("wiki/"):
                result.merged_files[path] = content
                result.actions.append(MergeAction(
                    path=path, action="import",
                    detail=f"No mapping for {target_product}, imported as-is",
                ))
                continue
            user_diff = _extract_user_diff_text(content, source_defaults.get(path, ""))
            if not user_diff:
                result.actions.append(MergeAction(
                    path=path, action="skip",
                    detail=f"{path} has no equivalent in {target_product} and no user changes, skipped",
                ))
                continue
            catch_all = _catch_all_file(target_product)
            block = f"## Imported from {source_product} {path}\n\n{user_diff}\n"
            overflow_blocks.append((catch_all, block))
            result.actions.append(MergeAction(
                path=catch_all, action="merged",
                detail=f"Mutually-exclusive content from {path} merged into {catch_all}",
            ))
            continue

        handled_target_paths.add(target_path)

        if not is_cross_product:
            if path in portable_files:
                result.merged_files[path] = content
                result.actions.append(MergeAction(
                    path=path, action="import",
                    detail="User data imported directly",
                ))
            elif path in config_files:
                result.merged_files[path] = content
                result.actions.append(MergeAction(
                    path=path, action="import",
                    detail="Same product, imported directly",
                ))
            elif path.startswith("memory/") or path.startswith("memories/"):
                result.merged_files[path] = content
                result.actions.append(MergeAction(
                    path=path, action="import",
                    detail="Memory file imported directly",
                ))
            else:
                result.merged_files[path] = content
                result.actions.append(MergeAction(
                    path=path, action="import",
                    detail="Imported directly",
                ))
            continue

        # ---- Cross-product logic ----
        if path in src_cls["portable"]:
            src_default = source_defaults.get(path, "")
            tgt_default = target_defaults.get(target_path, "")

            if target_path == path and tgt_default:
                mr = _section_merger.merge(content, src_default, tgt_default)
                result.merged_files[target_path] = mr.content
                summary = mr.actions[0].detail if mr.actions else "merged"
                result.actions.append(MergeAction(
                    path=target_path, action="merged",
                    detail=summary,
                ))
            elif tgt_default:
                user_diff = _extract_user_diff_text(content, src_default)
                if user_diff:
                    merged_content = tgt_default.rstrip() + \
                        f"\n\n## Imported from {source_product} {path}\n\n{user_diff}\n"
                    result.merged_files[target_path] = merged_content
                    result.actions.append(MergeAction(
                        path=target_path, action="merged",
                        detail=f"Target template + user data from {path}",
                    ))
                else:
                    result.merged_files[target_path] = tgt_default
                    result.actions.append(MergeAction(
                        path=target_path, action="default",
                        detail=f"No user changes detected, using {target_product} default",
                    ))
            else:
                result.merged_files[target_path] = content
                result.actions.append(MergeAction(
                    path=target_path, action="import",
                    detail=f"Imported from {path}" + (f" -> {target_path}" if target_path != path else ""),
                ))
            continue

        if path in config_files:
            src_default = source_defaults.get(path, "")
            tgt_default = target_defaults.get(target_path, "")
            if not tgt_default:
                result.merged_files[target_path] = content
                result.actions.append(MergeAction(
                    path=target_path, action="import",
                    detail="No target default available, imported as-is",
                ))
            else:
                merger = _heartbeat_merger if target_path == heartbeat_file else _section_merger
                mr = merger.merge(content, src_default, tgt_default)
                result.merged_files[target_path] = mr.content
                summary = mr.actions[0].detail if mr.actions else "merged"
                result.actions.append(MergeAction(
                    path=target_path, action="merged",
                    detail=summary,
                ))
            continue

        if path.startswith("memory/") or path.startswith("memories/"):
            result.merged_files[target_path] = content
            result.actions.append(MergeAction(
                path=target_path, action="import",
                detail="Memory file imported directly",
            ))
            continue

        result.merged_files[target_path] = content
        result.actions.append(MergeAction(
            path=target_path, action="import",
            detail="Imported directly",
        ))

    # Fill in missing files from target defaults
    for path, content in target_defaults.items():
        if path not in result.merged_files and path not in handled_target_paths:
            result.merged_files[path] = content
            result.actions.append(MergeAction(
                path=path, action="default",
                detail=f"Added from {target_product} default template",
            ))

    # Append overflow blocks
    for catch_all, block in overflow_blocks:
        base = result.merged_files.get(catch_all)
        if base is None:
            base = target_defaults.get(catch_all, "")
        result.merged_files[catch_all] = (
            base.rstrip() + "\n\n" + block if base.strip() else block
        )

    return result
