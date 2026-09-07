"""Real filesystem tools, confined to one sandbox root.

The 11 filesystem names in the tool bank come from the official
``@modelcontextprotocol/server-filesystem``. They need no API key and no network
— only a root to stay inside. Every path is resolved and checked against that
root before it is touched, so a tool call can never reach the training data or
the checkpoints.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
from typing import Any
from contextspan.duetaspan.common import paths

ROOT = os.path.realpath(
    os.environ.get("MOSHICP_FS_ROOT", f"{paths.WORK}/mcp_fs")
)


class FsError(Exception):
    pass


def _resolve(path: str) -> str:
    """Resolve ``path`` under ROOT, refusing anything that escapes it."""
    full = os.path.realpath(os.path.join(ROOT, str(path).lstrip("/")))
    if full != ROOT and not full.startswith(ROOT + os.sep):
        raise FsError(f"path escapes sandbox root: {path}")
    return full


def _ensure_root() -> None:
    os.makedirs(ROOT, exist_ok=True)


def read_file(path: str, head: int | None = None, tail: int | None = None) -> str:
    with open(_resolve(path), "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    if head:
        lines = lines[: int(head)]
    elif tail:
        lines = lines[-int(tail):]
    return "".join(lines)


def read_multiple_files(paths: list[str]) -> str:
    out = []
    for path in paths:
        try:
            out.append(f"{path}:\n{read_file(path)}")
        except OSError as exc:
            out.append(f"{path}: ERROR {exc}")
    return "\n---\n".join(out)


def write_file(path: str, content: str) -> str:
    _ensure_root()
    full = _resolve(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as handle:
        handle.write(content)
    return f"wrote {len(content)} bytes to {path}"


def edit_file(path: str, edits: list[dict], dryRun: bool = False) -> str:
    """Apply {oldText, newText} edits. Every oldText must occur exactly once."""
    full = _resolve(path)
    with open(full, "r", encoding="utf-8") as handle:
        text = handle.read()
    for edit in edits:
        old, new = edit["oldText"], edit["newText"]
        hits = text.count(old)
        if hits != 1:
            raise FsError(f"oldText matched {hits} times, need exactly 1")
        text = text.replace(old, new)
    if not dryRun:
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(text)
    return f"applied {len(edits)} edit(s) to {path}"


def create_directory(path: str) -> str:
    _ensure_root()
    os.makedirs(_resolve(path), exist_ok=True)
    return f"created {path}"


def list_directory(path: str = ".") -> str:
    full = _resolve(path)
    rows = []
    for name in sorted(os.listdir(full)):
        kind = "[DIR]" if os.path.isdir(os.path.join(full, name)) else "[FILE]"
        rows.append(f"{kind} {name}")
    return "\n".join(rows) or "(empty)"


def list_directory_with_sizes(path: str = ".") -> str:
    full = _resolve(path)
    rows = []
    for name in sorted(os.listdir(full)):
        target = os.path.join(full, name)
        if os.path.isdir(target):
            rows.append(f"[DIR]  {name}")
        else:
            rows.append(f"[FILE] {name} {os.path.getsize(target)} bytes")
    return "\n".join(rows) or "(empty)"


def directory_tree(path: str = ".") -> str:
    full = _resolve(path)
    lines = []
    for dirpath, dirnames, filenames in os.walk(full):
        depth = dirpath[len(full):].count(os.sep)
        indent = "  " * depth
        lines.append(f"{indent}{os.path.basename(dirpath) or '.'}/")
        for name in sorted(filenames):
            lines.append(f"{indent}  {name}")
        dirnames.sort()
    return "\n".join(lines)


def search_files(path: str = ".", pattern: str = "*",
                 excludePatterns: list | None = None) -> str:
    full = _resolve(path)
    excludes = excludePatterns or []
    hits = []
    for dirpath, _, filenames in os.walk(full):
        for name in filenames:
            if not fnmatch.fnmatch(name, pattern):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
            if any(fnmatch.fnmatch(rel, ex) for ex in excludes):
                continue
            hits.append(rel)
    return "\n".join(sorted(hits)) or "(no matches)"


def get_file_info(path: str) -> str:
    full = _resolve(path)
    stat = os.stat(full)
    kind = "directory" if os.path.isdir(full) else "file"
    return (f"type: {kind}, size: {stat.st_size}, mode: {oct(stat.st_mode)[-3:]}, "
            f"modified: {int(stat.st_mtime)}")


def list_allowed_directories() -> str:
    return ROOT


def reset_sandbox() -> None:
    """Wipe the sandbox — used by the test harness, never by a tool."""
    if os.path.isdir(ROOT):
        shutil.rmtree(ROOT)
    _ensure_root()


TOOLS: dict[str, Any] = {
    "read_file": read_file,
    "read_multiple_files": read_multiple_files,
    "write_file": write_file,
    "edit_file": edit_file,
    "create_directory": create_directory,
    "list_directory": list_directory,
    "list_directory_with_sizes": list_directory_with_sizes,
    "directory_tree": directory_tree,
    "search_files": search_files,
    "get_file_info": get_file_info,
    "list_allowed_directories": list_allowed_directories,
}
