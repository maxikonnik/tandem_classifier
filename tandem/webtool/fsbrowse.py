"""Server-side filesystem listing for the browse modal, with a
path-safety guard so requests cannot escape the active roots."""
from __future__ import annotations

import os
import string
from dataclasses import dataclass

VIDEO_EXTS = (".mp4",)


@dataclass
class DirEntry:
    name: str
    path: str
    is_dir: bool


def list_dir(path: str) -> list[DirEntry]:
    dirs: list[DirEntry] = []
    files: list[DirEntry] = []
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            full = os.path.join(path, entry.name)
            if entry.is_dir():
                dirs.append(DirEntry(entry.name, full, True))
            elif os.path.splitext(entry.name)[1].lower() in VIDEO_EXTS:
                files.append(DirEntry(entry.name, full, False))
    dirs.sort(key=lambda e: e.name.lower())
    files.sort(key=lambda e: e.name.lower())
    return dirs + files


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def resolve_within(path: str, roots: list[str]) -> str | None:
    target = _norm(path)
    for root in roots:
        base = _norm(root)
        if target == base or target.startswith(base.rstrip(os.sep) + os.sep):
            return os.path.abspath(path)
    return None


def drive_roots() -> list[str]:
    if os.name != "nt":
        return ["/"]
    roots = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            roots.append(drive)
    return roots
