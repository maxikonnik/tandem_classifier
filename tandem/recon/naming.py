"""Parse GoPro file names and group chapters into logical recordings.

GoPro splits a long recording into 4 GB chapters. The naming is
``<EE><CC><NNNN>.<ext>`` where ``EE`` is a two-letter encoding tag
(``GX`` for HEVC, ``GH`` for AVC), ``CC`` is the chapter number
starting at 01, and ``NNNN`` is the file/recording number shared by
every chapter of the same recording.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_GOPRO_RE = re.compile(r"^(?P<enc>G[HX])(?P<chapter>\d{2})(?P<num>\d{4})\.(?P<ext>MP4)$", re.IGNORECASE)


@dataclass(frozen=True)
class GoProName:
    encoding: str
    chapter: int
    file_number: int
    ext: str


@dataclass
class Recording:
    file_number: int
    chapters: list[str] = field(default_factory=list)


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def parse_gopro_name(name: str) -> GoProName | None:
    m = _GOPRO_RE.match(_basename(name))
    if not m:
        return None
    return GoProName(
        encoding=m.group("enc").upper(),
        chapter=int(m.group("chapter")),
        file_number=int(m.group("num")),
        ext=m.group("ext").upper(),
    )


def group_recordings(paths: list[str]) -> tuple[list[Recording], list[str]]:
    buckets: dict[int, list[tuple[int, str]]] = {}
    unmatched: list[str] = []
    for path in paths:
        parsed = parse_gopro_name(path)
        if parsed is None:
            unmatched.append(path)
            continue
        buckets.setdefault(parsed.file_number, []).append((parsed.chapter, path))
    recordings = []
    for number in sorted(buckets):
        chapters = [p for _, p in sorted(buckets[number])]
        recordings.append(Recording(file_number=number, chapters=chapters))
    return recordings, unmatched
