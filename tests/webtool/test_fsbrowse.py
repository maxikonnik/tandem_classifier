import os

from tandem.webtool.fsbrowse import DirEntry, list_dir, resolve_within


def test_list_dir_returns_dirs_then_mp4_sorted(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "Zclip.MP4").write_bytes(b"x")
    (tmp_path / "aclip.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")  # ignored

    entries = list_dir(str(tmp_path))
    names = [(e.name, e.is_dir) for e in entries]
    assert names == [
        ("a_dir", True), ("b_dir", True),
        ("aclip.mp4", False), ("Zclip.MP4", False),
    ]
    assert all(isinstance(e, DirEntry) for e in entries)


def test_resolve_within_accepts_child_and_rejects_escape(tmp_path):
    root = str(tmp_path)
    child = os.path.join(root, "sub", "clip.mp4")
    assert resolve_within(child, [root]) == os.path.abspath(child)

    escape = os.path.join(root, "..", "secret.mp4")
    assert resolve_within(escape, [root]) is None
