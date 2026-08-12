from pathlib import Path

from tandem.recon.windows import Windows
from tandem.recon.sampler import sample_plan, build_contact_sheet


def test_sample_plan_covers_freefall_and_canopy():
    w = Windows(freefall_start_s=10.0, freefall_end_s=20.0, canopy_open_s=22.0)
    shots = sample_plan(w, every_s=5.0)
    labels = {s.label for s in shots}
    assert "freefall" in labels
    assert "canopy" in labels
    freefall_times = [s.t_s for s in shots if s.label == "freefall"]
    assert min(freefall_times) >= 10.0
    assert max(freefall_times) <= 20.0


def test_contact_sheet_is_self_contained(tmp_path):
    img = tmp_path / "freefall_10.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker bytes
    out = tmp_path / "sheet.html"
    build_contact_sheet([str(img)], str(out), title="jump-001")
    html = out.read_text(encoding="utf-8")
    assert "jump-001" in html
    assert "data:image/jpeg;base64," in html  # embedded, no external refs
    assert "freefall_10.jpg" in html
