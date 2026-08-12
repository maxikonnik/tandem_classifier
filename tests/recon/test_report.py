from tandem.recon.probes import ProbeResult
from tandem.recon.report import render_report


def test_report_lists_every_assumption_with_status():
    results = [
        ProbeResult(1, "GPS UTC present", "measured", {"has_utc": True}),
        ProbeResult(3, "Ground background fraction", "needs_review",
                    {"contact_sheet": "/out/jump-1/freefall.html"}),
        ProbeResult(7, "Fraction with both cameras", "measured", {"fraction": 0.25},
                    degradation="NO_INSTRUCTOR_CAM"),
    ]
    md = render_report(results, archive="/data/archive")
    assert "# Reconnaissance findings" in md
    assert "Assumption 1" in md
    assert "measured" in md
    assert "needs_review" in md
    assert "/out/jump-1/freefall.html" in md
    assert "NO_INSTRUCTOR_CAM" in md
    assert "/data/archive" in md
