"""Render probe results into a single markdown findings report."""
from __future__ import annotations

import json

from tandem.recon.probes import ProbeResult


def render_report(results: list[ProbeResult], archive: str) -> str:
    lines = [
        "# Reconnaissance findings",
        "",
        f"Archive: `{archive}`",
        "",
        "Each section answers one section-9 assumption. `measured` values",
        "are automatic; `needs_review` items link a contact sheet to tally",
        "by hand; `blocked` items record why a probe could not run.",
        "",
    ]
    for r in sorted(results, key=lambda x: x.assumption):
        lines.append(f"## Assumption {r.assumption}: {r.title}")
        lines.append("")
        lines.append(f"- Status: **{r.status}**")
        if r.degradation:
            lines.append(f"- Degradation: `{r.degradation}`")
        lines.append(f"- Value: `{json.dumps(r.value, ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines)
