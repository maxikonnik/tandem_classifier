"""Public surface of the physical phase detector: Signals -> PhaseResult.

Emits phases up to freefall and the exit event, all source="telemetry".
Records NO_TELEMETRY when there is no usable signal, and
PHASE_ORDER_VIOLATION when the detected order is physically impossible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tandem.phases.detect import (
    Event, Segment, detect_exit, detect_freefall, detect_ground_climb,
)
from tandem.phases.fsm import validate_order


@dataclass
class PhaseResult:
    phases: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)


def detect_phases(sig) -> PhaseResult:
    usable = (sig.has_accel or sig.has_gps) and (sig.accel_mag or sig.speed_3d)
    if not usable:
        return PhaseResult(degradations=["NO_TELEMETRY"])

    exit_event = detect_exit(sig)
    ground_climb = detect_ground_climb(sig, exit_event)
    freefall = detect_freefall(sig, exit_event)

    phases: list[Segment] = list(ground_climb)
    if freefall is not None:
        phases.append(freefall)
    events: list[Event] = [exit_event] if exit_event is not None else []

    degradations: list[str] = []
    violation = validate_order(phases)
    if violation is not None:
        degradations.append(violation)
        # Mark every phase unreliable when the order did not converge.
        phases = [Segment(p.type, p.start_s, p.end_s, p.source, 0.0) for p in phases]

    return PhaseResult(phases=phases, events=events, degradations=degradations)
