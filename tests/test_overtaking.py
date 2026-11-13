from __future__ import annotations

import numpy as np
import pytest

from src.lane_detection.interfaces import LaneDetectionResult
from src.overtaking.detector import OvertakingDetector
from src.overtaking.schemas import (
    CorridorPosition,
    IdentityContinuityStatus,
    OvertakingEvent,
    OvertakingEventStatus,
    OvertakingState,
)
from src.schemas import BoundingBox, Frame, PerceivedObject


def make_frame(frame_id: int, timestamp: float | None, width: int = 1280, height: int = 720) -> Frame:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=frame_id, timestamp=timestamp, source="test")


def make_box(cx: float, cy: float, w: float = 60.0, h: float = 60.0) -> BoundingBox:
    return BoundingBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def make_perceived(
    track_id: int,
    bbox: BoundingBox,
    class_name: str = "car",
    depth: float = 20.0,
    depth_confidence: float = 1.0,
) -> PerceivedObject:
    return PerceivedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=bbox,
        depth=depth,
        distance=depth,
        depth_confidence=depth_confidence,
    )


# ===================================================================== #
# Group 1: Timestamp & Timing Precision Tests (Tests 1 to 8)
# ===================================================================== #

def test_1_start_timestamp_backtracking() -> None:
    """1. Start timestamp corresponds to earliest physical onset, not confirmation timestamp."""
    detector = OvertakingDetector()

    # Baseline observations ahead (t=0.1..0.3)
    for i in range(1, 4):
        detector.process(
            [make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0), depth=50.0)],
            None, None, make_frame(i, i * 0.1)
        )

    # Physical approach begins at t=0.4 (scale increases, cy descends)
    # Consecutive approach frames t=0.4..0.8
    for i in range(4, 9):
        t = i * 0.1
        size = 40.0 + (i - 3) * 6.0
        cy = 280.0 + (i - 3) * 25.0
        depth = 50.0 - (i - 3) * 6.0
        detector.process(
            [make_perceived(1, make_box(cx=600.0, cy=cy, w=size, h=size), depth=depth)],
            None, None, make_frame(i, t)
        )

    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.state in (OvertakingState.OVERTAKING_STARTED, OvertakingState.PASSING)
    # Earliest physical onset was at t=0.4
    assert candidate.start_timestamp == pytest.approx(0.4, 0.05)
    # Confirmation occurred later (after min_consecutive_frames)
    assert candidate.start_confirmation_timestamp is not None
    assert candidate.start_confirmation_timestamp > candidate.start_timestamp
    assert candidate.start_confirmation_delay_seconds is not None
    assert candidate.start_confirmation_delay_seconds > 0.0


def test_2_end_timestamp_selection() -> None:
    """2. End timestamp corresponds to optical exit boundary observation, not confirmation delay."""
    detector = OvertakingDetector()

    # Step A: Ahead
    for i in range(1, 4):
        detector.process(
            [make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))],
            None, None, make_frame(i, i * 0.1)
        )

    # Step B: Approach
    for i in range(4, 9):
        size = 40.0 + (i - 3) * 8.0
        cy = 280.0 + (i - 3) * 30.0
        detector.process(
            [make_perceived(1, make_box(cx=500.0, cy=cy, w=size, h=size))],
            None, None, make_frame(i, i * 0.1)
        )

    # Step C: Passing alongside to left
    for i in range(9, 15):
        t = i * 0.1
        cx = 500.0 - (i - 8) * 50.0
        cy = 430.0 + (i - 8) * 15.0
        detector.process(
            [make_perceived(1, make_box(cx=cx, cy=cy, w=100.0, h=100.0))],
            None, None, make_frame(i, t)
        )

    # Step D: Exit screen at left boundary at t=1.5
    exit_box = BoundingBox(x1=20.0, y1=380.0, x2=120.0, y2=480.0)
    detector.process([make_perceived(1, exit_box)], None, None, make_frame(15, 1.5))

    # Step E: Target disappears; detector confirms completion at t=2.1 (after grace period)
    detector.process([], None, None, make_frame(21, 2.1))

    assert len(detector.completed_events) == 1
    completed = detector.completed_events[0]
    # End timestamp must be the exit frame timestamp (1.5), not confirmation timestamp (2.1)
    assert completed.end_timestamp == pytest.approx(1.5, 0.05)
    assert completed.end_confirmation_timestamp == pytest.approx(2.1, 0.05)
    assert completed.completion_confirmation_delay_seconds == pytest.approx(0.6, 0.05)


def test_3_duration_calculation_precision() -> None:
    """3. Duration = end - start with high precision; non-negative; None for incomplete."""
    evt = OvertakingEvent(
        event_id="evt_precision",
        track_id=1,
        track_instance_id="1_inst1",
        vehicle_class="car",
        start_timestamp=1.123456,
    )
    # Prior to completion, duration is None
    assert evt.duration_seconds is None

    evt.finalize_completion(end_time=3.654321, confirmation_time=3.8)
    assert evt.duration_seconds == pytest.approx(2.530865, abs=1e-6)
    assert evt.duration_seconds >= 0.0

    # Incomplete event has duration None
    evt_inc = OvertakingEvent(
        event_id="evt_inc",
        track_id=2,
        track_instance_id="2_inst1",
        vehicle_class="truck",
        start_timestamp=2.0,
    )
    evt_inc.finalize_incomplete("TRACK_LOST")
    assert evt_inc.duration_seconds is None


def test_4_confirmation_delays() -> None:
    """4. Confirmation delays are recorded accurately and non-negative."""
    evt = OvertakingEvent(
        event_id="evt_delays",
        track_id=1,
        track_instance_id="1_inst1",
        vehicle_class="car",
    )
    evt.confirm_start(start_time=1.10, confirmation_time=1.45)
    assert evt.start_timestamp == 1.10
    assert evt.start_confirmation_timestamp == 1.45
    assert evt.start_confirmation_delay_seconds == pytest.approx(0.35, 1e-5)
    assert evt.start_confirmation_delay_seconds >= 0.0

    evt.finalize_completion(end_time=3.20, confirmation_time=3.70)
    assert evt.end_timestamp == 3.20
    assert evt.end_confirmation_timestamp == 3.70
    assert evt.completion_confirmation_delay_seconds == pytest.approx(0.50, 1e-5)
    assert evt.completion_confirmation_delay_seconds >= 0.0


def test_5_variable_frame_intervals() -> None:
    """5. Variable frame intervals and jitter do not distort timestamps or event progression."""
    detector = OvertakingDetector()
    timestamps = [0.0, 0.033, 0.098, 0.165, 0.280, 0.450, 0.650, 0.890]

    for frame_id, t in enumerate(timestamps, start=1):
        size = 40.0 + frame_id * 6.0
        cy = 280.0 + frame_id * 20.0
        detector.process(
            [make_perceived(1, make_box(cx=550.0, cy=cy, w=size, h=size))],
            None, None, make_frame(frame_id, t)
        )

    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.start_timestamp is not None
    # Timestamps strictly reflect video timestamp values
    assert candidate.start_timestamp in timestamps


def test_6_dropped_frames_resilience() -> None:
    """6. Dropped frames within grace period do not cause premature termination."""
    detector = OvertakingDetector()

    # Step A: Ahead
    for i in range(1, 4):
        detector.process(
            [make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))],
            None, None, make_frame(i, i * 0.1)
        )

    # Step B: Approach into PASSING
    for i in range(4, 9):
        size = 40.0 + (i - 3) * 8.0
        cy = 280.0 + (i - 3) * 30.0
        detector.process(
            [make_perceived(1, make_box(cx=500.0, cy=cy, w=size, h=size))],
            None, None, make_frame(i, i * 0.1)
        )

    assert 1 in detector._candidates
    # Gap of 0.25s (less than track_loss_grace_seconds = 0.4s)
    detector.process([], None, None, make_frame(10, 1.05))

    # Candidate should still be alive during grace period
    assert 1 in detector._candidates
    assert detector._candidates[1].status in (OvertakingEventStatus.ACTIVE, OvertakingEventStatus.IN_PROGRESS)

    # Re-observed at t=1.15
    detector.process(
        [make_perceived(1, make_box(cx=450.0, cy=450.0, w=100.0, h=100.0))],
        None, None, make_frame(11, 1.15)
    )
    assert detector._candidates[1].status in (OvertakingEventStatus.ACTIVE, OvertakingEventStatus.IN_PROGRESS)


def test_7_missing_timestamp_fallback() -> None:
    """7. Missing frame.timestamp falls back gracefully to FPS-synthesized timestamps."""
    detector = OvertakingDetector()

    for i in range(1, 16):
        # frame.timestamp is None
        frame = make_frame(i, timestamp=None)
        size = 40.0 + i * 5.0
        cy = 280.0 + i * 20.0
        detector.process(
            [make_perceived(1, make_box(cx=550.0, cy=cy, w=size, h=size))],
            None, None, frame
        )

    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.start_timestamp is not None
    # Default fps = 30.0, so synthetic timestamp advances ~ 1/30 per frame
    assert candidate.start_timestamp >= 0.0


def test_8_timestamp_monotonicity_enforcement() -> None:
    """8. Non-monotonic or backwards timestamps are clamped to maintain strict monotonicity."""
    detector = OvertakingDetector()

    # t=1.0, then disordered backwards timestamp t=0.5
    f1 = make_frame(1, 1.0)
    detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))], None, None, f1)

    f2 = make_frame(2, 0.5)  # Backwards!
    detector.process([make_perceived(1, make_box(cx=640.0, cy=290.0, w=45.0, h=45.0))], None, None, f2)

    assert detector._last_frame_timestamp is not None
    assert detector._last_frame_timestamp >= 1.0


# ===================================================================== #
# Group 2: Event Lifecycle & Status Transition Tests (Tests 9 to 15)
# ===================================================================== #

def test_9_complete_overtaking_event() -> None:
    """9. Complete lifecycle: VEHICLE_AHEAD -> OVERTAKING_STARTED -> PASSING -> COMPLETED."""
    detector = OvertakingDetector()

    # Ahead (t=0.1..0.3)
    for i in range(1, 4):
        detector.process(
            [make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))],
            None, None, make_frame(i, i * 0.1)
        )

    # Approach (t=0.4..0.8)
    for i in range(4, 9):
        size = 40.0 + (i - 3) * 8.0
        cy = 280.0 + (i - 3) * 30.0
        detector.process(
            [make_perceived(1, make_box(cx=500.0, cy=cy, w=size, h=size))],
            None, None, make_frame(i, i * 0.1)
        )

    # Passing alongside to left (t=0.9..1.4)
    for i in range(9, 15):
        t = i * 0.1
        cx = 500.0 - (i - 8) * 70.0
        cy = 430.0 + (i - 8) * 15.0
        detector.process(
            [make_perceived(1, make_box(cx=cx, cy=cy, w=100.0, h=100.0))],
            None, None, make_frame(i, t)
        )

    # Exit at boundary (t=1.5)
    exit_box = BoundingBox(x1=20.0, y1=380.0, x2=120.0, y2=480.0)
    detector.process([make_perceived(1, exit_box)], None, None, make_frame(15, 1.5))

    # Confirmation after disappearance (t=2.1)
    detector.process([], None, None, make_frame(21, 2.1))

    assert len(detector.completed_events) == 1
    completed = detector.completed_events[0]
    assert completed.status == OvertakingEventStatus.COMPLETED
    assert completed.state == OvertakingState.OVERTAKING_COMPLETED
    assert completed.passing_side == "LEFT"
    assert completed.duration_seconds is not None
    assert completed.duration_seconds > 0.5


def test_10_simultaneous_overtaking_events() -> None:
    """10. Two vehicles overtaken simultaneously on different sides produce distinct events."""
    detector = OvertakingDetector()

    # Both vehicles start ahead
    for i in range(1, 4):
        p1 = make_perceived(1, make_box(cx=500.0, cy=280.0, w=40.0, h=40.0))
        p2 = make_perceived(2, make_box(cx=780.0, cy=280.0, w=40.0, h=40.0))
        detector.process([p1, p2], None, None, make_frame(i, i * 0.1))

    # Both approach
    for i in range(4, 9):
        size = 40.0 + (i - 3) * 8.0
        cy = 280.0 + (i - 3) * 30.0
        p1 = make_perceived(1, make_box(cx=450.0, cy=cy, w=size, h=size))
        p2 = make_perceived(2, make_box(cx=830.0, cy=cy, w=size, h=size))
        detector.process([p1, p2], None, None, make_frame(i, i * 0.1))

    # Both pass alongside towards respective boundaries
    for i in range(9, 15):
        t = i * 0.1
        cx1 = 450.0 - (i - 8) * 60.0
        cx2 = 830.0 + (i - 8) * 60.0
        cy = 430.0 + (i - 8) * 15.0
        p1 = make_perceived(1, make_box(cx=cx1, cy=cy, w=90.0, h=90.0))
        p2 = make_perceived(2, make_box(cx=cx2, cy=cy, w=90.0, h=90.0))
        detector.process([p1, p2], None, None, make_frame(i, t))

    # Both reach boundary at t=1.5
    exit1 = BoundingBox(x1=20.0, y1=400.0, x2=110.0, y2=500.0)
    exit2 = BoundingBox(x1=1170.0, y1=400.0, x2=1260.0, y2=500.0)
    detector.process([make_perceived(1, exit1), make_perceived(2, exit2)], None, None, make_frame(15, 1.5))

    # Both disappear; confirmed at t=2.1
    detector.process([], None, None, make_frame(21, 2.1))

    assert len(detector.completed_events) == 2
    sides = {evt.track_id: evt.passing_side for evt in detector.completed_events}
    assert sides[1] == "LEFT"
    assert sides[2] == "RIGHT"


def test_11_sequential_overtaking_events() -> None:
    """11. Sequential overtaking of distinct vehicles produces distinct events."""
    detector = OvertakingDetector()

    # --- Manoeuvre 1: Vehicle 1 (t=0.1..2.1) ---
    for i in range(1, 4):
        detector.process([make_perceived(1, make_box(cx=500.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))
    for i in range(4, 9):
        size = 40.0 + (i - 3) * 8.0
        cy = 280.0 + (i - 3) * 30.0
        detector.process([make_perceived(1, make_box(cx=450.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))
    detector.process([make_perceived(1, BoundingBox(20.0, 400.0, 110.0, 500.0))], None, None, make_frame(15, 1.5))
    detector.process([], None, None, make_frame(21, 2.1))
    assert len(detector.completed_events) == 1

    # --- Manoeuvre 2: Vehicle 2 (t=3.0..5.0) ---
    for i in range(30, 33):
        detector.process([make_perceived(2, make_box(cx=780.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))
    for i in range(33, 38):
        size = 40.0 + (i - 32) * 8.0
        cy = 280.0 + (i - 32) * 30.0
        detector.process([make_perceived(2, make_box(cx=830.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))
    detector.process([make_perceived(2, BoundingBox(1170.0, 400.0, 1260.0, 500.0))], None, None, make_frame(44, 4.4))
    detector.process([], None, None, make_frame(50, 5.0))

    assert len(detector.completed_events) == 2
    assert detector.completed_events[0].track_id == 1
    assert detector.completed_events[1].track_id == 2
    assert detector.completed_events[0].end_timestamp < detector.completed_events[1].start_timestamp


def test_12_event_never_starts() -> None:
    """12. Vehicle stays ahead at constant distance or turns away; never transitions to start."""
    detector = OvertakingDetector()

    for i in range(1, 15):
        t = i * 0.1
        # Maintains distance and size
        detector.process([make_perceived(1, make_box(cx=640.0, cy=300.0, w=50.0, h=50.0))], None, None, make_frame(i, t))

    assert len(detector.active_events) == 0
    assert len(detector.completed_events) == 0
    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.state == OvertakingState.VEHICLE_AHEAD
    assert candidate.start_timestamp is None


def test_13_event_starts_never_completes() -> None:
    """13. Vehicle starts approach but aborts or accelerates away; marked INCOMPLETE."""
    detector = OvertakingDetector()

    # Step A: Ahead
    for i in range(1, 4):
        detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))

    # Step B: Begins approach (reaches OVERTAKING_STARTED)
    for i in range(4, 8):
        detector.process([make_perceived(1, make_box(cx=600.0, cy=330.0, w=55.0, h=55.0))], None, None, make_frame(i, i * 0.1))

    assert detector._candidates[1].state == OvertakingState.OVERTAKING_STARTED

    # Target accelerates away (shrinks and ascends)
    for i in range(8, 14):
        detector.process([make_perceived(1, make_box(cx=640.0, cy=250.0, w=35.0, h=35.0))], None, None, make_frame(i, i * 0.1))

    # Then drops out in mid-field
    detector.process([], None, None, make_frame(20, 2.0))

    assert len(detector.completed_events) == 0
    assert len(detector.incomplete_events) == 1
    assert detector.incomplete_events[0].status == OvertakingEventStatus.INCOMPLETE


def test_14_video_ending_during_manoeuvre() -> None:
    """14. Video terminates while overtake is in progress; marked INCOMPLETE: VIDEO_ENDED."""
    detector = OvertakingDetector()

    for i in range(1, 8):
        size = 40.0 + i * 6.0
        cy = 280.0 + i * 20.0
        detector.process([make_perceived(1, make_box(cx=600.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))

    # In-flight active manoeuvre
    assert len(detector.active_events) == 1

    # Video terminates
    detector.finish_session(final_timestamp=1.2)

    assert len(detector.active_events) == 0
    assert len(detector.completed_events) == 0
    assert len(detector.incomplete_events) == 1
    inc = detector.incomplete_events[0]
    assert inc.status == OvertakingEventStatus.INCOMPLETE
    assert "VIDEO_ENDED" in inc.uncertainty_reason
    assert inc.duration_seconds is None


def test_15_temporal_confirmation_required() -> None:
    """15. Transient single-frame bounding box spike does not trigger start."""
    detector = OvertakingDetector()

    # Steady ahead frames
    for i in range(1, 4):
        detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))

    # Single-frame sudden spike in box size (detector noise / flicker)
    detector.process([make_perceived(1, make_box(cx=550.0, cy=400.0, w=90.0, h=90.0))], None, None, make_frame(4, 0.4))

    # Next frame immediately returns to baseline
    detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(5, 0.5))

    candidate = detector._candidates.get(1)
    assert candidate is not None
    # Must NOT have confirmed an overtake start on a 1-frame transient spike
    assert candidate.state != OvertakingState.PASSING
    assert candidate.start_confirmation_timestamp is None


# ===================================================================== #
# Group 3: Tracking Interruptions, Grace Periods & Robustness (Tests 16 to 24)
# ===================================================================== #

def test_16_temporary_track_loss_recovery() -> None:
    """16. Temporary track loss within grace period recovers with continuous identity."""
    detector = OvertakingDetector()

    # Establish approach into PASSING
    for i in range(1, 9):
        size = 40.0 + i * 7.0
        cy = 280.0 + i * 25.0
        detector.process([make_perceived(1, make_box(cx=500.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))

    # Tracking drops out for 0.2s (less than 0.4s grace)
    detector.process([], None, None, make_frame(9, 0.9))
    detector.process([], None, None, make_frame(10, 1.0))

    # Re-appears at t=1.1
    detector.process([make_perceived(1, make_box(cx=450.0, cy=480.0, w=96.0, h=96.0))], None, None, make_frame(11, 1.1))

    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.status in (OvertakingEventStatus.ACTIVE, OvertakingEventStatus.IN_PROGRESS)
    assert candidate.identity_continuity in (IdentityContinuityStatus.CONTINUOUS, IdentityContinuityStatus.RECOVERED)


def test_17_long_track_loss_incomplete() -> None:
    """17. Prolonged track loss exceeding grace period marks event INCOMPLETE."""
    detector = OvertakingDetector()

    for i in range(1, 8):
        size = 40.0 + i * 6.0
        cy = 280.0 + i * 20.0
        detector.process([make_perceived(1, make_box(cx=600.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))

    # Long absence > 0.4s grace period in mid-field
    detector.process([], None, None, make_frame(15, 1.5))

    assert len(detector.completed_events) == 0
    assert len(detector.incomplete_events) == 1
    assert "LOST" in detector.incomplete_events[0].uncertainty_reason


def test_18_id_switch_reassociation() -> None:
    """18. ID switch during manoeuvre is reassociated with suspected switch flag."""
    detector = OvertakingDetector()

    # Track 1 active in PASSING
    for i in range(1, 9):
        size = 40.0 + i * 7.0
        cy = 280.0 + i * 25.0
        detector.process([make_perceived(1, make_box(cx=450.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))

    # Frame 9: Track 1 disappears; Track 10 appears at nearby position (cx=440, cy=485)
    detector.process([make_perceived(10, make_box(cx=440.0, cy=485.0, w=96.0, h=96.0))], None, None, make_frame(9, 0.9))

    # Reassociated candidate under track 10
    cand = detector._candidates.get(10)
    assert cand is not None
    assert cand.identity_continuity == IdentityContinuityStatus.SUSPECTED_ID_SWITCH
    assert 1 in cand.reassociated_track_ids


def test_19_tracker_id_reuse_isolation() -> None:
    """19. Track ID reused after large temporal gap is isolated into a fresh episode."""
    detector = OvertakingDetector()

    # Episode 1: Track 1 at t=1.0
    detector.process([make_perceived(1, make_box(cx=640.0, cy=300.0, w=40.0, h=40.0))], None, None, make_frame(1, 1.0))
    inst1 = detector._current_instance_id[1]
    assert inst1 == "1_inst1"

    # Gap of 3 seconds
    detector.process([], None, None, make_frame(10, 4.0))

    # Episode 2: Track 1 re-appears at t=4.5
    detector.process([make_perceived(1, make_box(cx=640.0, cy=300.0, w=40.0, h=40.0))], None, None, make_frame(15, 4.5))
    inst2 = detector._current_instance_id[1]
    assert inst2 == "1_inst2"
    assert inst1 != inst2


def test_20_temporary_occlusion() -> None:
    """20. Temporary occlusion is tolerated within grace window and completes cleanly."""
    detector = OvertakingDetector()

    # Approach
    for i in range(1, 9):
        size = 40.0 + i * 7.0
        cy = 280.0 + i * 25.0
        detector.process([make_perceived(1, make_box(cx=500.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))

    # Temporary occlusion for 1 frame at t=0.9
    detector.process([], None, None, make_frame(9, 0.9))

    # Re-appears at t=1.0, reaches left boundary at t=1.3
    detector.process([make_perceived(1, make_box(cx=300.0, cy=460.0, w=95.0, h=95.0))], None, None, make_frame(10, 1.0))
    detector.process([make_perceived(1, BoundingBox(20.0, 400.0, 110.0, 500.0))], None, None, make_frame(13, 1.3))
    # Exits
    detector.process([], None, None, make_frame(18, 1.8))

    assert len(detector.completed_events) == 1
    assert detector.completed_events[0].status == OvertakingEventStatus.COMPLETED


def test_21_lane_data_unavailable_fallback() -> None:
    """21. Functions reliably with geometric corridor when lane detection is None."""
    detector = OvertakingDetector()

    # Lane detection is explicitly None
    for i in range(1, 6):
        detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=45.0, h=45.0))], None, None, make_frame(i, i * 0.1))

    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.latest_corridor_position == CorridorPosition.IN_EGO_CORRIDOR


def test_22_invalid_depth_handling() -> None:
    """22. Missing or uncalibrated depth operates reliably via optical bounding box scale."""
    detector = OvertakingDetector()

    for i in range(1, 4):
        obj = make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0), depth=0.0, depth_confidence=0.0)
        detector.process([obj], None, None, make_frame(i, i * 0.1))

    for i in range(4, 9):
        size = 40.0 + (i - 3) * 7.0
        cy = 280.0 + (i - 3) * 20.0
        obj = make_perceived(1, make_box(cx=600.0, cy=cy, w=size, h=size), depth=0.0, depth_confidence=0.0)
        detector.process([obj], None, None, make_frame(i, i * 0.1))

    candidate = detector._candidates.get(1)
    assert candidate is not None
    assert candidate.state in (OvertakingState.OVERTAKING_STARTED, OvertakingState.PASSING)
    assert candidate.start_timestamp is not None


def test_23_ambiguous_completion_evidence() -> None:
    """23. Ambiguous disappearance without clear boundary is classified as UNCERTAIN or INCOMPLETE."""
    detector = OvertakingDetector()

    # Establish approach
    for i in range(1, 7):
        size = 40.0 + i * 5.0
        cy = 280.0 + i * 15.0
        detector.process([make_perceived(1, make_box(cx=600.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))

    cand = detector._candidates.get(1)
    assert cand is not None
    cand.is_uncertain = True

    # Disappears in mid-field
    detector.process([], None, None, make_frame(15, 1.5))

    # Must NOT be classified as COMPLETED
    assert len(detector.completed_events) == 0
    assert (len(detector.uncertain_events) + len(detector.incomplete_events)) > 0


def test_24_duplicate_event_prevention() -> None:
    """24. Single physical manoeuvre produces exactly one completed event record."""
    detector = OvertakingDetector()

    for i in range(1, 4):
        detector.process([make_perceived(1, make_box(cx=500.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))
    for i in range(4, 9):
        size = 40.0 + (i - 3) * 8.0
        cy = 280.0 + (i - 3) * 30.0
        detector.process([make_perceived(1, make_box(cx=450.0, cy=cy, w=size, h=size))], None, None, make_frame(i, i * 0.1))
    detector.process([make_perceived(1, BoundingBox(20.0, 400.0, 110.0, 500.0))], None, None, make_frame(15, 1.5))
    detector.process([], None, None, make_frame(21, 2.1))
    detector.process([], None, None, make_frame(25, 2.5))

    assert len(detector.completed_events) == 1
    assert len([e for e in detector.all_events if e.track_id == 1 and e.status == OvertakingEventStatus.COMPLETED]) == 1


# ===================================================================== #
# Group 4: Session Lifecycle & Reset Cleanliness (Tests 25 to 27)
# ===================================================================== #

def test_25_sequential_videos_no_leakage() -> None:
    """25. Sequential video runs with finish_session() prevent cross-video leakage."""
    detector = OvertakingDetector()

    # Video 1
    for i in range(1, 5):
        detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))
    detector.finish_session(final_timestamp=0.5)

    assert len(detector.active_events) == 0

    # Video 2 starts with fresh session
    detector.start_session(initial_timestamp=0.0)
    assert len(detector.active_events) == 0
    assert len(detector._candidates) == 0
    assert detector._last_frame_timestamp == 0.0


def test_26_empty_event_list() -> None:
    """26. Video with no vehicles produces empty event lists without false positives."""
    detector = OvertakingDetector()

    for i in range(1, 20):
        detector.process([], None, None, make_frame(i, i * 0.1))

    assert len(detector.all_events) == 0
    assert len(detector.active_events) == 0
    assert len(detector.completed_events) == 0
    assert len(detector.incomplete_events) == 0
    assert len(detector.uncertain_events) == 0


def test_27_reinitialization_clears_state() -> None:
    """27. Resetting detector instance cleanly clears all internal state."""
    detector = OvertakingDetector()

    for i in range(1, 5):
        detector.process([make_perceived(1, make_box(cx=640.0, cy=280.0, w=40.0, h=40.0))], None, None, make_frame(i, i * 0.1))

    assert len(detector.all_events) > 0

    detector.reset()

    assert len(detector.active_events) == 0
    assert len(detector.completed_events) == 0
    assert len(detector.incomplete_events) == 0
    assert len(detector.uncertain_events) == 0
    assert len(detector.all_events) == 0
    assert len(detector._candidates) == 0
    assert len(detector._track_histories) == 0
    assert len(detector._instance_counters) == 0
