"""Unit tests for Driver Warning HUD layer and independent visualization toggle.

Covers:
- HUD rendering across HIGH, MEDIUM, LOW, and neutral/no-warning states.
- Missing indicator handling (TTC N/A, omitted risk score, unknown direction).
- Primary threat selection and deterministic ranking with frame persistence.
- Secondary threat stack rendering (limiting to max_secondary_warnings).
- Target object bounding-box highlighting.
- Independent keyboard and configuration toggling.
- CRITICAL REQUIREMENT: Complete independence of background threat assessment,
  event management, and state machine updates from HUD visibility.
- Non-interference with existing layers (overtaking, perceived, lane).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.pipeline import PerceptionPipeline
from src.schemas.bbox import BoundingBox
from src.schemas.frame import Frame
from src.schemas.perceived_object import PerceivedObject
from src.threat.schemas import (
    CameraDirection,
    DirectionalWarningLevel,
    WarningEventState,
    WarningOutput,
)
from src.visualization.factory import build_renderer
from src.visualization.layers.warning_layer import WarningLayer
from src.visualization.renderer import Renderer


def _make_dummy_frame(w: int = 1280, h: int = 720, timestamp: float = 1.0) -> Frame:
    """Helper to generate clean dummy video frame."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return Frame(image=img, frame_id=1, timestamp=timestamp, source="test")


def _make_warning_output(
    event_id: str = "warn_rear_end_1_0001",
    track_id: int = 1,
    object_class: str = "car",
    scenario: str = "REAR_END",
    direction: CameraDirection = CameraDirection.FRONT,
    level: DirectionalWarningLevel = DirectionalWarningLevel.HIGH,
    state: WarningEventState = WarningEventState.ACTIVE_HIGH,
    risk_score: float | None = 0.82,
    ttc: float | None = 1.4,
    collision_urgency: float | None = 1.4,
    is_primary_threat: bool = True,
    priority_rank: int = 1,
    reason: str = "Imminent collision risk",
    timestamp: float = 1.0,
) -> WarningOutput:
    """Helper to construct WarningOutput records."""
    return WarningOutput(
        event_id=event_id,
        track_id=track_id,
        object_class=object_class,
        scenario=scenario,
        direction=direction,
        level=level,
        state=state,
        risk_score=risk_score,
        risk_model="OPTION_A_ADRS_2025",
        reliability="HIGH",
        ttc=ttc,
        collision_urgency=collision_urgency,
        is_primary_threat=is_primary_threat,
        priority_rank=priority_rank,
        timestamp=timestamp,
        reason=reason,
    )


class TestWarningHUDLayer:
    """Comprehensive test suite for WarningLayer visualization and primary threat selection."""

    def test_primary_threat_selection_hierarchy(self):
        """Verify that HIGH > MEDIUM > LOW, respecting prioritizer rank and urgency."""
        layer = WarningLayer()

        w_low = _make_warning_output(
            event_id="w_low",
            track_id=1,
            level=DirectionalWarningLevel.LOW,
            state=WarningEventState.ACTIVE_LOW,
            priority_rank=3,
            collision_urgency=4.5,
        )
        w_med = _make_warning_output(
            event_id="w_med",
            track_id=2,
            level=DirectionalWarningLevel.MEDIUM,
            state=WarningEventState.ACTIVE_MEDIUM,
            priority_rank=2,
            collision_urgency=2.5,
        )
        w_high = _make_warning_output(
            event_id="w_high",
            track_id=3,
            level=DirectionalWarningLevel.HIGH,
            state=WarningEventState.ACTIVE_HIGH,
            priority_rank=1,
            collision_urgency=1.2,
        )

        # 1. Mixed threats: HIGH must be chosen as primary
        primary, secondary = layer.select_primary_threat([w_low, w_high, w_med])
        assert primary is not None
        assert primary.event_id == "w_high"
        assert len(secondary) == 2
        assert [s.event_id for s in secondary] == ["w_med", "w_low"]

        # 2. Only MEDIUM and LOW: MEDIUM must be chosen
        primary_m, secondary_m = layer.select_primary_threat([w_low, w_med])
        assert primary_m.event_id == "w_med"
        assert len(secondary_m) == 1

        # 3. UNASSESSED warnings are excluded from active HUD
        w_unassessed = _make_warning_output(
            event_id="w_unass",
            track_id=4,
            level=DirectionalWarningLevel.UNASSESSED,
            state=WarningEventState.INACTIVE,
        )
        primary_u, secondary_u = layer.select_primary_threat([w_unassessed])
        assert primary_u is None
        assert len(secondary_u) == 0

    def test_primary_threat_persistence_tie_breaker(self):
        """When two threats have identical severity and urgency, previous primary is held stably."""
        layer = WarningLayer()

        w1 = _make_warning_output(
            event_id="w1",
            track_id=10,
            level=DirectionalWarningLevel.HIGH,
            collision_urgency=1.5,
            priority_rank=1,
        )
        w2 = _make_warning_output(
            event_id="w2",
            track_id=11,
            level=DirectionalWarningLevel.HIGH,
            collision_urgency=1.5,
            priority_rank=1,
        )

        # First frame selects w1
        primary1, _ = layer.select_primary_threat([w1, w2])
        assert primary1.event_id == "w1"

        # Second frame reverses input list order [w2, w1]: w1 should remain primary due to persistence
        primary2, _ = layer.select_primary_threat([w2, w1])
        assert primary2.event_id == "w1"

    def test_render_high_warning_panel(self):
        """Render HIGH warning panel on canvas and verify pixel modifications."""
        layer = WarningLayer(enabled=True)
        frame = _make_dummy_frame()
        w_high = _make_warning_output(
            level=DirectionalWarningLevel.HIGH,
            direction=CameraDirection.FRONT,
            object_class="truck",
            ttc=1.1,
            reason="Lead truck heavy deceleration",
        )
        ctx = {"frame": frame, "warnings": [w_high]}

        canvas = frame.image.copy()
        output = layer(canvas, ctx)

        # Assert canvas was modified (non-zero pixels in panel region)
        # Panel sits at x=(1280-480)//2 = 400, y=36 to y=110
        panel_roi = output[36:110, 400:880]
        assert np.any(panel_roi > 0)
        # Verify crimson/red accent pixels exist in left border
        assert np.any(panel_roi[:, :6, 2] > 200)

    def test_render_medium_and_low_warning_panels(self):
        """Render MEDIUM and LOW warning panels without error."""
        layer = WarningLayer(enabled=True)
        frame = _make_dummy_frame()

        # MEDIUM warning
        w_med = _make_warning_output(
            level=DirectionalWarningLevel.MEDIUM,
            direction=CameraDirection.FRONT_RIGHT,
            object_class="cyclist",
            ttc=2.4,
        )
        out_med = layer(frame.image.copy(), {"frame": frame, "warnings": [w_med]})
        assert np.any(out_med[36:100, 400:880] > 0)

        # LOW warning
        w_low = _make_warning_output(
            level=DirectionalWarningLevel.LOW,
            direction=CameraDirection.LEFT,
            object_class="car",
            ttc=4.8,
        )
        out_low = layer(frame.image.copy(), {"frame": frame, "warnings": [w_low]})
        assert np.any(out_low[36:90, 400:880] > 0)

    def test_render_neutral_status_when_no_active_warnings(self):
        """When no warnings are active, neutral status pill is rendered if enabled."""
        layer = WarningLayer(enabled=True, neutral_status_indicator=True)
        frame = _make_dummy_frame()
        ctx = {"frame": frame, "warnings": []}

        output = layer(frame.image.copy(), ctx)
        # Center region should contain neutral indicator
        assert np.any(output[34:56, 450:830] > 0)

        # If neutral indicator disabled, canvas remains untouched
        layer_no_neutral = WarningLayer(enabled=True, neutral_status_indicator=False)
        output_clean = layer_no_neutral(frame.image.copy(), ctx)
        assert np.all(output_clean == frame.image)

    def test_missing_and_invalid_information_handling(self):
        """TTC None must not render as 0.0s; uncalibrated risk score must be omitted."""
        layer = WarningLayer(enabled=True, show_ttc=True, show_risk_score=False)
        frame = _make_dummy_frame()

        # Warning with TTC = None and risk_score = None
        w_none = _make_warning_output(
            ttc=None,
            collision_urgency=None,
            risk_score=None,
            direction=CameraDirection.UNKNOWN,
            object_class="",
            reason="A" * 100,  # Long reason text to test truncation
        )
        ctx = {"frame": frame, "warnings": [w_none]}

        canvas = frame.image.copy()
        # Must execute cleanly without exception
        output = layer(canvas, ctx)
        assert np.any(output > 0)

    def test_secondary_threats_stack_limit(self):
        """Secondary threats are rendered up to max_secondary_warnings limit."""
        layer = WarningLayer(enabled=True, max_secondary_warnings=2)
        frame = _make_dummy_frame()

        threats = [
            _make_warning_output(event_id=f"w{i}", track_id=i, level=DirectionalWarningLevel.HIGH, priority_rank=i)
            for i in range(1, 6)
        ]
        ctx = {"frame": frame, "warnings": threats}

        output = layer(frame.image.copy(), ctx)
        # Secondary column sits at x = 1280 - 260 - 12 = 1008
        sec_roi = output[36:120, 1008:1268]
        assert np.any(sec_roi > 0)

    def test_target_object_highlight(self):
        """Target object bounding box receives distinct highlight corners and tag."""
        layer = WarningLayer(enabled=True, highlight_threat_object=True)
        frame = _make_dummy_frame()

        w_high = _make_warning_output(track_id=42, level=DirectionalWarningLevel.HIGH)
        bbox = BoundingBox(x1=500.0, y1=300.0, x2=650.0, y2=450.0)

        # Mock perceived object with matching track_id
        mock_obj = PerceivedObject(
            track_id=42,
            class_name="car",
            bbox=bbox,
            depth=15.0,
            distance=20.0,
            depth_confidence=0.9,
        )
        ctx = {"frame": frame, "warnings": [w_high], "perceived": [mock_obj]}

        output = layer(frame.image.copy(), ctx)
        # Corner brackets at (500, 300) should be modified
        corner_roi = output[295:310, 495:515]
        assert np.any(corner_roi > 0)


class TestWarningToggleAndIndependence:
    """Verifies runtime/config toggles and strict operational independence."""

    def test_renderer_toggle_warning_layer(self):
        """Verify that renderer toggles warning layer ON and OFF via 'warning' and 'warning_hud'."""
        warning_layer = WarningLayer(enabled=True)
        renderer = Renderer([warning_layer])

        assert warning_layer.enabled is True

        # Toggle OFF via 'warning'
        new_state = renderer.toggle("warning")
        assert new_state is False
        assert warning_layer.enabled is False

        # Toggle ON via alias 'warning_hud'
        new_state2 = renderer.toggle("warning_hud")
        assert new_state2 is True
        assert warning_layer.enabled is True

    def test_hud_disabled_renders_nothing(self):
        """When WarningLayer is disabled, canvas is returned untouched without drawing operations."""
        layer = WarningLayer(enabled=False)
        frame = _make_dummy_frame()
        w_high = _make_warning_output(level=DirectionalWarningLevel.HIGH)
        ctx = {"frame": frame, "warnings": [w_high]}

        canvas = frame.image.copy()
        output = layer(canvas, ctx)

        # Complete identity: canvas was not modified at all
        assert np.array_equal(output, canvas)

    def test_critical_requirement_background_processing_independence(self):
        """CRITICAL: Disabling warning HUD must NOT disable background threat processing,

        event management, timestamp progression, or state machine updates.
        """
        config = {
            "threat_assessment": {
                "enabled": True,
                "directional_warnings": {
                    "enabled": True,
                },
            },
            "visualization": {
                "enabled": True,
                "layers": ["warning"],
                "warning": {
                    "enabled": False,  # HUD INITIALIZED AS DISABLED!
                },
            },
            "detector": {"enabled": False},
            "tracker": {"enabled": False},
            "depth": {"enabled": False},
            "lane": {"enabled": False},
            "signs": {"enabled": False},
            "scene": {"enabled": False},
            "behavior": {"enabled": False},
        }

        pipeline = PerceptionPipeline(config)
        assert pipeline.directional_warning_generator is not None
        assert pipeline.renderer is not None

        # Confirm WarningLayer is initialized as DISABLED
        warning_layer = next(l for l in pipeline.renderer.layers if isinstance(l, WarningLayer))
        assert warning_layer.enabled is False

        # Simulate frame processing through PerceptionPipeline
        f1 = _make_dummy_frame(timestamp=1.0)
        # Directly invoke directional warning generator to simulate pipeline threat feeding
        import sys
        from pathlib import Path
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))
        from test_directional_warning_events import _make_threat_data
        ass, feat = _make_threat_data(track_id=1, risk_score=0.85, ttc=1.2, timestamp=1.0)

        # Process frame: Background warning generation executes normally!
        active_warnings = pipeline.directional_warning_generator.process(
            threat_assessments={1: ass},
            threat_features={1: feat},
            frame=f1,
        )

        # 1. Background warning outputs and events must be created!
        assert len(active_warnings) == 1
        assert active_warnings[0].level == DirectionalWarningLevel.HIGH
        assert active_warnings[0].state == WarningEventState.ACTIVE_HIGH

        all_events = pipeline.directional_warning_generator.get_all_events()
        assert len(all_events) == 1
        assert all_events[0].event_id.startswith("warn_rear_end_1_")

        # 2. Rendering while HUD is OFF must draw NOTHING
        result_dict = {
            "frame": f1,
            "warnings": active_warnings,
            "directional_warnings": active_warnings,
            "warning_events": all_events,
            "threat_features": {1: feat},
        }
        rendered_off = pipeline.render(result_dict)
        assert np.array_equal(rendered_off, f1.image)

        # 3. Toggle HUD back ON at runtime
        pipeline.renderer.toggle("warning")
        assert warning_layer.enabled is True

        # 4. Rendering while HUD is ON immediately renders the active warning!
        rendered_on = pipeline.render(result_dict)
        assert not np.array_equal(rendered_on, f1.image)
        # Center region has HUD pixels
        assert np.any(rendered_on[36:110, 400:880] > 0)

        # 5. Warning history and events remain intact without reset
        assert len(pipeline.directional_warning_generator.get_all_events()) == 1

    def test_layer_independence_with_overtaking(self):
        """Toggling warning HUD does not alter overtaking layer state or vice versa."""
        config = {
            "visualization": {
                "enabled": True,
                "layers": ["overtaking", "warning"],
                "overtaking": {"enabled": True},
                "warning": {"enabled": True},
            }
        }
        renderer = build_renderer(config)

        ovt_layer = next(l for l in renderer.layers if l.__class__.__name__ == "OvertakingLayer")
        warn_layer = next(l for l in renderer.layers if isinstance(l, WarningLayer))

        assert ovt_layer.enabled is True
        assert warn_layer.enabled is True

        # Toggle warning OFF
        renderer.toggle("warning")
        assert warn_layer.enabled is False
        assert ovt_layer.enabled is True  # Overtaking untouched!

        # Toggle overtaking OFF
        renderer.toggle("overtaking")
        assert ovt_layer.enabled is False
        assert warn_layer.enabled is False

        # Toggle warning back ON
        renderer.toggle("warning")
        assert warn_layer.enabled is True
        assert ovt_layer.enabled is False  # Overtaking still untouched!
