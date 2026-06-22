from __future__ import annotations

import numpy as np
import pytest

from src.fusion.object_fusion import ObjectFusion
from src.schemas import BoundingBox, PerceivedObject, TrackedObject


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _make_tracked(
    track_id: int,
    x1: float = 10,
    y1: float = 20,
    x2: float = 110,
    y2: float = 120,
    class_name: str = "car",
    confidence: float = 0.9,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1, y1, x2, y2),
    )


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestObjectFusion:
    def test_fuse_without_depth(self) -> None:
        """When depth_map is None, all fields are 0 / zero confidence."""
        fusion = ObjectFusion()
        tracked = [_make_tracked(1), _make_tracked(2)]
        perceived = fusion.fuse(tracked, depth_map=None)

        assert len(perceived) == 2
        for p in perceived:
            assert isinstance(p, PerceivedObject)
            assert p.depth == 0.0
            assert p.distance == 0.0
            assert p.depth_confidence == 0.0

    def test_fuse_with_depth(self) -> None:
        """Median depth over centre crop matches a known gradient."""
        h, w = 240, 320
        # Synthetic depth map: larger Y = farther (simple gradient)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)
        depth_map = np.tile(y_coords, (1, w))  # top=0, bottom=239

        # Bbox in top half of image: y1=20, y2=70 → centre ~45
        # Centre sub-box (ratio=0.5): bbox h=50 → sub_h=25
        #   y-centre=45, sub range: y1=32.5, y2=57.5 → int [33, 57)
        # Median of depth[33:57, :] should be ~45
        tracked = [_make_tracked(1, x1=50, y1=20, x2=150, y2=70)]

        fusion = ObjectFusion(
            sampling="median_center", center_box_ratio=0.5
        )
        perceived = fusion.fuse(tracked, depth_map)

        assert len(perceived) == 1
        p = perceived[0]
        assert p.track_id == 1
        assert p.class_name == "car"

        # Median depth value should be close to the vertical centre of the
        # sub-box, which is ~45
        assert 40.0 <= p.depth <= 50.0, f"depth={p.depth} not in [40, 50]"
        assert p.depth_confidence > 0.9  # all pixels are valid

    def test_fuse_with_depth_min_strategy(self) -> None:
        """'min' strategy returns the minimum depth in the centre crop."""
        h, w = 100, 100
        depth_map = np.ones((h, w), dtype=np.float32) * 50.0
        depth_map[40:60, 40:60] = 5.0  # central region has low depth

        tracked = [_make_tracked(1, x1=0, y1=0, x2=100, y2=100)]

        fusion = ObjectFusion(sampling="min", center_box_ratio=0.5)
        perceived = fusion.fuse(tracked, depth_map)

        assert len(perceived) == 1
        assert perceived[0].depth == pytest.approx(5.0, abs=0.5)

    def test_fuse_multiple_objects(self) -> None:
        """Multiple tracked objects each get their own PerceivedObject."""
        h, w = 100, 200
        depth_map = np.ones((h, w), dtype=np.float32) * 10.0

        tracked = [
            _make_tracked(1, 0, 0, 50, 50, class_name="car"),
            _make_tracked(2, 100, 50, 180, 90, class_name="person"),
        ]
        fusion = ObjectFusion()
        perceived = fusion.fuse(tracked, depth_map)

        assert len(perceived) == 2
        assert perceived[0].track_id == 1
        assert perceived[0].class_name == "car"
        assert perceived[1].track_id == 2
        assert perceived[1].class_name == "person"
        assert perceived[0].depth == pytest.approx(10.0, abs=1.0)
        assert perceived[1].depth == pytest.approx(10.0, abs=1.0)

    def test_fuse_empty_tracked(self) -> None:
        fusion = ObjectFusion()
        perceived = fusion.fuse([], np.zeros((100, 100), dtype=np.float32))
        assert perceived == []

    def test_fuse_nan_in_depth(self) -> None:
        """NaN pixels should be ignored in the estimate."""
        h, w = 100, 100
        depth_map = np.full((h, w), np.nan, dtype=np.float32)
        depth_map[45:55, 45:55] = 7.5  # only centre is valid

        tracked = [_make_tracked(1, x1=0, y1=0, x2=100, y2=100)]
        fusion = ObjectFusion(min_valid_depth_pixels=5)
        perceived = fusion.fuse(tracked, depth_map)

        assert len(perceived) == 1
        assert perceived[0].depth == pytest.approx(7.5, abs=0.5)
        assert perceived[0].depth_confidence > 0

    def test_unknown_sampling_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown_strategy"):
            ObjectFusion(sampling="unknown_strategy")
