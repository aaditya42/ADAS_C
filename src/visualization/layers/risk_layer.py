from __future__ import annotations

import numpy as np

from src.schemas import PerceivedObject, RiskObject
from src.visualization.layers._utils import draw_label, risk_color
from src.visualization.layers.base_layer import BaseLayer


class RiskLayer(BaseLayer):
    """Draw risk badge near each perceived object."""

    def __call__(self, canvas: np.ndarray, ctx: dict) -> np.ndarray:
        if not self._enabled:
            return canvas

        perceived: list[PerceivedObject] = ctx.get("perceived", [])
        risk_list: list[RiskObject] | None = ctx.get("risk")
        if risk_list is None:
            return canvas

        risk_by_id: dict[int, RiskObject] = {r.track_id: r for r in risk_list}
        obj_by_id: dict[int, PerceivedObject] = {o.track_id: o for o in perceived}

        for tid, r in risk_by_id.items():
            obj = obj_by_id.get(tid)
            if obj is None:
                continue

            color = risk_color(r.risk_level)
            x2 = int(round(obj.bbox.x2))
            y2 = int(round(obj.bbox.y2))

            draw_label(
                canvas, x2 - 40, y2,
                f"{r.risk_level.name} {r.risk_score:.0f}",
                color,
            )

        return canvas
