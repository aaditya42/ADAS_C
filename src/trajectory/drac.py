from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DRACResult:
    """Calculated DRAC metric and pairwise vehicle state.

    Attributes:
        following_track_id: Identifier of following vehicle (e.g. "EGO" or int track ID).
        leading_track_id: Identifier of leading vehicle.
        following_speed_mps: Physical longitudinal speed of follower in m/s.
        leading_speed_mps: Physical longitudinal speed of leader in m/s.
        inter_vehicle_gap_m: Physical longitudinal gap in metres.
        drac_mps2: Calculated DRAC in m/s^2, or None if unavailable.
        drac_status: Status ("VALID", "ZERO_CLOSING_SPEED", or "UNAVAILABLE").
    """

    following_track_id: str | int | None = None
    leading_track_id: str | int | None = None
    following_speed_mps: float | None = None
    leading_speed_mps: float | None = None
    inter_vehicle_gap_m: float | None = None
    drac_mps2: float | None = None
    drac_status: str = "UNAVAILABLE"


def kmh_to_mps(speed_kmh: float) -> float:
    """Convert speed from km/h to m/s."""
    return float(speed_kmh) / 3.6


def mps_to_kmh(speed_mps: float) -> float:
    """Convert speed from m/s to km/h."""
    return float(speed_mps) * 3.6


def cm_to_m(distance_cm: float) -> float:
    """Convert distance from centimetres to metres."""
    return float(distance_cm) / 100.0


def calculate_drac(
    following_speed_mps: float | None,
    leading_speed_mps: float | None,
    gap_m: float | None,
    minimum_gap_meters: float = 0.01,
    is_metric: bool = True,
    require_metric: bool = True,
) -> tuple[float | None, str]:
    """Calculate Deceleration Rate to Avoid Collision (DRAC).

    DRAC = (v_f - v_l)^2 / (2 * d)

    Where:
        v_f: Following vehicle's longitudinal speed in m/s.
        v_l: Leading vehicle's longitudinal speed in m/s.
        d:   Longitudinal inter-vehicle gap in metres.
        DRAC: Required deceleration in m/s^2.

    Returns:
        (drac_mps2, drac_status)
        Where drac_status is one of:
        - "VALID": Positive closing speed (v_f > v_l), positive gap, metric calibrated.
        - "ZERO_CLOSING_SPEED": Non-closing case (v_f <= v_l), positive gap, metric calibrated.
        - "UNAVAILABLE": Missing speeds/gap, gap <= 0, uncalibrated units, or invalid values.
    """
    # Unit validation: if metric units required but distance/speeds not calibrated
    if require_metric and not is_metric:
        return None, "UNAVAILABLE"

    # Input validation: check for missing values
    if following_speed_mps is None or leading_speed_mps is None or gap_m is None:
        return None, "UNAVAILABLE"

    try:
        vf = float(following_speed_mps)
        vl = float(leading_speed_mps)
        d = float(gap_m)
    except (TypeError, ValueError):
        return None, "UNAVAILABLE"

    # Check for NaN or inf
    if not (math.isfinite(vf) and math.isfinite(vl) and math.isfinite(d)):
        return None, "UNAVAILABLE"

    # Inter-vehicle gap validation: must be strictly positive and >= minimum threshold
    if d <= 0.0 or d < minimum_gap_meters:
        return None, "UNAVAILABLE"

    # Non-closing case: following vehicle is not travelling faster than leading vehicle
    if vf <= vl:
        return 0.0, "ZERO_CLOSING_SPEED"

    # Closing case: compute DRAC = (v_f - v_l)^2 / (2 * d)
    closing_speed = vf - vl
    drac = (closing_speed**2) / (2.0 * d)
    return float(drac), "VALID"


def associate_drac_pair(
    target_track_id: int,
    target_class_name: str,
    target_distance: float | None,
    target_depth_confidence: float,
    all_objects: list[Any],
    motion_dict: dict[int, Any] | None,
    lane_info: Any = None,
    scene_context: Any = None,
    distance_type: str = "relative",
    drac_config: dict[str, Any] | None = None,
) -> DRACResult:
    """Identify plausible following/leading vehicle association and calculate DRAC.

    Reuses existing TTC and SceneContext lane association to avoid inventing
    arbitrary pairs.
    """
    cfg = drac_config or {}
    if not cfg.get("enabled", True):
        return DRACResult(drac_status="UNAVAILABLE")

    require_metric = cfg.get("require_metric_units", True)
    is_metric = distance_type == "calibrated"

    # If physical metric calibration is missing and required, keep UNAVAILABLE
    if require_metric and not is_metric:
        return DRACResult(drac_status="UNAVAILABLE")

    # In forward-facing camera ADAS, check if target object is in ego lane
    # (i.e. ego vehicle following lead vehicle)
    in_ego_lane = False

    # Check via SceneContext if available
    if scene_context is not None:
        nearest_tid = getattr(scene_context, "nearest_in_lane_track_id", None)
        if nearest_tid == target_track_id:
            in_ego_lane = True

    # Check via target distance and class
    vehicle_classes = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "van",
        "vehicle",
    }
    is_vehicle = target_class_name.lower() in vehicle_classes

    # If target is a vehicle and either marked in-lane or is nearest lead object
    if (in_ego_lane or (scene_context is None and is_vehicle)) and target_distance is not None:
        if target_depth_confidence > 0 and target_distance > 0:
            following_id = "EGO"
            leading_id = target_track_id

            # Determine ego speed
            ego_speed: float | None = cfg.get("default_ego_speed_mps")
            if ego_speed is None and scene_context is not None:
                limit_kmh = getattr(scene_context, "active_speed_limit", None)
                if limit_kmh is not None:
                    ego_speed = kmh_to_mps(limit_kmh)

            # Determine lead speed from relative closing speed
            mot = motion_dict.get(target_track_id) if motion_dict is not None else None
            v_rel = mot.speed_mps if (mot is not None and mot.speed_mps is not None) else None

            if ego_speed is not None and v_rel is not None:
                vf = float(ego_speed)
                vl = float(ego_speed - v_rel)
            elif ego_speed is not None:
                vf = float(ego_speed)
                vl = None
            else:
                vf = None
                vl = None

            min_gap = float(cfg.get("minimum_gap_meters", 0.01))
            drac_val, status = calculate_drac(
                following_speed_mps=vf,
                leading_speed_mps=vl,
                gap_m=target_distance,
                minimum_gap_meters=min_gap,
                is_metric=is_metric,
                require_metric=require_metric,
            )

            return DRACResult(
                following_track_id=following_id,
                leading_track_id=leading_id,
                following_speed_mps=vf,
                leading_speed_mps=vl,
                inter_vehicle_gap_m=target_distance,
                drac_mps2=drac_val,
                drac_status=status,
            )

    return DRACResult(drac_status="UNAVAILABLE")
