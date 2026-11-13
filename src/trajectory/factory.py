from __future__ import annotations

from typing import Any

from src.trajectory.trajectory_predictor import TrajectoryPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_trajectory_predictor(
    config: dict[str, Any]
) -> TrajectoryPredictor | None:
    """Build a TrajectoryPredictor instance from configuration dictionary.

    Returns:
        TrajectoryPredictor if enabled, otherwise None.
    """
    traj_cfg = config.get("trajectory", {})
    if not traj_cfg.get("enabled", True):
        logger.info("Trajectory prediction is disabled in config.")
        return None

    predictor = TrajectoryPredictor(config)
    logger.info(
        "TrajectoryPredictor initialized | horizon=%.2fs interval=%.2fs model=%s",
        predictor.horizon_seconds,
        predictor.update_interval,
        predictor.model_name,
    )
    return predictor
