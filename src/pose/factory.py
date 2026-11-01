from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.pose.base import BaseAnimalPoseEstimator, BaseHumanPoseEstimator
from src.pose.config import PoseConfig
from src.pose.null_estimator import NullAnimalPoseEstimator, NullHumanPoseEstimator


_HUMAN_ESTIMATORS = frozenset({"yolov8_pose", "rtmpose", "null"})
_ANIMAL_ESTIMATORS = frozenset({"ap10k_hrnet", "null"})


@dataclass(frozen=True, slots=True)
class PoseEstimatorBundle:
    """Container for the human and animal pose estimators built by the factory.

    Attributes:
        human: The human pose estimator instance.
        animal: The animal pose estimator instance.
    """

    human: BaseHumanPoseEstimator
    animal: BaseAnimalPoseEstimator


def _build_human_estimator(cfg: PoseConfig) -> BaseHumanPoseEstimator:
    """Build a human pose estimator from the pose configuration.

    Args:
        cfg: Parsed pose configuration.

    Returns:
        A BaseHumanPoseEstimator instance.

    Raises:
        ValueError: If the estimator type is unknown.
        NotImplementedError: If the estimator type is known but not yet implemented.
    """
    estimator_type = cfg.human.estimator

    if estimator_type not in _HUMAN_ESTIMATORS:
        raise ValueError(
            f"Unknown human pose estimator: {estimator_type!r}. "
            f"Expected one of: {sorted(_HUMAN_ESTIMATORS)}."
        )

    if estimator_type == "null":
        return NullHumanPoseEstimator()

    if estimator_type == "yolov8_pose":
        from src.pose.human.yolov8_pose_estimator import (  # noqa: PLC0415
            YOLOv8HumanPoseEstimator,
        )

        return YOLOv8HumanPoseEstimator(
            model_path=str(cfg.human.model_path or "yolov8n-pose.pt"),
            device=cfg.human.device,
            conf_threshold=cfg.human.confidence_threshold,
            iou_threshold=0.65,
            input_size=max(cfg.human.input_size),
            max_persons=cfg.max_persons,
        )

    raise NotImplementedError(
        f"Human pose estimator {estimator_type!r} is not implemented yet. "
        f"Use 'null' for a no-op placeholder."
    )


def _build_animal_estimator(
    cfg: PoseConfig,
    raw_animal: dict[str, Any] | None = None,
) -> BaseAnimalPoseEstimator:
    """Build an animal pose estimator from the pose configuration.

    Args:
        cfg: Parsed pose configuration.
        raw_animal: Raw animal sub-section of the pose configuration.
            Used to pass species mapping overrides to the estimator.

    Returns:
        A BaseAnimalPoseEstimator instance.

    Raises:
        ValueError: If the estimator type is unknown.
        NotImplementedError: If the estimator type is known but not yet implemented.
    """
    estimator_type = cfg.animal.estimator

    if estimator_type not in _ANIMAL_ESTIMATORS:
        raise ValueError(
            f"Unknown animal pose estimator: {estimator_type!r}. "
            f"Expected one of: {sorted(_ANIMAL_ESTIMATORS)}."
        )

    if estimator_type == "null":
        return NullAnimalPoseEstimator()

    if estimator_type == "ap10k_hrnet":
        from src.pose.animal.ap10k_hrnet_estimator import (  # noqa: PLC0415
            AP10KHRNetAnimalPoseEstimator,
        )
        from src.pose.animal.species_mapper import (  # noqa: PLC0415
            SpeciesMapper,
        )

        species_mapper = SpeciesMapper.from_config(raw_animal or {})
        return AP10KHRNetAnimalPoseEstimator(
            model_path=str(cfg.animal.model_path or "ap10k_hrnet.onnx"),
            device=cfg.animal.device,
            conf_threshold=cfg.animal.confidence_threshold,
            input_size=cfg.animal.input_size,
            max_animals=cfg.max_animals,
            species_mapper=species_mapper,
        )

    raise NotImplementedError(
        f"Animal pose estimator {estimator_type!r} is not implemented yet. "
        f"Use 'null' for a no-op placeholder."
    )


def build_pose_estimators(config: dict[str, Any]) -> PoseEstimatorBundle:
    """Build human and animal pose estimators from the pipeline configuration.

    The configuration must contain a ``pose`` section. If the section is
    missing or ``pose.enabled`` is ``False``, null (no-op) estimators are
    returned, which guarantees backward compatibility — no additional
    computation or model loading occurs.

    Args:
        config: Full pipeline configuration dictionary.

    Returns:
        A PoseEstimatorBundle containing the human and animal estimators.
    """
    raw_pose = config.get("pose", {})

    if not raw_pose.get("enabled", False):
        return PoseEstimatorBundle(
            human=NullHumanPoseEstimator(),
            animal=NullAnimalPoseEstimator(),
        )

    cfg = PoseConfig.from_dict(raw_pose)

    return PoseEstimatorBundle(
        human=_build_human_estimator(cfg),
        animal=_build_animal_estimator(cfg, raw_animal=raw_pose.get("animal", {})),
    )
