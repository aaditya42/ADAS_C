from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class PoseSubConfig:
    """Configuration for a single pose estimation domain (human or animal).

    Attributes:
        estimator: Estimator type string (e.g. ``"null"``, ``"yolov8_pose"``).
        model_path: Path to the model weights file, or None.
        device: Torch/ONNX device string (``"cpu"``, ``"cuda:0"``, etc.).
        input_size: (height, width) input resolution for the model.
        confidence_threshold: Minimum confidence to accept a keypoint.
    """

    estimator: str
    model_path: Optional[str] = None
    device: str = "cpu"
    input_size: tuple[int, int] = (256, 256)
    confidence_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseSubConfig:
        """Build a PoseSubConfig from a parsed YAML dictionary.

        Missing keys are filled with defaults.
        """
        raw_size = data.get("input_size", (256, 256))
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            input_size: tuple[int, int] = (int(raw_size[0]), int(raw_size[1]))
        else:
            input_size = (256, 256)

        return cls(
            estimator=str(data.get("estimator", "null")),
            model_path=data.get("model_path"),
            device=str(data.get("device", "cpu")),
            input_size=input_size,
            confidence_threshold=float(
                data.get("confidence_threshold", 0.5)
            ),
        )


@dataclass(frozen=True, slots=True)
class AnimalPoseSubConfig(PoseSubConfig):
    """Configuration for animal pose estimation.

    Extends PoseSubConfig with a list of supported species.
    """

    supported_species: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnimalPoseSubConfig:
        base = PoseSubConfig.from_dict(data)
        species = data.get("supported_species", [])
        if not isinstance(species, (list, tuple)):
            species = []
        return cls(
            estimator=base.estimator,
            model_path=base.model_path,
            device=base.device,
            input_size=base.input_size,
            confidence_threshold=base.confidence_threshold,
            supported_species=tuple(str(s) for s in species),
        )


@dataclass(frozen=True, slots=True)
class PoseConfig:
    """Top-level configuration for the pose estimation subsystem.

    Attributes:
        enabled: Master toggle. When False, null estimators are used.
        human: Configuration for human pose estimation.
        animal: Configuration for animal pose estimation.
        max_persons: Maximum number of human poses to track per frame.
        max_animals: Maximum number of animal poses to track per frame.
    """

    enabled: bool = False
    human: PoseSubConfig = field(default_factory=lambda: PoseSubConfig(estimator="null"))
    animal: AnimalPoseSubConfig = field(
        default_factory=lambda: AnimalPoseSubConfig(estimator="null")
    )
    max_persons: int = 10
    max_animals: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseConfig:
        """Build a PoseConfig from a parsed YAML dictionary.

        Args:
            data: The ``pose`` section of the pipeline configuration.

        Returns:
            A fully-populated PoseConfig instance.
        """
        if not data:
            return cls()

        raw_human = data.get("human", {})
        raw_animal = data.get("animal", {})

        # Merge top-level max_* into sub-configs for backward compatibility
        max_persons = int(data.get("max_persons", raw_human.get("max_persons", 10)))
        max_animals = int(data.get("max_animals", raw_animal.get("max_animals", 10)))

        return cls(
            enabled=bool(data.get("enabled", False)),
            human=PoseSubConfig.from_dict(raw_human),
            animal=AnimalPoseSubConfig.from_dict(raw_animal),
            max_persons=max_persons,
            max_animals=max_animals,
        )
