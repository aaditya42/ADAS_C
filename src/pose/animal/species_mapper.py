from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple


class SpeciesMapper:
    """Config-driven mapper from COCO class IDs to animal species strings.

    Provides a default mapping for common COCO animal classes (indices 16–23)
    which covers dog, horse, sheep, cow, elephant, bear, zebra, and giraffe.
    Users can override or extend this mapping via configuration.

    The ``supported_species`` tuple acts as an allow-list: species not in this
    list are treated as unsupported and their detections will be skipped during
    pose estimation.

    Args:
        class_to_species: Mapping from COCO class ID (int) to species name (str).
        supported_species: Tuple of species strings that are allowed.
            If empty, ALL species in the mapping are considered supported.
    """

    DEFAULT_CLASS_TO_SPECIES: ClassVar[Dict[int, str]] = {
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
    }

    def __init__(
        self,
        class_to_species: Optional[Dict[int, str]] = None,
        supported_species: Sequence[str] = (),
    ) -> None:
        self._class_to_species: Dict[int, str] = dict(
            self.DEFAULT_CLASS_TO_SPECIES
        )
        if class_to_species is not None:
            self._class_to_species.update(class_to_species)
        self._supported_species: Tuple[str, ...] = tuple(supported_species)

    @property
    def supported_species(self) -> Tuple[str, ...]:
        """Return the configured list of supported species."""
        return self._supported_species

    def map(self, class_id: int) -> Optional[str]:
        """Map a COCO class ID to a species string.

        Args:
            class_id: COCO class index (e.g. 16 for dog, 19 for cow).

        Returns:
            Species string if the class ID is recognised, otherwise ``None``.
        """
        return self._class_to_species.get(class_id)

    def is_supported(self, species: str) -> bool:
        """Check whether a species is in the supported allow-list.

        When ``supported_species`` is empty, all species are considered
        supported.
        """
        if not self._supported_species:
            return True
        return species in self._supported_species

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> SpeciesMapper:
        """Build a SpeciesMapper from a parsed YAML configuration section.

        Expected keys:
            ``class_to_species``: optional dict of ``{int: str}`` overrides.
            ``supported_species``: optional list of species strings.

        Args:
            data: The animal sub-configuration dictionary.

        Returns:
            A configured ``SpeciesMapper`` instance.
        """
        raw_mapping: dict[str, Any] = data.get("class_to_species", {})
        class_to_species: Dict[int, str] = {}
        for k, v in raw_mapping.items():
            class_to_species[int(k)] = str(v)

        raw_supported: list[str] = data.get("supported_species", [])
        supported_species: Tuple[str, ...] = tuple(
            str(s) for s in raw_supported if isinstance(s, str)
        )

        return cls(
            class_to_species=class_to_species or None,
            supported_species=supported_species,
        )
