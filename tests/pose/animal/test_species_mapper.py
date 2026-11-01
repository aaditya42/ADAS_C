from __future__ import annotations

import pytest

from src.pose.animal.species_mapper import SpeciesMapper


class TestSpeciesMapperDefaults:
    """Tests for the default COCO class → species mapping."""

    def test_known_dog(self) -> None:
        mapper = SpeciesMapper()
        assert mapper.map(16) == "dog"

    def test_known_cow(self) -> None:
        mapper = SpeciesMapper()
        assert mapper.map(19) == "cow"

    def test_known_bear(self) -> None:
        mapper = SpeciesMapper()
        assert mapper.map(21) == "bear"

    def test_unknown_class_id(self) -> None:
        mapper = SpeciesMapper()
        assert mapper.map(999) is None

    def test_negative_class_id(self) -> None:
        mapper = SpeciesMapper()
        assert mapper.map(-1) is None

    def test_non_animal_class_id(self) -> None:
        mapper = SpeciesMapper()
        assert mapper.map(0) is None  # person
        assert mapper.map(1) is None  # bicycle
        assert mapper.map(2) is None  # car

    def test_all_default_mappings_present(self) -> None:
        mapper = SpeciesMapper()
        for class_id in range(16, 24):
            species = mapper.map(class_id)
            assert species is not None, f"Class {class_id} should map to a species"
            assert isinstance(species, str)
            assert len(species) > 0


class TestSpeciesMapperCustomMapping:
    """Tests for custom mapping overrides."""

    def test_override_default(self) -> None:
        mapper = SpeciesMapper(class_to_species={16: "canine"})
        assert mapper.map(16) == "canine"

    def test_add_new_species(self) -> None:
        mapper = SpeciesMapper(class_to_species={42: "fox"})
        assert mapper.map(42) == "fox"
        assert mapper.map(16) == "dog"  # default still works

    def test_override_clears_default_when_empty_dict(self) -> None:
        mapper = SpeciesMapper(class_to_species={})
        assert mapper.map(16) == "dog"  # defaults still present

    def test_multiple_overrides(self) -> None:
        mapper = SpeciesMapper(class_to_species={16: "canine", 19: "bovine"})
        assert mapper.map(16) == "canine"
        assert mapper.map(19) == "bovine"
        assert mapper.map(20) == "elephant"  # default still present


class TestSpeciesMapperSupportedSpecies:
    """Tests for the supported-species allow-list."""

    def test_empty_supported_all_allowed(self) -> None:
        mapper = SpeciesMapper(supported_species=[])
        assert mapper.is_supported("dog") is True
        assert mapper.is_supported("cow") is True
        assert mapper.is_supported("whatever") is True

    def test_known_species_is_supported(self) -> None:
        mapper = SpeciesMapper(supported_species=["dog", "cow"])
        assert mapper.is_supported("dog") is True
        assert mapper.is_supported("cow") is True

    def test_unknown_species_not_supported(self) -> None:
        mapper = SpeciesMapper(supported_species=["dog", "cow"])
        assert mapper.is_supported("elephant") is False
        assert mapper.is_supported("zebra") is False

    def test_species_not_in_default_mapping(self) -> None:
        mapper = SpeciesMapper(
            class_to_species={42: "fox"}, supported_species=["fox"]
        )
        assert mapper.is_supported("fox") is True
        assert mapper.is_supported("dog") is False


class TestSpeciesMapperFromConfig:
    """Tests for the ``from_config`` factory method."""

    def test_empty_config(self) -> None:
        mapper = SpeciesMapper.from_config({})
        assert mapper.map(16) == "dog"
        assert mapper.is_supported("dog") is True

    def test_with_supported_species(self) -> None:
        mapper = SpeciesMapper.from_config(
            {"supported_species": ["dog", "cow"]}
        )
        assert mapper.is_supported("dog") is True
        assert mapper.is_supported("horse") is False

    def test_with_class_to_species_override(self) -> None:
        mapper = SpeciesMapper.from_config(
            {
                "class_to_species": {"16": "canine", "42": "fox"},
                "supported_species": ["canine", "fox"],
            }
        )
        assert mapper.map(16) == "canine"
        assert mapper.map(42) == "fox"
        assert mapper.is_supported("canine") is True
        assert mapper.is_supported("dog") is False

    def test_class_to_species_empty(self) -> None:
        mapper = SpeciesMapper.from_config(
            {"class_to_species": {}, "supported_species": []}
        )
        # Defaults still apply
        assert mapper.map(16) == "dog"
        assert mapper.map(19) == "cow"
