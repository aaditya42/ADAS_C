"""Indicator normalization component.

Implements transparent, documented normalizations mapping heterogeneous physical
measurements to bounded [0.0, 1.0] risk indicators, respecting risk direction
and empirical reference ranges from published methodologies.
"""

from __future__ import annotations

import math
from typing import Any

from src.threat.schemas import NormalizedIndicator, ValidityStatus


class IndicatorNormalizer:
    """Normalizes surrogate safety and kinematic indicators to [0.0, 1.0].

    Reference ranges:
    - TTC: [1.0, 6.0] s (Inverse: lower is higher risk)
    - PET: [1.0, 5.0] s (Inverse: lower is higher risk)
    - DRAC: [0.0, 5.0] m/s^2 (Direct: higher is higher risk)
    - Speed standard deviation (Spd): [0.0, 3.0] m/s (Direct: higher is higher risk)
    - Acceleration noise (AN): [0.0, 2.0] m/s^2 (Direct: higher is higher risk)
    - Lateral deviation (LD): [0.0, 1.0] normalized ratio (Direct: higher is higher risk)
    - Combined speed (V_comb): [0.0, 15.0] m/s (Direct: higher is higher risk)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        norm_cfg = self.config.get("threat_assessment", {}).get("normalization", {})

        self.ttc_crit_s: float = float(norm_cfg.get("ttc_crit_s", 1.0))
        self.ttc_safe_s: float = float(norm_cfg.get("ttc_safe_s", 6.0))

        self.pet_crit_s: float = float(norm_cfg.get("pet_crit_s", 1.0))
        self.pet_safe_s: float = float(norm_cfg.get("pet_safe_s", 5.0))

        self.drac_max_mps2: float = float(norm_cfg.get("drac_max_mps2", 5.0))
        self.spd_var_max_mps: float = float(norm_cfg.get("spd_var_max_mps", 3.0))
        self.an_max_mps2: float = float(norm_cfg.get("an_max_mps2", 2.0))
        self.v_comb_max_mps: float = float(norm_cfg.get("v_comb_max_mps", 15.0))

    def normalize_ttc(
        self,
        raw_ttc: float | None,
        validity: ValidityStatus = ValidityStatus.VALID_PROXY,
        source_model: str = "general",
    ) -> NormalizedIndicator:
        """Normalize Time to Collision (TTC). Lower TTC -> higher normalized risk."""
        ref_range = (self.ttc_crit_s, self.ttc_safe_s)
        if raw_ttc is None or not math.isfinite(raw_ttc) or raw_ttc <= 0.0 or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_ttc,
                normalized_value=None,
                method="piecewise_linear_inverse",
                reference_range=ref_range,
                risk_direction="INVERSE",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="Non-closing or unavailable TTC.",
            )

        val = float(raw_ttc)
        if val <= self.ttc_crit_s:
            norm_val = 1.0
        elif val >= self.ttc_safe_s:
            norm_val = 0.0
        else:
            norm_val = (self.ttc_safe_s - val) / (self.ttc_safe_s - self.ttc_crit_s)

        return NormalizedIndicator(
            raw_value=val,
            normalized_value=max(0.0, min(1.0, norm_val)),
            method="piecewise_linear_inverse",
            reference_range=ref_range,
            risk_direction="INVERSE",
            validity=validity,
            source_model=source_model,
        )

    def normalize_spd(
        self,
        raw_sigma_v: float | None,
        validity: ValidityStatus = ValidityStatus.VALID_PROXY,
        source_model: str = "Option A (ADRS 2025)",
    ) -> NormalizedIndicator:
        """Normalize speed variation (standard deviation of speed). Higher -> higher risk."""
        ref_range = (0.0, self.spd_var_max_mps)
        if raw_sigma_v is None or not math.isfinite(raw_sigma_v) or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_sigma_v,
                normalized_value=None,
                method="linear_saturation",
                reference_range=ref_range,
                risk_direction="DIRECT",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="Speed variation unavailable.",
            )

        val = max(0.0, float(raw_sigma_v))
        norm_val = min(1.0, val / self.spd_var_max_mps)
        return NormalizedIndicator(
            raw_value=val,
            normalized_value=norm_val,
            method="linear_saturation",
            reference_range=ref_range,
            risk_direction="DIRECT",
            validity=validity,
            source_model=source_model,
        )

    def normalize_an(
        self,
        raw_an: float | None,
        validity: ValidityStatus = ValidityStatus.VALID_PROXY,
        source_model: str = "Option A (ADRS 2025)",
    ) -> NormalizedIndicator:
        """Normalize acceleration noise. Higher -> higher risk."""
        ref_range = (0.0, self.an_max_mps2)
        if raw_an is None or not math.isfinite(raw_an) or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_an,
                normalized_value=None,
                method="linear_saturation",
                reference_range=ref_range,
                risk_direction="DIRECT",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="Acceleration noise unavailable.",
            )

        val = max(0.0, float(raw_an))
        norm_val = min(1.0, val / self.an_max_mps2)
        return NormalizedIndicator(
            raw_value=val,
            normalized_value=norm_val,
            method="linear_saturation",
            reference_range=ref_range,
            risk_direction="DIRECT",
            validity=validity,
            source_model=source_model,
        )

    def normalize_drac(
        self,
        raw_drac: float | None,
        validity: ValidityStatus = ValidityStatus.VALID_METRIC,
        source_model: str = "general",
    ) -> NormalizedIndicator:
        """Normalize Deceleration Rate to Avoid Collision (DRAC). Higher -> higher risk."""
        ref_range = (0.0, self.drac_max_mps2)
        if raw_drac is None or not math.isfinite(raw_drac) or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_drac,
                normalized_value=None,
                method="linear_saturation",
                reference_range=ref_range,
                risk_direction="DIRECT",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="DRAC unavailable (requires calibrated metric distance and closing speed).",
            )

        val = max(0.0, float(raw_drac))
        norm_val = min(1.0, val / self.drac_max_mps2)
        return NormalizedIndicator(
            raw_value=val,
            normalized_value=norm_val,
            method="linear_saturation",
            reference_range=ref_range,
            risk_direction="DIRECT",
            validity=validity,
            source_model=source_model,
        )

    def normalize_ld(
        self,
        raw_ld: float | None,
        validity: ValidityStatus = ValidityStatus.VALID_METRIC,
        source_model: str = "Option A (ADRS 2025)",
    ) -> NormalizedIndicator:
        """Normalize lateral deviation from lane center. Higher -> higher risk."""
        ref_range = (0.0, 1.0)
        if raw_ld is None or not math.isfinite(raw_ld) or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_ld,
                normalized_value=None,
                method="linear_saturation",
                reference_range=ref_range,
                risk_direction="DIRECT",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="Lateral deviation unavailable (missing lane boundaries).",
            )

        val = max(0.0, float(raw_ld))
        norm_val = min(1.0, val)
        return NormalizedIndicator(
            raw_value=val,
            normalized_value=norm_val,
            method="linear_saturation",
            reference_range=ref_range,
            risk_direction="DIRECT",
            validity=validity,
            source_model=source_model,
        )

    def normalize_pet(
        self,
        raw_pet: float | None,
        validity: ValidityStatus = ValidityStatus.UNAVAILABLE,
        source_model: str = "Option B (Intersection 2025)",
    ) -> NormalizedIndicator:
        """Normalize Post-Encroachment Time (PET). Lower PET -> higher risk."""
        ref_range = (self.pet_crit_s, self.pet_safe_s)
        if raw_pet is None or not math.isfinite(raw_pet) or raw_pet <= 0.0 or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_pet,
                normalized_value=None,
                method="piecewise_linear_inverse",
                reference_range=ref_range,
                risk_direction="INVERSE",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="PET unavailable in forward monocular camera (no ground-plane conflict polygon).",
            )

        val = float(raw_pet)
        if val <= self.pet_crit_s:
            norm_val = 1.0
        elif val >= self.pet_safe_s:
            norm_val = 0.0
        else:
            norm_val = (self.pet_safe_s - val) / (self.pet_safe_s - self.pet_crit_s)

        return NormalizedIndicator(
            raw_value=val,
            normalized_value=max(0.0, min(1.0, norm_val)),
            method="piecewise_linear_inverse",
            reference_range=ref_range,
            risk_direction="INVERSE",
            validity=validity,
            source_model=source_model,
        )

    def normalize_v_comb(
        self,
        raw_v_comb: float | None,
        validity: ValidityStatus = ValidityStatus.VALID_PROXY,
        source_model: str = "Option B (Intersection 2025)",
    ) -> NormalizedIndicator:
        """Normalize combined conflict velocity. Higher -> higher risk."""
        ref_range = (0.0, self.v_comb_max_mps)
        if raw_v_comb is None or not math.isfinite(raw_v_comb) or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=raw_v_comb,
                normalized_value=None,
                method="linear_saturation",
                reference_range=ref_range,
                risk_direction="DIRECT",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note="Combined velocity unavailable.",
            )

        val = max(0.0, float(raw_v_comb))
        norm_val = min(1.0, val / self.v_comb_max_mps)
        return NormalizedIndicator(
            raw_value=val,
            normalized_value=norm_val,
            method="linear_saturation",
            reference_range=ref_range,
            risk_direction="DIRECT",
            validity=validity,
            source_model=source_model,
        )

    def normalize_binary(
        self,
        flag: bool | int | float | None,
        indicator_name: str,
        validity: ValidityStatus = ValidityStatus.VALID_PROXY,
        source_model: str = "general",
    ) -> NormalizedIndicator:
        """Normalize binary indicators (e.g. ROW or Urgent Braking)."""
        ref_range = (0.0, 1.0)
        if flag is None or validity == ValidityStatus.UNAVAILABLE:
            return NormalizedIndicator(
                raw_value=None,
                normalized_value=None,
                method="binary_flag",
                reference_range=ref_range,
                risk_direction="DIRECT",
                validity=ValidityStatus.UNAVAILABLE,
                source_model=source_model,
                adaptation_note=f"{indicator_name} flag unavailable.",
            )

        norm_val = 1.0 if bool(flag) else 0.0
        return NormalizedIndicator(
            raw_value=norm_val,
            normalized_value=norm_val,
            method="binary_flag",
            reference_range=ref_range,
            risk_direction="DIRECT",
            validity=validity,
            source_model=source_model,
        )

