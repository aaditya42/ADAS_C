"""Scenario-specific risk assessment engine and model selector.

Routes classified interactions to their appropriate scientific risk model,
applying exact published coefficients without alteration, and handling missing
measurements without weight redistribution or value fabrication.
"""

from __future__ import annotations

from typing import Any

from src.threat.models.base import BaseRiskModel
from src.threat.models.option_a import OptionARiskModel
from src.threat.models.option_b import OptionBRiskModel
from src.threat.models.rule_based import RuleBasedRiskModel
from src.threat.models.ssam import SSAMDiagnosticModel
from src.threat.normalization import IndicatorNormalizer
from src.threat.schemas import (
    ModelEvaluationStatus,
    RiskModelType,
    ScenarioAssessment,
    ScenarioType,
    ThreatFeatures,
    ThreatRiskAssessment,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScenarioRiskEngine:
    """Selects and executes risk models by scenario classification."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        threat_cfg = self.config.get("threat_assessment", {})
        risk_cfg = threat_cfg.get("risk_scoring", {})

        self.enabled: bool = bool(threat_cfg.get("enabled", True) and risk_cfg.get("enabled", True))
        self.normalizer = IndicatorNormalizer(self.config)

        # Model Registry
        self.models: dict[RiskModelType, BaseRiskModel] = {
            RiskModelType.OPTION_A_ADRS_2025: OptionARiskModel(self.config),
            RiskModelType.OPTION_B_INTERSECTION_2025: OptionBRiskModel(self.config),
            RiskModelType.FHWA_SSAM_DIAGNOSTIC: SSAMDiagnosticModel(self.config),
            RiskModelType.RULE_BASED_ASSESSMENT: RuleBasedRiskModel(self.config),
        }

        # Allow fallback to adapted model when full published model lacks indicators
        self.allow_adapted_fallback: bool = bool(risk_cfg.get("allow_adapted_fallback", True))

    def select_model(
        self, features: ThreatFeatures, assessment: ScenarioAssessment
    ) -> BaseRiskModel:
        """Select appropriate risk model based on scenario and measurement conditions."""
        scen = assessment.scenario
        c = features.object_class.lower()

        # 1. Car-following / Rear-End -> Option A ADRS 2025
        if scen == ScenarioType.REAR_END and c in ("car", "truck", "bus", "motorcycle"):
            return self.models[RiskModelType.OPTION_A_ADRS_2025]

        # 2. Left-turn vehicle-cyclist intersection conflict -> Option B (Intersection 2025)
        if scen == ScenarioType.CYCLIST_CONFLICT and any("INTERSECTION" in ev or "CROSSING" in ev for ev in assessment.evidence):
            return self.models[RiskModelType.OPTION_B_INTERSECTION_2025]

        # 3. Cut-In / Lane Change -> SSAM Diagnostic or Rule-Based
        if scen == ScenarioType.VEHICLE_CUT_IN:
            return self.models[RiskModelType.RULE_BASED_ASSESSMENT]

        # 4. Pedestrian Crossing -> Rule-Based VRU Model
        if scen == ScenarioType.PEDESTRIAN_CROSSING:
            return self.models[RiskModelType.RULE_BASED_ASSESSMENT]

        # 5. Adjacent-Lane Threat -> Rule-Based
        if scen == ScenarioType.ADJACENT_LANE_THREAT:
            return self.models[RiskModelType.RULE_BASED_ASSESSMENT]

        # Default fallback
        return self.models[RiskModelType.RULE_BASED_ASSESSMENT]

    def evaluate(
        self,
        features_map: dict[int, ThreatFeatures],
        assessments_map: dict[int, ScenarioAssessment],
    ) -> dict[int, ThreatRiskAssessment]:
        """Evaluate risk assessments for all tracked objects.

        Args:
            features_map: ThreatFeatures per track ID.
            assessments_map: ScenarioAssessment per track ID.

        Returns:
            Dict mapping track_id to ThreatRiskAssessment.
        """
        if not self.enabled:
            return {}

        results: dict[int, ThreatRiskAssessment] = {}

        for tid, feat in features_map.items():
            scen_ass = assessments_map.get(tid)
            if scen_ass is None:
                continue

            primary_model = self.select_model(feat, scen_ass)
            risk_ass = primary_model.evaluate(feat, scen_ass, self.normalizer)

            # If published model cannot be evaluated due to missing indicators (e.g. uncalibrated DRAC or PET),
            # optionally provide an adapted rule-based fallback while keeping model status explicit
            if (
                risk_ass.model_status in (
                    ModelEvaluationStatus.INSUFFICIENT_INDICATORS,
                    ModelEvaluationStatus.UNSUPPORTED_SCENARIO,
                )
                and self.allow_adapted_fallback
                and primary_model.model_name != RiskModelType.RULE_BASED_ASSESSMENT
            ):
                fallback_model = self.models[RiskModelType.RULE_BASED_ASSESSMENT]
                fallback_ass = fallback_model.evaluate(feat, scen_ass, self.normalizer)

                # Attach note explaining fallback adaptation
                note = (
                    f"Primary model {primary_model.model_name.value} reported {risk_ass.model_status.value} "
                    f"({'; '.join(risk_ass.rejection_reasons)}). Evaluated adapted rule-based score."
                )
                fallback_ass.rejection_reasons = list(risk_ass.rejection_reasons)
                fallback_ass.evidence.append("ADAPTED_FALLBACK_EVALUATION")
                results[tid] = fallback_ass
            else:
                results[tid] = risk_ass

        return results

