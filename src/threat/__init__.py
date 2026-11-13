"""Threat assessment, risk scoring, and threat prioritization subsystem.

Provides structured per-object threat indicators with physical validity tagging,
scenario-specific classification, scientific risk modeling (Option A ADRS, Option B Composite,
SSAM diagnostics, adapted rule-based models), and dynamic threat prioritization.
"""

from __future__ import annotations

from src.threat.direction import DirectionEstimator
from src.threat.feature_extractor import ThreatFeatureExtractor
from src.threat.models import (
    BaseRiskModel,
    OptionARiskModel,
    OptionBRiskModel,
    RuleBasedRiskModel,
    SSAMDiagnosticModel,
)
from src.threat.normalization import IndicatorNormalizer
from src.threat.prioritizer import ThreatPrioritizer
from src.threat.risk_engine import ScenarioRiskEngine
from src.threat.scenario_classifier import ScenarioClassifier
from src.threat.schemas import (
    CameraDirection,
    DirectionalWarningLevel,
    IndicatorValidity,
    LaneRelationship,
    ModelEvaluationStatus,
    NormalizedIndicator,
    ReliabilityGrade,
    RiskModelType,
    ScenarioAssessment,
    ScenarioState,
    ScenarioType,
    ThreatFeatures,
    ThreatRiskAssessment,
    ThreatRiskLevel,
    TrajectoryConflictState,
    ValidityStatus,
    VulnerabilityLevel,
    WarningEvent,
    WarningEventState,
    WarningOutput,
)
from src.threat.excel_exporter import ThreatExcelExporter
from src.threat.warning_generator import DirectionalWarningGenerator

__all__ = [
    "ThreatFeatureExtractor",
    "ScenarioClassifier",
    "ScenarioRiskEngine",
    "ThreatPrioritizer",
    "IndicatorNormalizer",
    "DirectionEstimator",
    "DirectionalWarningGenerator",
    "ThreatExcelExporter",
    "BaseRiskModel",
    "OptionARiskModel",
    "OptionBRiskModel",
    "SSAMDiagnosticModel",
    "RuleBasedRiskModel",
    "ThreatFeatures",
    "ScenarioAssessment",
    "ThreatRiskAssessment",
    "NormalizedIndicator",
    "IndicatorValidity",
    "ValidityStatus",
    "CameraDirection",
    "LaneRelationship",
    "TrajectoryConflictState",
    "VulnerabilityLevel",
    "ScenarioType",
    "ScenarioState",
    "ModelEvaluationStatus",
    "RiskModelType",
    "ReliabilityGrade",
    "ThreatRiskLevel",
    "DirectionalWarningLevel",
    "WarningEventState",
    "WarningEvent",
    "WarningOutput",
]
