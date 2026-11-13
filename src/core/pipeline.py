from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.schemas import Detection, Frame, PerceivedObject, TrackedObject

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """Orchestrator for the ADAS perception system.

    Modules are registered into the pipeline in later phases. Each module
    consumes and produces well-defined data structures without calling other
    modules directly.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._modules: dict[str, Any] = {}
        self._frame_source: Any = None
        self._detector: Any = None
        self._tracker: Any = None
        self._depth_estimator: Any = None
        self._depth_cache: np.ndarray | None = None
        self._last_depth_frame_id: int = -1
        self._fusion: Any = None
        self._history_mgr: Any = None
        self._distance_estimator: Any = None
        self._motion_estimator: Any = None
        self._ttc_estimator: Any = None
        self._risk_engine: Any = None
        self._warning_manager: Any = None
        self._lane_detector: Any = None
        self._lane_tracker: Any = None
        self._signs_cache: list[Any] = []
        self._last_signs_frame_id: int = -1
        self._scene_builder: Any = None
        self._renderer: Any = None
        self._pose_subsystem: Any = None
        self._behavior_subsystem: Any = None
        self._trajectory_predictor: Any = None
        self._overtaking_detector: Any = None
        self._threat_feature_extractor: Any = None
        self._scenario_classifier: Any = None
        self._threat_risk_engine: Any = None
        self._threat_prioritizer: Any = None
        self._directional_warning_generator: Any = None
        self._last_threat_features: dict[int, Any] = {}
        self._last_threat_scenarios: dict[int, Any] = {}
        self._last_threat_assessments: dict[int, Any] = {}
        self._last_prioritized_threats: list[Any] = []
        self._last_directional_warnings: list[Any] = []
        self._last_directional_warning_events: list[Any] = []
        self._processed_frames: int = 0
        self._last_processed_timestamp: float = 0.0

        # Eager init for sign detector (controlled by config)
        signs_cfg = config.get("signs", {})
        if signs_cfg.get("enabled", False):
            from src.signs.factory import build_sign_detector  # noqa: PLC0415

            self._sign_detector = build_sign_detector(config)
        else:
            self._sign_detector = None

        # Eager init for scene builder (controlled by config)
        scene_cfg = config.get("scene", {})
        if scene_cfg.get("enabled", True):
            from src.scene.factory import build_scene_builder  # noqa: PLC0415

            self._scene_builder = build_scene_builder(config)

        # Eager init for pose subsystem (controlled by config)
        pose_cfg = config.get("pose", {})
        if pose_cfg.get("enabled", False):
            from src.pose.subsystem import PoseSubsystem  # noqa: PLC0415

            self._pose_subsystem = PoseSubsystem(config)
        else:
            self._pose_subsystem = None

        # Eager init for behaviour subsystem (controlled by config)
        behavior_cfg = config.get("behavior", {})
        if behavior_cfg.get("enabled", False):
            from src.behavior.subsystem import BehaviorSubsystem  # noqa: PLC0415

            self._behavior_subsystem = BehaviorSubsystem(config)
        else:
            self._behavior_subsystem = None

    @property
    def renderer(self) -> Any:
        """Lazy-initialised renderer built from the configuration."""
        if self._renderer is None:
            vis_cfg = self.config.get("visualization", {})
            if vis_cfg.get("enabled", True):
                from src.visualization.factory import build_renderer  # noqa: PLC0415

                self._renderer = build_renderer(self.config)
            else:
                self._renderer = False  # sentinel
        return self._renderer if self._renderer is not False else None

    @property
    def frame_source(self) -> Any:
        """Lazy-initialised frame source built from the configuration."""
        if self._frame_source is None:
            from src.video.factory import build_frame_source  # noqa: PLC0415

            self._frame_source = build_frame_source(self.config)
        return self._frame_source

    @property
    def detector(self) -> Any:
        """Lazy-initialised detector built from the configuration."""
        if self._detector is None:
            from src.detection.factory import build_detector  # noqa: PLC0415

            self._detector = build_detector(self.config)
        return self._detector

    @property
    def tracker(self) -> Any:
        """Lazy-initialised tracker built from the configuration."""
        if self._tracker is None:
            from src.tracking.factory import build_tracker  # noqa: PLC0415

            self._tracker = build_tracker(self.config)
        return self._tracker

    @property
    def depth_estimator(self) -> Any:
        """Lazy-initialised depth estimator built from the configuration."""
        if self._depth_estimator is None:
            from src.depth.factory import build_depth_estimator  # noqa: PLC0415

            self._depth_estimator = build_depth_estimator(self.config)
        return self._depth_estimator

    @property
    def fusion(self) -> Any:
        """Lazy-initialised object fusion module."""
        if self._fusion is None:
            from src.fusion.object_fusion import ObjectFusion  # noqa: PLC0415

            fusion_cfg = self.config.get("fusion", {})
            self._fusion = ObjectFusion(
                sampling=str(fusion_cfg.get("sampling", "median_center")),
                center_box_ratio=float(
                    fusion_cfg.get("center_box_ratio", 0.5)
                ),
                min_valid_depth_pixels=int(
                    fusion_cfg.get("min_valid_depth_pixels", 10)
                ),
            )
        return self._fusion

    @property
    def history(self) -> Any:
        """Lazy-initialised track history manager."""
        if self._history_mgr is None:
            from src.memory.track_history_manager import (  # noqa: PLC0415
                TrackHistoryManager,
            )

            mem_cfg = self.config.get("memory", {})
            self._history_mgr = TrackHistoryManager(
                max_history_per_track=int(
                    mem_cfg.get("max_history_per_track", 60)
                ),
                stale_after_frames=int(
                    mem_cfg.get("stale_after_frames", 60)
                ),
            )
        return self._history_mgr

    @property
    def distance_estimator(self) -> Any:
        """Lazy-initialised distance estimator built from the configuration."""
        if self._distance_estimator is None:
            from src.distance.factory import (  # noqa: PLC0415
                build_distance_estimator,
            )

            self._distance_estimator = build_distance_estimator(self.config)
        return self._distance_estimator

    def register(self, name: str, module: Any) -> None:
        self._modules[name] = module

    def _enrich_distances(
        self,
        perceived: list[PerceivedObject],
        depth_map: np.ndarray | None,
        frame: Frame,
    ) -> list[PerceivedObject]:
        """Replace the placeholder distance with the calibrated metric value."""
        enriched: list[PerceivedObject] = []
        for p in perceived:
            metric = self.distance_estimator.estimate(p, depth_map, frame)
            if metric is not None:
                enriched.append(replace(p, distance=metric))
            else:
                enriched.append(p)
        return enriched

    def process(self, frame: Frame) -> dict[str, Any]:
        """Run the full perception pipeline on a single frame.

        Args:
            frame: The current video frame.

        Returns:
            A dictionary keyed by module name. Contains ``"frame"``,
            ``"detections"``, ``"tracked"``, ``"depth_map"``,
            ``"perceived"`` and ``"history"``.
        """
        self._processed_frames += 1
        if frame is not None and getattr(frame, "timestamp", None) is not None:
            self._last_processed_timestamp = float(frame.timestamp)

        detections: list[Detection] = self.detector.detect(frame)
        tracked: list[TrackedObject] = self.tracker.update(detections, frame)
        pose_frame = self._estimate_pose(frame, detections, tracked)
        depth_map: np.ndarray = self._estimate_depth(frame)
        perceived: list[PerceivedObject] = self.fusion.fuse(tracked, depth_map)
        perceived = self._enrich_distances(perceived, depth_map, frame)
        self.history.update(perceived, frame)
        motion = self._compute_motion()
        ttc = self._compute_ttc(perceived, motion)
        lane_info = self._detect_lanes(frame)
        signs = self._detect_signs(frame)
        scene_context = self._build_scene(perceived, lane_info, signs, frame)
        behavior_frame = self._run_behavior(pose_frame, motion, lane_info, frame)
        risk_objects = self._evaluate_risk(
            perceived, motion, ttc, lane_info, scene_context,
            behavior_frame=behavior_frame,
        )
        alerts = self._process_warnings(risk_objects, frame)
        trajectories = self._predict_trajectories(
            perceived, frame, motion=motion, lane_info=lane_info, scene_context=scene_context
        )
        evaluations = (
            getattr(self.trajectory_predictor, "last_active_evaluations", [])
            if self.trajectory_predictor is not None
            else []
        )
        overtaking_result = self._process_overtaking(
            perceived, lane_info, scene_context, frame
        )
        threat_features = self._extract_threat_features(
            perceived=perceived,
            motion=motion,
            ttc=ttc,
            lane_info=lane_info,
            scene_context=scene_context,
            behavior_frame=behavior_frame,
            trajectories=trajectories,
            frame=frame,
        )
        threat_scenarios = self._classify_threat_scenarios(
            threat_features=threat_features,
            frame=frame,
        )
        threat_assessments = self._evaluate_threat_risks(
            threat_features=threat_features,
            threat_scenarios=threat_scenarios,
        )
        prioritized_threats = self._prioritize_threats(
            threat_assessments=threat_assessments,
        )
        directional_warnings, warning_events = self._generate_directional_warnings(
            threat_assessments=threat_assessments,
            threat_features=threat_features,
            lane_info=lane_info,
            frame=frame,
        )
        self._last_threat_features = threat_features
        self._last_threat_scenarios = threat_scenarios
        self._last_threat_assessments = threat_assessments
        self._last_prioritized_threats = prioritized_threats
        self._last_directional_warnings = directional_warnings
        self._last_directional_warning_events = warning_events

        result: dict[str, Any] = {
            "frame": frame,
            "detections": detections,
            "tracked": tracked,
            "pose": pose_frame,
            "depth_map": depth_map,
            "perceived": perceived,
            "history": self.history,
            "motion": motion,
            "trajectories": trajectories,
            "trajectory": trajectories,
            "trajectory_evaluations": evaluations,
            "overtaking": overtaking_result,
            "threat_features": threat_features,
            "threat_scenarios": threat_scenarios,
            "threat_assessments": threat_assessments,
            "prioritized_threats": prioritized_threats,
            "warnings": directional_warnings,
            "directional_warnings": directional_warnings,
            "warning_events": warning_events,
            "ttc": ttc,
            "risk": risk_objects,
            "alerts": alerts,
            "lane": lane_info,
            "signs": signs,
            "scene": scene_context,
            "behavior": behavior_frame,
        }
        return result

    @property
    def trajectory_predictor(self) -> Any:
        """Lazy-initialised trajectory predictor (or None if disabled)."""
        if self._trajectory_predictor is None:
            traj_cfg = self.config.get("trajectory", {})
            if traj_cfg.get("enabled", True):
                from src.trajectory.factory import build_trajectory_predictor  # noqa: PLC0415

                self._trajectory_predictor = build_trajectory_predictor(self.config)
            else:
                self._trajectory_predictor = False
        return self._trajectory_predictor if self._trajectory_predictor is not False else None

    def _predict_trajectories(
        self,
        objects: list[PerceivedObject | TrackedObject],
        frame: Frame,
        motion: dict[int, Any] | None = None,
        lane_info: Any = None,
        scene_context: Any = None,
    ) -> list[Any]:
        """Generate future trajectory predictions for tracked objects."""
        predictor = self.trajectory_predictor
        if predictor is None:
            return []
        try:
            return predictor.predict(
                objects,
                frame,
                motion=motion,
                lane_info=lane_info,
                scene_context=scene_context,
            )
        except Exception:
            return []

    def export_trajectory_evaluation(
        self, output_path: str | None = None
    ) -> Any:
        """Export trajectory evaluation workbook to Excel."""
        if self.trajectory_predictor is not None:
            return self.trajectory_predictor.export_evaluation(output_path)
        return None

    @property
    def overtaking_detector(self) -> Any:
        """Lazy-initialised overtaking detector (or None if disabled)."""
        if self._overtaking_detector is None:
            ot_cfg = self.config.get("overtaking", {})
            if ot_cfg.get("enabled", True):
                from src.overtaking.detector import OvertakingDetector  # noqa: PLC0415

                self._overtaking_detector = OvertakingDetector(self.config)
            else:
                self._overtaking_detector = False
        return self._overtaking_detector if self._overtaking_detector is not False else None

    def _process_overtaking(
        self,
        objects: list[PerceivedObject | TrackedObject],
        lane_info: Any,
        scene_context: Any,
        frame: Frame,
    ) -> Any:
        """Process overtaking events for tracked objects."""
        detector = self.overtaking_detector
        if detector is None:
            return None
        try:
            return detector.process(objects, lane_info, scene_context, frame)
        except Exception:
            return None

    def finish_overtaking_session(self) -> None:
        """Finalize any pending active overtaking events when the video stream ends."""
        if self.overtaking_detector is not None:
            self.overtaking_detector.finish_session(final_timestamp=self._last_processed_timestamp)
            ot_cfg = self.config.get("overtaking", {})
            if ot_cfg.get("export", {}).get("export_on_session_end", True):
                try:
                    self.export_overtaking_evaluation()
                except Exception as exc:
                    logger.warning("Failed to auto-export overtaking evaluation report: %s", exc)

    def export_overtaking_evaluation(self, output_path: str | None = None) -> Any:
        """Export overtaking evaluation workbook to Excel."""
        detector = self.overtaking_detector
        if detector is None:
            return None

        events = detector.all_events
        ot_cfg = self.config.get("overtaking", {})
        exp_cfg = ot_cfg.get("export", {})

        output_dir = exp_cfg.get("output_dir", "./reports")
        filename = exp_cfg.get("filename", "Overtaking_Evaluation.xlsx")

        if output_path is not None:
            out_p = Path(output_path)
            output_dir = str(out_p.parent)
            filename = out_p.name

        from src.overtaking.excel_exporter import OvertakingExcelExporter  # noqa: PLC0415

        metadata = {
            "video_path": getattr(self.frame_source, "path", self.config.get("video", {}).get("path", "N/A")),
            "fps": getattr(self.frame_source, "fps", 30.0),
            "width": getattr(self.frame_source, "width", 1280),
            "height": getattr(self.frame_source, "height", 720),
            "frame_count": self._processed_frames,
            "duration_seconds": self._last_processed_timestamp,
        }

        exporter = OvertakingExcelExporter(output_dir=output_dir, filename=filename)
        return exporter.export(events, metadata=metadata, config=self.config)



    @property
    def motion_estimator(self) -> Any:
        """Lazy-initialised motion estimator built from the configuration."""
        if self._motion_estimator is None:
            from src.motion.factory import build_motion_estimator  # noqa: PLC0415

            self._motion_estimator = build_motion_estimator(self.config)
        return self._motion_estimator

    @property
    def ttc_estimator(self) -> Any:
        """Lazy-initialised TTC estimator built from the configuration."""
        if self._ttc_estimator is None:
            from src.ttc.factory import build_ttc  # noqa: PLC0415

            self._ttc_estimator = build_ttc(self.config)
        return self._ttc_estimator

    def _compute_motion(self) -> dict[int, Any]:
        """Estimate motion for all active tracks."""
        return self.motion_estimator.estimate(self.history)

    @property
    def risk_engine(self) -> Any:
        """Lazy-initialised risk engine built from the configuration."""
        if self._risk_engine is None:
            from src.risk.factory import build_risk_engine  # noqa: PLC0415

            self._risk_engine = build_risk_engine(self.config)
        return self._risk_engine

    def _compute_ttc(
        self,
        perceived: list[PerceivedObject],
        motion: dict[int, Any],
    ) -> dict[int, float | None]:
        """Compute time-to-collision for all perceived objects."""
        return self.ttc_estimator.estimate(perceived, motion)

    @property
    def warning_manager(self) -> Any:
        """Lazy-initialised warning manager built from the configuration."""
        if self._warning_manager is None:
            from src.warning.factory import build_warning_manager  # noqa: PLC0415

            self._warning_manager = build_warning_manager(self.config)
        return self._warning_manager

    @property
    def lane_detector(self) -> Any:
        """Lazy-initialised lane detector (or None if disabled)."""
        if self._lane_detector is None:
            lane_cfg = self.config.get("lane", {})
            if lane_cfg.get("enabled", True):
                from src.lane_detection.lane_detector_factory import (  # noqa: PLC0415
                    LaneDetectorFactory,
                )
                self._lane_detector = LaneDetectorFactory.create(self.config)
            else:
                self._lane_detector = False
        return self._lane_detector if self._lane_detector is not False else None

    @property
    def lane_tracker(self) -> Any:
        """Lazy-initialised lane tracker (or None if disabled)."""
        if self._lane_tracker is None:
            track_cfg = self.config.get("lane_tracking", {})
            if track_cfg.get("enabled", False):
                from src.lane_tracking import PointLaneTracker  # noqa: PLC0415
                self._lane_tracker = PointLaneTracker(track_cfg)
            else:
                self._lane_tracker = False
        return self._lane_tracker if self._lane_tracker is not False else None

    def _detect_lanes(self, frame: Frame) -> Any:
        """Run lane detection and tracking.

        Returns a stabilised ``LaneDetectionResult`` (or ``None`` if disabled).
        """
        det = self.lane_detector
        if det is None:
            return None
        try:
            raw = det.detect(frame.image)
        except FileNotFoundError:
            return None

        tracker = self.lane_tracker
        if tracker is not None:
            h, w = frame.image.shape[:2]
            raw = tracker.process(raw, img_h=h, img_w=w)

        return raw

    def _detect_signs(self, frame: Frame) -> list[Any]:
        """Run sign detection with optional frame skipping."""
        if self._sign_detector is None:
            return []

        signs_cfg = self.config.get("signs", {})
        run_every = int(signs_cfg.get("run_every_n_frames", 2))

        if (
            run_every > 1
            and self._signs_cache
            and (frame.frame_id - self._last_signs_frame_id) < run_every
        ):
            return self._signs_cache

        self._signs_cache = self._sign_detector.detect(frame)
        self._last_signs_frame_id = frame.frame_id
        return self._signs_cache

    def _build_scene(
        self,
        perceived: list[PerceivedObject],
        lane_info: Any,
        signs: list[Any],
        frame: Frame,
    ) -> Any:
        """Synthesise a ``SceneContext`` from objects, lanes, and signs."""
        if self._scene_builder is None:
            return None
        return self._scene_builder.build(perceived, lane_info, signs, frame)

    def _run_behavior(
        self,
        pose_frame: Any,
        motion: dict[int, Any],
        lane_info: Any,
        frame: Frame,
    ) -> Any:
        """Run behaviour classification via the BehaviourSubsystem (if enabled).

        Returns a ``BehaviorFrame`` or ``None`` when behaviour is disabled.
        """
        if self._behavior_subsystem is None:
            return None
        p_mgr = getattr(self._pose_subsystem, "history", None) if self._pose_subsystem is not None else None
        return self._behavior_subsystem.process(
            pose_frame=pose_frame,
            pose_history_mgr=p_mgr,
            track_history_mgr=self.history,
            motion_frame=motion,
            lane_info=lane_info,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )

    def _evaluate_risk(
        self,
        perceived: list[PerceivedObject],
        motion: dict[int, Any],
        ttc: dict[int, float | None],
        lane_info: Any = None,
        scene_context: Any = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Run risk assessment on all perceived objects."""
        return self.risk_engine.evaluate(
            perceived, motion, ttc,
            lane_info=lane_info,
            scene_context=scene_context,
            **kwargs,
        )

    def _process_warnings(
        self, risk_objects: list[Any], frame: Frame
    ) -> list[dict[str, Any]]:
        """Route risk objects through warning channels."""
        return self.warning_manager.process(risk_objects, frame)

    def _estimate_depth(self, frame: Frame) -> np.ndarray:
        """Run depth estimation with optional frame skipping."""
        depth_cfg = self.config.get("depth", {})
        if not depth_cfg.get("enabled", False):
            h, w = frame.image.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        run_every = int(depth_cfg.get("run_every_n_frames", 1))

        if (
            run_every > 1
            and self._depth_cache is not None
            and (frame.frame_id - self._last_depth_frame_id) < run_every
        ):
            return self._depth_cache

        self._depth_cache = self.depth_estimator.estimate(frame)
        self._last_depth_frame_id = frame.frame_id
        return self._depth_cache

    def _estimate_pose(
        self,
        frame: Frame,
        detections: list[Detection],
        tracked: list[TrackedObject],
    ) -> Any:
        """Run pose estimation via the PoseSubsystem (if enabled).

        Returns a ``PoseFrame`` or ``None`` when pose is disabled.
        """
        if self._pose_subsystem is None:
            return None
        return self._pose_subsystem.process(
            frame=frame.image,
            detections=detections,
            tracks=tracked,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )

    def render(self, pipeline_output: dict) -> np.ndarray:
        """Render all visualisation layers onto the raw frame.

        Args:
            pipeline_output: The dict returned by ``process()``.

        Returns:
            An annotated BGR image array.
        """
        renderer = self.renderer
        if renderer is None:
            frame = pipeline_output.get("frame")
            return frame.image.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return renderer.render(pipeline_output)

    def run(self) -> None:
        """Main loop: read frames from the source and process them.

        The caller is responsible for displaying or otherwise consuming the
        result dictionary returned by ``process()``.
        """
        with self.frame_source as source:
            while True:
                frame = source.read()
                if frame is None:
                    break
                self.process(frame)

    @property
    def threat_feature_extractor(self) -> Any:
        """Lazy-initialised threat feature extractor."""
        if self._threat_feature_extractor is None:
            threat_cfg = self.config.get("threat_assessment", {})
            if threat_cfg.get("enabled", True):
                from src.threat.feature_extractor import ThreatFeatureExtractor  # noqa: PLC0415

                self._threat_feature_extractor = ThreatFeatureExtractor(self.config)
            else:
                self._threat_feature_extractor = False
        return self._threat_feature_extractor if self._threat_feature_extractor is not False else None

    @property
    def scenario_classifier(self) -> Any:
        """Lazy-initialised scenario classifier."""
        if self._scenario_classifier is None:
            threat_cfg = self.config.get("threat_assessment", {})
            if threat_cfg.get("enabled", True):
                from src.threat.scenario_classifier import ScenarioClassifier  # noqa: PLC0415

                self._scenario_classifier = ScenarioClassifier(self.config)
            else:
                self._scenario_classifier = False
        return self._scenario_classifier if self._scenario_classifier is not False else None

    def _extract_threat_features(
        self,
        perceived: list[PerceivedObject],
        motion: dict[int, Any],
        ttc: dict[int, float | None],
        lane_info: Any,
        scene_context: Any,
        behavior_frame: Any,
        trajectories: Any,
        frame: Frame,
    ) -> dict[int, Any]:
        """Extract threat features for all perceived objects."""
        extractor = self.threat_feature_extractor
        if extractor is None:
            return {}
        try:
            return extractor.extract(
                perceived_objects=perceived,
                motion_map=motion,
                ttc_map=ttc,
                lane_info=lane_info,
                scene_context=scene_context,
                behavior_frame=behavior_frame,
                trajectories=trajectories,
                frame=frame,
            )
        except Exception as exc:
            logger.warning("Threat feature extraction failed: %s", exc)
            return {}

    def _classify_threat_scenarios(
        self,
        threat_features: dict[int, Any],
        frame: Frame,
    ) -> dict[int, Any]:
        """Classify conflict scenarios for extracted threat features."""
        classifier = self.scenario_classifier
        if classifier is None or not threat_features:
            return {}
        try:
            t = float(frame.timestamp) if frame is not None else 0.0
            return classifier.classify(threat_features, frame_timestamp=t)
        except Exception as exc:
            logger.warning("Scenario classification failed: %s", exc)
            return {}

    def get_threat_features(self) -> dict[int, Any]:
        """Return the most recently extracted threat features."""
        return self._last_threat_features

    def get_threat_scenarios(self) -> dict[int, Any]:
        """Return the most recently classified threat scenarios."""
        return self._last_threat_scenarios

    def get_threat_assessments(self) -> dict[int, Any]:
        """Return the most recently evaluated threat risk assessments."""
        return self._last_threat_assessments

    def get_prioritized_threats(self) -> list[Any]:
        """Return the most recently prioritized threat list."""
        return self._last_prioritized_threats

    def get_primary_threat(self) -> Any | None:
        """Return the top primary threat if present."""
        if self._last_prioritized_threats and getattr(self._last_prioritized_threats[0], "is_primary_threat", False):
            return self._last_prioritized_threats[0]
        return None

    @property
    def threat_risk_engine(self) -> Any:
        """Lazy-initialised threat risk assessment engine."""
        if self._threat_risk_engine is None:
            threat_cfg = self.config.get("threat_assessment", {})
            if threat_cfg.get("enabled", True):
                from src.threat.risk_engine import ScenarioRiskEngine  # noqa: PLC0415

                self._threat_risk_engine = ScenarioRiskEngine(self.config)
            else:
                self._threat_risk_engine = False
        return self._threat_risk_engine if self._threat_risk_engine is not False else None

    @property
    def threat_prioritizer(self) -> Any:
        """Lazy-initialised dynamic threat prioritizer."""
        if self._threat_prioritizer is None:
            threat_cfg = self.config.get("threat_assessment", {})
            if threat_cfg.get("enabled", True):
                from src.threat.prioritizer import ThreatPrioritizer  # noqa: PLC0415

                self._threat_prioritizer = ThreatPrioritizer(self.config)
            else:
                self._threat_prioritizer = False
        return self._threat_prioritizer if self._threat_prioritizer is not False else None

    def _evaluate_threat_risks(
        self,
        threat_features: dict[int, Any],
        threat_scenarios: dict[int, Any],
    ) -> dict[int, Any]:
        """Evaluate risk assessments using scenario-specific models."""
        engine = self.threat_risk_engine
        if engine is None or not threat_features:
            return {}
        try:
            return engine.evaluate(threat_features, threat_scenarios)
        except Exception as exc:
            logger.warning("Threat risk evaluation failed: %s", exc)
            return {}

    def _prioritize_threats(
        self,
        threat_assessments: dict[int, Any],
    ) -> list[Any]:
        """Prioritize and rank active threat assessments."""
        prioritizer = self.threat_prioritizer
        if prioritizer is None or not threat_assessments:
            return []
        try:
            return prioritizer.prioritize(threat_assessments)
        except Exception as exc:
            logger.warning("Threat prioritization failed: %s", exc)
            return []

    @property
    def directional_warning_generator(self) -> Any:
        """Lazy-initialised directional warning generator and event manager."""
        if self._directional_warning_generator is None:
            threat_cfg = self.config.get("threat_assessment", {})
            warn_cfg = threat_cfg.get("directional_warnings", {})
            if threat_cfg.get("enabled", True) and warn_cfg.get("enabled", True):
                from src.threat.warning_generator import DirectionalWarningGenerator  # noqa: PLC0415

                self._directional_warning_generator = DirectionalWarningGenerator(self.config)
            else:
                self._directional_warning_generator = False
        return self._directional_warning_generator if self._directional_warning_generator is not False else None

    def _generate_directional_warnings(
        self,
        threat_assessments: dict[int, Any],
        threat_features: dict[int, Any],
        lane_info: Any,
        frame: Frame,
    ) -> tuple[list[Any], list[Any]]:
        """Generate directional warnings and manage warning event lifecycles."""
        generator = self.directional_warning_generator
        if generator is None:
            return [], []
        try:
            active_warnings = generator.process(
                threat_assessments=threat_assessments,
                threat_features=threat_features,
                lane_info=lane_info,
                frame=frame,
            )
            return active_warnings, generator.get_all_events()
        except Exception as exc:
            logger.warning("Directional warning generation failed: %s", exc)
            return [], []

    def get_directional_warnings(self) -> list[Any]:
        """Return the most recently generated active directional warnings."""
        return self._last_directional_warnings

    def get_directional_warning_events(self) -> list[Any]:
        """Return all tracked directional warning events (active + historical)."""
        return self._last_directional_warning_events

    def finish_threat_session(self) -> None:
        """Finalize any pending active threat warning events when the video stream ends."""
        generator = self.directional_warning_generator
        if generator is not None:
            generator.finish_session(final_timestamp=self._last_processed_timestamp)
            threat_cfg = self.config.get("threat_assessment", {})
            warn_cfg = threat_cfg.get("directional_warnings", {})
            if warn_cfg.get("export", {}).get("export_on_session_end", True):
                try:
                    self.export_threat_evaluation()
                except Exception as exc:
                    logger.warning("Failed to auto-export threat evaluation report: %s", exc)

    def export_threat_evaluation(self, output_path: str | None = None) -> Any:
        """Export threat assessment and warning evaluation workbook to Excel."""
        generator = self.directional_warning_generator
        if generator is None:
            return None

        events = generator.get_all_events()
        history = generator.get_warning_history()
        details = generator.get_risk_assessment_details()
        suppressed = generator.get_suppressed_duplicate_count()

        threat_cfg = self.config.get("threat_assessment", {})
        warn_cfg = threat_cfg.get("directional_warnings", {})
        exp_cfg = warn_cfg.get("export", {})

        output_dir = exp_cfg.get("output_dir", "./reports")
        filename = exp_cfg.get("filename", "Threat_Assessment_Evaluation.xlsx")

        if output_path is not None:
            out_p = Path(output_path)
            output_dir = str(out_p.parent)
            filename = out_p.name

        from src.threat.excel_exporter import ThreatExcelExporter  # noqa: PLC0415

        metadata = {
            "video_path": getattr(self.frame_source, "path", self.config.get("video", {}).get("path", "N/A")),
            "fps": getattr(self.frame_source, "fps", 30.0),
            "width": getattr(self.frame_source, "width", 1280),
            "height": getattr(self.frame_source, "height", 720),
            "frame_count": self._processed_frames,
            "duration_s": self._last_processed_timestamp,
            "processing_fps": getattr(self.frame_source, "fps", 30.0),
        }

        exporter = ThreatExcelExporter(output_dir=output_dir, filename=filename)
        return exporter.export(
            events=events,
            history=history,
            risk_details=details,
            suppressed_duplicates=suppressed,
            metadata=metadata,
            config=self.config,
        )

    def reset_threat_state(self) -> None:
        """Reset internal states of threat extractor, classifier, prioritizer, and warning manager."""
        if self.threat_feature_extractor is not None:
            self.threat_feature_extractor.reset()
        if self.scenario_classifier is not None:
            self.scenario_classifier.reset()
        if self.threat_prioritizer is not None:
            self.threat_prioritizer.reset()
        if self.directional_warning_generator is not None:
            self.directional_warning_generator.reset()
        self._last_threat_features = {}
        self._last_threat_scenarios = {}
        self._last_threat_assessments = {}
        self._last_prioritized_threats = []
        self._last_directional_warnings = []
        self._last_directional_warning_events = []

