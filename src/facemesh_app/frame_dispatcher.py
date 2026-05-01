"""
FrameDispatcher module for FaceMesh application.
Orchestrates the synchronous frame processing pipeline.
"""

import json
import logging
import math
import time
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Any

import cv2
import numpy as np

from .facemesh_dao import (
    FaceMeshEvent,
    safe_float,
)
from .calibration import (
    CalibratedFaceAndGazeEvent,
    CalibrationMatrix,
    CalibrationPoint,
    apply_calibration_model,
    compute_calibration_matrix,
    save_calibration,
)
from .capture import save_capture, build_camera_capture_marked_image
from .capture_window import CaptureWindowManager
from .overlay_calibration import CalibrationOverlayManager
from .overlay_common import get_display_geo
from .overlay_runtime import RuntimeOverlayManager
from .state_machine import StateMachine, DispatcherState
from .pipeline_steps import (
    FaceMeshStep,
    CalibrationAdapterStep,
    CaptureStep,
    OverlayStep,
    UDPForwardStep,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path("face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
CALIBRATION_DATA_DIR = Path("calibration_data")
CALIBRATION_DATAPOINT_DIR = Path("calibration_datapoint")


def ensure_model():
    """Download the FaceLandmarker model if not present."""
    if MODEL_PATH.exists():
        return
    logger.info(f"Downloading FaceMesh model from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    logger.info("FaceMesh model downloaded successfully")


def enrich_runtime_evt(
    evt: Optional[FaceMeshEvent],
    calibrated_evt: Optional[CalibratedFaceAndGazeEvent] = None,
) -> Optional[Dict]:
    """Build a runtime payload that mirrors downstream calibrated outputs."""
    if not evt:
        return None

    raw_left_yaw = evt.left_eye_gaze_yaw
    raw_left_pitch = evt.left_eye_gaze_pitch
    raw_right_yaw = evt.right_eye_gaze_yaw
    raw_right_pitch = evt.right_eye_gaze_pitch

    raw_combined_yaw = None
    if raw_left_yaw is not None and raw_right_yaw is not None:
        raw_combined_yaw = (raw_left_yaw + raw_right_yaw) / 2.0

    raw_combined_pitch = None
    if raw_left_pitch is not None and raw_right_pitch is not None:
        raw_combined_pitch = (raw_left_pitch + raw_right_pitch) / 2.0

    payload = {
        "type": evt.type,
        "hasFace": evt.has_face,
        "landmarkCount": evt.landmark_count,
        "ts": evt.ts,
        "zeta": evt.zeta,
        "head_yaw": evt.head_yaw,
        "head_pitch": evt.head_pitch,
        "head_x": evt.camera_x,
        "head_y": evt.camera_y,
        "head_z": evt.camera_z,
        "head_raw_transform_z": evt.raw_transform_z,
        "raw_left_eye_gaze_yaw": raw_left_yaw,
        "raw_left_eye_gaze_pitch": raw_left_pitch,
        "raw_right_eye_gaze_yaw": raw_right_yaw,
        "raw_right_eye_gaze_pitch": raw_right_pitch,
        "raw_combined_eye_gaze_yaw": raw_combined_yaw,
        "raw_combined_eye_gaze_pitch": raw_combined_pitch,
    }
    if calibrated_evt is not None and evt.has_face:
        corrected_yaw = calibrated_evt.corrected_yaw
        corrected_pitch = calibrated_evt.corrected_pitch
        corrected_screen_x = calibrated_evt.corrected_screen_x
        corrected_screen_y = calibrated_evt.corrected_screen_y
        overlay_x = (
            float(corrected_screen_x) if corrected_screen_x is not None else None
        )
        overlay_y = (
            float(corrected_screen_y) if corrected_screen_y is not None else None
        )
        payload["face_delta_yaw"] = calibrated_evt.face_delta_yaw
        payload["face_delta_pitch"] = calibrated_evt.face_delta_pitch
        payload["corrected_eye_yaw"] = calibrated_evt.corrected_eye_yaw
        payload["corrected_eye_pitch"] = calibrated_evt.corrected_eye_pitch
        payload["corrected_yaw"] = corrected_yaw
        payload["corrected_pitch"] = corrected_pitch
        payload["corrected_screen_x"] = corrected_screen_x
        payload["corrected_screen_y"] = corrected_screen_y
        payload["corrected_yaw_linear"] = calibrated_evt.corrected_yaw_linear
        payload["corrected_pitch_linear"] = calibrated_evt.corrected_pitch_linear
        payload["head_ref_x"] = calibrated_evt.head_ref_x
        payload["head_ref_y"] = calibrated_evt.head_ref_y
        payload["head_ref_z"] = calibrated_evt.head_ref_z
        payload["origin_x"] = calibrated_evt.origin_x
        payload["origin_y"] = calibrated_evt.origin_y
        payload["display_width"] = calibrated_evt.display_width
        payload["display_height"] = calibrated_evt.display_height
        payload["center_eye_yaw"] = calibrated_evt.yaw_calibration
        payload["center_eye_pitch"] = calibrated_evt.pitch_calibration
        payload["face_center_yaw"] = calibrated_evt.face_center_yaw
        payload["face_center_pitch"] = calibrated_evt.face_center_pitch
        payload["face_center_x"] = calibrated_evt.face_center_x
        payload["face_center_y"] = calibrated_evt.face_center_y
        payload["face_center_z"] = calibrated_evt.face_center_z
        payload["center_zeta"] = calibrated_evt.center_zeta
        payload["yaw_coefficient_positive"] = calibrated_evt.yaw_coefficient_positive
        payload["yaw_coefficient_negative"] = calibrated_evt.yaw_coefficient_negative
        payload["pitch_coefficient_positive"] = calibrated_evt.pitch_coefficient_positive
        payload["pitch_coefficient_negative"] = calibrated_evt.pitch_coefficient_negative
        payload["yaw_from_pitch_coupling"] = calibrated_evt.yaw_from_pitch_coupling
        payload["pitch_from_yaw_coupling"] = calibrated_evt.pitch_from_yaw_coupling
        payload["eye_yaw_min"] = calibrated_evt.eye_yaw_min
        payload["eye_yaw_max"] = calibrated_evt.eye_yaw_max
        payload["eye_pitch_min"] = calibrated_evt.eye_pitch_min
        payload["eye_pitch_max"] = calibrated_evt.eye_pitch_max
        payload["screen_center_cam_x"] = calibrated_evt.screen_center_cam_x
        payload["screen_center_cam_y"] = calibrated_evt.screen_center_cam_y
        payload["screen_center_cam_z"] = calibrated_evt.screen_center_cam_z
        payload["screen_axis_x_x"] = calibrated_evt.screen_axis_x_x
        payload["screen_axis_x_y"] = calibrated_evt.screen_axis_x_y
        payload["screen_axis_x_z"] = calibrated_evt.screen_axis_x_z
        payload["screen_axis_y_x"] = calibrated_evt.screen_axis_y_x
        payload["screen_axis_y_y"] = calibrated_evt.screen_axis_y_y
        payload["screen_axis_y_z"] = calibrated_evt.screen_axis_y_z
        payload["screen_scale_x"] = calibrated_evt.screen_scale_x
        payload["screen_scale_y"] = calibrated_evt.screen_scale_y
        payload["screen_fit_rmse"] = calibrated_evt.screen_fit_rmse
        payload["overlay_x"] = overlay_x
        payload["overlay_y"] = overlay_y
    return payload


class FrameDispatcher:
    """Synchronous frame processing dispatcher coordinating pipeline steps."""

    def __init__(
        self,
        args,
        calibration=None,
        overlay_manager=None,
        state_machine=None,
        face_mesh_step=None,
        calibration_adapter_step=None,
        capture_step=None,
        overlay_step=None,
        udp_forward_step=None,
    ):
        self.args = args
        self.calibration = calibration
        self.overlay_manager = overlay_manager
        self.state_machine = state_machine
        self.face_mesh_step = face_mesh_step
        self.calibration_adapter_step = calibration_adapter_step
        self.capture_step = capture_step
        self.overlay_step = overlay_step
        self.udp_forward_step = udp_forward_step

        self.display: Optional[Dict] = None
        self.running = False

        self.display_width = 0
        self.display_height = 0
        self.origin_x = 0.0
        self.origin_y = 0.0

        self._latest_evt: Optional[FaceMeshEvent] = None
        self._latest_calibrated_evt: Optional[CalibratedFaceAndGazeEvent] = None

    def start(self):
        """Initialize display geometry."""
        self.display = get_display_geo()
        self.running = True

    def stop(self):
        """Stop and release overlay resources."""
        self.running = False
        if self.overlay_manager:
            self.overlay_manager.shutdown()
            self.overlay_manager = None

    def _process_frame(
        self, frame: np.ndarray, timestamp_ms: int, pixel_format: str = "bgr"
    ) -> Optional[FaceMeshEvent]:
        """Run FaceMesh detection on a single frame."""
        evt = self.face_mesh_step.receive_frame(frame, timestamp_ms, pixel_format)
        self._latest_evt = evt
        return evt

    def _run_pipeline_steps(
        self,
        frame: np.ndarray,
        evt: Optional[FaceMeshEvent],
        run_downstream: bool = True,
    ) -> Tuple[Optional[CalibratedFaceAndGazeEvent], np.ndarray]:
        """Run calibration adapter and downstream pipeline steps for a frame."""
        calibrated_evt = None
        if self.calibration_adapter_step is not None:
            calibrated_evt = self.calibration_adapter_step.receive_frame(frame, evt)
        self._latest_calibrated_evt = calibrated_evt

        pipeline_frame = frame
        if not run_downstream:
            return calibrated_evt, pipeline_frame

        if self.overlay_step is not None:
            overlay_frame = self.overlay_step.receive_frame(
                pipeline_frame, evt, calibrated_evt
            )
            if overlay_frame is not None:
                pipeline_frame = overlay_frame

        if self.capture_step is not None:
            self.capture_step.receive_frame(pipeline_frame, evt, calibrated_evt)

        if self.udp_forward_step is not None:
            self.udp_forward_step.receive_frame(pipeline_frame, evt, calibrated_evt)

        return calibrated_evt, pipeline_frame

    def _calibration_sample_payload(
        self,
        evt: Optional[FaceMeshEvent],
        calibrated_evt: Optional[CalibratedFaceAndGazeEvent],
        timestamp_ms: int,
        phase: str,
        current_point: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build one calibration diagnostic sample for offline analysis."""
        payload: Dict[str, Any] = {
            "frameTimestampMs": int(timestamp_ms),
            "phase": str(phase),
            "target": {
                "name": current_point.get("name") if current_point else None,
                "x": current_point.get("x") if current_point else None,
                "y": current_point.get("y") if current_point else None,
                "noseX": current_point.get("nose_x") if current_point else None,
                "noseY": current_point.get("nose_y") if current_point else None,
                "eyeX": current_point.get("eye_x") if current_point else None,
                "eyeY": current_point.get("eye_y") if current_point else None,
                "instruction": current_point.get("instruction") if current_point else None,
            },
            "eventTimestampMs": None,
            "hasFace": False,
            "landmarkCount": 0,
            "headYaw": None,
            "headPitch": None,
            "headX": None,
            "headY": None,
            "headZ": None,
            "roll": None,
            "zeta": None,
            "rawCombinedEyeYaw": None,
            "rawCombinedEyePitch": None,
            "rawLeftEyeYaw": None,
            "rawLeftEyePitch": None,
            "rawRightEyeYaw": None,
            "rawRightEyePitch": None,
            "faceDeltaYaw": None,
            "faceDeltaPitch": None,
            "correctedEyeYaw": None,
            "correctedEyePitch": None,
            "correctedYaw": None,
            "correctedPitch": None,
            "correctedScreenX": None,
            "correctedScreenY": None,
            "rawInputs": None,
        }
        if evt is None:
            return payload

        payload["eventTimestampMs"] = int(evt.ts)
        payload["hasFace"] = bool(evt.has_face)
        payload["landmarkCount"] = int(evt.landmark_count)
        payload["headYaw"] = evt.head_yaw
        payload["headPitch"] = evt.head_pitch
        payload["headX"] = evt.camera_x
        payload["headY"] = evt.camera_y
        payload["headZ"] = evt.camera_z
        payload["roll"] = evt.roll
        payload["zeta"] = evt.zeta
        payload["rawCombinedEyeYaw"] = evt.combined_eye_gaze_yaw
        payload["rawCombinedEyePitch"] = evt.combined_eye_gaze_pitch
        payload["rawLeftEyeYaw"] = evt.left_eye_gaze_yaw
        payload["rawLeftEyePitch"] = evt.left_eye_gaze_pitch
        payload["rawRightEyeYaw"] = evt.right_eye_gaze_yaw
        payload["rawRightEyePitch"] = evt.right_eye_gaze_pitch
        if calibrated_evt is not None:
            payload["faceDeltaYaw"] = calibrated_evt.face_delta_yaw
            payload["faceDeltaPitch"] = calibrated_evt.face_delta_pitch
            payload["correctedEyeYaw"] = calibrated_evt.corrected_eye_yaw
            payload["correctedEyePitch"] = calibrated_evt.corrected_eye_pitch
            payload["correctedYaw"] = calibrated_evt.corrected_yaw
            payload["correctedPitch"] = calibrated_evt.corrected_pitch
            payload["correctedScreenX"] = calibrated_evt.corrected_screen_x
            payload["correctedScreenY"] = calibrated_evt.corrected_screen_y
        payload["rawInputs"] = evt.raw_mesh_inputs_dict()
        return payload

    def _save_calibration_session_data(
        self,
        session_timestamp_ms: int,
        samples: List[Dict[str, Any]],
        points: List[CalibrationPoint],
        calib_matrix: Optional[CalibrationMatrix],
    ) -> Path:
        """Persist one calibration session payload for diagnostics."""
        CALIBRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        profile_name = getattr(self.args, "calibration_profile", "") or "default"
        payload = {
            "sessionTimestampMs": int(session_timestamp_ms),
            "profile": profile_name,
            "sampleCount": len(samples),
            "samples": samples,
            "points": [
                {
                    "name": p.name,
                    "screenX": p.screen_x,
                    "screenY": p.screen_y,
                    "rawEyeYaw": p.raw_eye_yaw,
                    "rawEyePitch": p.raw_eye_pitch,
                    "rawLeftEyeYaw": p.raw_left_eye_yaw,
                    "rawLeftEyePitch": p.raw_left_eye_pitch,
                    "rawRightEyeYaw": p.raw_right_eye_yaw,
                    "rawRightEyePitch": p.raw_right_eye_pitch,
                    "headYaw": p.head_yaw,
                    "headPitch": p.head_pitch,
                    "zeta": p.zeta,
                    "headX": p.head_x,
                    "headY": p.head_y,
                    "headZ": p.head_z,
                    "noseTargetX": p.nose_target_x,
                    "noseTargetY": p.nose_target_y,
                    "eyeTargetX": p.eye_target_x,
                    "eyeTargetY": p.eye_target_y,
                    "sampleCount": p.sample_count,
                }
                for p in points
            ],
            "calibrationMatrix": (
                {
                    "centerYaw": calib_matrix.center_yaw,
                    "centerPitch": calib_matrix.center_pitch,
                    "faceCenterYaw": calib_matrix.face_center_yaw,
                    "faceCenterPitch": calib_matrix.face_center_pitch,
                    "centerZeta": calib_matrix.center_zeta,
                    "yawCoefficientPositive": calib_matrix.yaw_coefficient_positive,
                    "yawCoefficientNegative": calib_matrix.yaw_coefficient_negative,
                    "pitchCoefficientPositive": calib_matrix.pitch_coefficient_positive,
                    "pitchCoefficientNegative": calib_matrix.pitch_coefficient_negative,
                    "yawFromPitchCoupling": calib_matrix.yaw_from_pitch_coupling,
                    "pitchFromYawCoupling": calib_matrix.pitch_from_yaw_coupling,
                    "eyeYawMin": calib_matrix.eye_yaw_min,
                    "eyeYawMax": calib_matrix.eye_yaw_max,
                    "eyePitchMin": calib_matrix.eye_pitch_min,
                    "eyePitchMax": calib_matrix.eye_pitch_max,
                    "faceCenterX": calib_matrix.face_center_x,
                    "faceCenterY": calib_matrix.face_center_y,
                    "faceCenterZ": calib_matrix.face_center_z,
                    "screenCenterCamX": calib_matrix.screen_center_cam_x,
                    "screenCenterCamY": calib_matrix.screen_center_cam_y,
                    "screenCenterCamZ": calib_matrix.screen_center_cam_z,
                    "screenAxisXX": calib_matrix.screen_axis_x_x,
                    "screenAxisXY": calib_matrix.screen_axis_x_y,
                    "screenAxisXZ": calib_matrix.screen_axis_x_z,
                    "screenAxisYX": calib_matrix.screen_axis_y_x,
                    "screenAxisYY": calib_matrix.screen_axis_y_y,
                    "screenAxisYZ": calib_matrix.screen_axis_y_z,
                    "screenScaleX": calib_matrix.screen_scale_x,
                    "screenScaleY": calib_matrix.screen_scale_y,
                    "screenFitRmse": calib_matrix.screen_fit_rmse,
                    "sampleCount": calib_matrix.sample_count,
                    "timestampMs": calib_matrix.timestamp_ms,
                }
                if calib_matrix is not None
                else None
            ),
        }
        path = CALIBRATION_DATA_DIR / f"calibration_session_{session_timestamp_ms}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _clear_calibration_session_data(self) -> int:
        """Ensure calibration diagnostics start with one fresh session payload."""
        CALIBRATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
        removed = 0
        for path in CALIBRATION_DATA_DIR.glob("calibration_session_*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                logger.warning("Failed to remove calibration session file: %s", path)
        CALIBRATION_DATAPOINT_DIR.mkdir(parents=True, exist_ok=True)
        for path in CALIBRATION_DATAPOINT_DIR.glob("*"):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                logger.warning("Failed to remove calibration datapoint file: %s", path)
        return removed

    def _save_calibration_datapoint(
        self,
        calib_point: CalibrationPoint,
        evt: Any,
        frame: Any,
        calibrated_evt: Optional[CalibratedFaceAndGazeEvent],
        timestamp_ms: int,
    ) -> None:
        """Dump per-point diagnostics (overlayed PNG + JSON) named by point position."""
        if self.display is None:
            return
        CALIBRATION_DATAPOINT_DIR.mkdir(parents=True, exist_ok=True)
        name = str(calib_point.name)
        png_path = CALIBRATION_DATAPOINT_DIR / f"{name}.png"
        json_path = CALIBRATION_DATAPOINT_DIR / f"{name}.json"

        landmarks = None
        if isinstance(evt, FaceMeshEvent) and evt.landmarks:
            landmarks = list(evt.landmarks)
        elif isinstance(evt, dict):
            landmarks = evt.get("landmarks")
        snap = {"evt": evt, "frame": frame, "landmarks": landmarks}
        nose_click = (
            float(calib_point.nose_target_x)
            if calib_point.nose_target_x is not None
            else float(calib_point.screen_x),
            float(calib_point.nose_target_y)
            if calib_point.nose_target_y is not None
            else float(calib_point.screen_y),
        )
        img, err = build_camera_capture_marked_image(
            snap,
            overlay_w=float(self.display["width"]),
            overlay_h=float(self.display["height"]),
            click_pos=nose_click,
            draw_click=True,
            draw_info_panel=True,
        )
        if img is not None:
            cv2.imwrite(str(png_path), img)
        elif err:
            logger.warning("Calibration datapoint image failed for %s: %s", name, err)

        payload = {
            "timestamp_ms": timestamp_ms,
            "name": name,
            "target": {
                "screen_x": float(calib_point.screen_x),
                "screen_y": float(calib_point.screen_y),
                "nose_x": calib_point.nose_target_x,
                "nose_y": calib_point.nose_target_y,
                "eye_x": calib_point.eye_target_x,
                "eye_y": calib_point.eye_target_y,
            },
            "head": {
                "yaw": calib_point.head_yaw,
                "pitch": calib_point.head_pitch,
                "x_mm": calib_point.head_x,
                "y_mm": calib_point.head_y,
                "z_mm": calib_point.head_z,
                "zeta_mm": calib_point.zeta,
            },
            "gaze": {
                "raw_eye_yaw": calib_point.raw_eye_yaw,
                "raw_eye_pitch": calib_point.raw_eye_pitch,
                "raw_left_eye_yaw": calib_point.raw_left_eye_yaw,
                "raw_left_eye_pitch": calib_point.raw_left_eye_pitch,
                "raw_right_eye_yaw": calib_point.raw_right_eye_yaw,
                "raw_right_eye_pitch": calib_point.raw_right_eye_pitch,
            },
            "sample_count": calib_point.sample_count,
        }
        if calibrated_evt is not None:
            payload["calibrated_gaze"] = {
                "gaze_x": getattr(calibrated_evt, "gaze_x", None),
                "gaze_y": getattr(calibrated_evt, "gaze_y", None),
            }
        try:
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            logger.warning("Calibration datapoint JSON failed for %s: %s", name, exc)

    def _transition_state(self, new_state: DispatcherState) -> None:
        if self.state_machine is None:
            return
        current_state = self.state_machine.get_state()
        if current_state == new_state:
            return
        self.state_machine.transition_to(new_state)

    def run_capture_loop(
        self, camera_reader, on_capture_click: Optional[Callable] = None
    ):
        """Run the main capture and display loop until user exits."""
        if self.display is None:
            raise RuntimeError("FrameDispatcher not started")
        self.start_operational()

        overlay_enabled = bool(self.args.overlay)
        capture_enabled = bool(self.args.capture)
        capture_live_enabled = bool(capture_enabled and self.args.capture_live)
        quiet = bool(getattr(self.args, "quiet", False))
        log_interval = float(getattr(self.args, "log_interval", 2.0))

        w = int(self.display["width"])
        h = int(self.display["height"])
        overlay_manager: Optional[RuntimeOverlayManager] = None
        capture_window_manager: Optional[CaptureWindowManager] = None
        last_log_time = 0.0
        running = True
        pixel_format = camera_reader.pixel_format

        camera_fps = float(getattr(camera_reader, "fps", 0.0) or 0.0)
        camera_period_ms = (1000.0 / camera_fps) if camera_fps > 0 else 0.0
        ewma_read_ms: Optional[float] = None
        ewma_proc_ms: Optional[float] = None
        ewma_alpha = 0.1
        frames_since_log = 0

        try:
            if overlay_enabled:
                overlay_manager = RuntimeOverlayManager(
                    self.display,
                    capture_enabled=False,
                    overlay_fps=self.args.overlay_fps,
                    click_through=True,
                )
                overlay_manager.initialize()

            if capture_enabled:
                capture_window_manager = CaptureWindowManager(self.display)
                capture_window_manager.initialize()

            while running:
                t_read_start = time.perf_counter()
                frame, timestamp_ms = camera_reader.read_frame()
                t_read_ms = (time.perf_counter() - t_read_start) * 1000.0
                if frame is None:
                    time.sleep(0.001)
                    continue

                t_proc_start = time.perf_counter()
                evt = self._process_frame(frame, timestamp_ms, pixel_format)
                calibrated_evt, _ = self._run_pipeline_steps(
                    frame, evt, run_downstream=True
                )
                t_proc_ms = (time.perf_counter() - t_proc_start) * 1000.0

                ewma_read_ms = (
                    t_read_ms
                    if ewma_read_ms is None
                    else ewma_alpha * t_read_ms + (1.0 - ewma_alpha) * ewma_read_ms
                )
                ewma_proc_ms = (
                    t_proc_ms
                    if ewma_proc_ms is None
                    else ewma_alpha * t_proc_ms + (1.0 - ewma_alpha) * ewma_proc_ms
                )
                frames_since_log += 1

                runtime_evt = enrich_runtime_evt(evt, calibrated_evt)

                if not quiet and log_interval > 0:
                    now = time.time()
                    if now - last_log_time >= log_interval:
                        elapsed = (
                            now - last_log_time if last_log_time > 0 else log_interval
                        )
                        processed_fps = (
                            frames_since_log / elapsed if elapsed > 0 else 0.0
                        )
                        buffering = (
                            camera_period_ms > 0
                            and ewma_proc_ms is not None
                            and ewma_read_ms is not None
                            and ewma_proc_ms > camera_period_ms
                            and ewma_read_ms < 0.5 * camera_period_ms
                        )
                        logger.info(
                            f"Pipeline: fps={processed_fps:.1f} "
                            f"t_read={ewma_read_ms:.1f}ms "
                            f"t_proc={ewma_proc_ms:.1f}ms "
                            f"cam_period={camera_period_ms:.1f}ms"
                            + (" [BUFFERING]" if buffering else "")
                        )
                        last_log_time = now
                        frames_since_log = 0
                        if evt is not None and evt.has_face:
                            logger.info(
                                f"Face detected - landmarks: {evt.landmark_count} "
                                f"head=({safe_float(evt.head_yaw):.1f}, {safe_float(evt.head_pitch):.1f}) "
                                f"gaze=({safe_float(evt.combined_eye_gaze_yaw):.1f}, "
                                f"{safe_float(evt.combined_eye_gaze_pitch):.1f}) "
                                f"position=({safe_float(evt.x):.1f}, {safe_float(evt.y):.1f}, {safe_float(evt.z):.1f})"
                            )
                        elif evt is not None:
                            logger.info("No face detected")

                if overlay_manager is not None:
                    overlay_manager.handle_events()
                    if not overlay_manager.is_running():
                        running = False
                        break

                capture_live_img = None
                if capture_live_enabled and capture_window_manager is not None:
                    mouse_x, mouse_y = capture_window_manager.get_mouse_position()
                    snap = {
                        "evt": evt,
                        "frame": frame,
                        "landmarks": list(evt.landmarks) if evt and evt.landmarks else None,
                    }
                    capture_live_img, _ = build_camera_capture_marked_image(
                        snap,
                        overlay_w=float(w),
                        overlay_h=float(h),
                        click_pos=(mouse_x, mouse_y),
                        draw_click=False,
                        draw_info_panel=False,
                    )

                if capture_window_manager is not None:
                    capture_window_manager.render(runtime_evt, capture_live_img)
                    if not capture_window_manager.is_running():
                        running = False
                        break
                    clicked = capture_window_manager.consume_click()
                    if clicked is not None:
                        if on_capture_click is not None:
                            on_capture_click(clicked)
                        else:
                            save_capture(
                                self.display,
                                w,
                                h,
                                clicked,
                                frame,
                                evt,
                                runtime_evt=runtime_evt,
                            )

                if overlay_manager is not None:
                    overlay_manager.render_mesh(runtime_evt)
        finally:
            if overlay_manager is not None:
                overlay_manager.shutdown()
            if capture_window_manager is not None:
                capture_window_manager.shutdown()

    def run_calibration_workflow(
        self, camera_reader
    ) -> Tuple[Optional[CalibrationMatrix], List[CalibrationPoint]]:
        """Execute the 9-point calibration workflow with on-screen guidance."""
        if self.display is None:
            raise RuntimeError("FrameDispatcher not started")
        self.start_calibration()

        logger.info("Starting 9-point calibration workflow...")
        print("Starting 9-point calibration workflow...", flush=True)
        print(
            "Look forward for center, then align nose and eye with the dual targets at each step.",
            flush=True,
        )
        cleared_sessions = self._clear_calibration_session_data()
        if cleared_sessions > 0:
            print(
                f"Removed {cleared_sessions} previous calibration session file(s).",
                flush=True,
            )

        try:
            if not isinstance(self.overlay_manager, CalibrationOverlayManager):
                if self.overlay_manager is not None:
                    self.overlay_manager.shutdown()
                self.overlay_manager = CalibrationOverlayManager(
                    self.display,
                    overlay_fps=self.args.overlay_fps,
                )
            self.overlay_manager.initialize()
            self.overlay_manager.start_calibration_sequence(
                self.display["width"], self.display["height"]
            )

            calib_points: List[CalibrationPoint] = []
            session_timestamp_ms = int(time.time() * 1000)
            calibration_samples: List[Dict[str, Any]] = []
            pixel_format = camera_reader.pixel_format

            while True:
                frame, timestamp_ms = camera_reader.read_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                evt = self._process_frame(frame, timestamp_ms, pixel_format)
                calibrated_evt, _ = self._run_pipeline_steps(
                    frame, evt, run_downstream=False
                )

                self.overlay_manager.handle_events()
                if not self.overlay_manager.is_running():
                    print("Calibration cancelled by user.", flush=True)
                    break

                calibration_samples.append(
                    self._calibration_sample_payload(
                        evt=evt,
                        calibrated_evt=calibrated_evt,
                        timestamp_ms=timestamp_ms,
                        phase=self.overlay_manager.get_calibration_phase(),
                        current_point=self.overlay_manager.get_current_calib_point(),
                    )
                )

                evt_dict = enrich_runtime_evt(evt, calibrated_evt)

                completed, calib_point = self.overlay_manager.update_calibration_state(
                    evt_dict
                )
                self.overlay_manager.render_mesh(evt_dict)

                if calib_point is not None:
                    calib_points.append(calib_point)

                    print(
                        f"Calibration point {len(calib_points)}/9 completed at '{calib_point.name}' "
                        f"head=({calib_point.head_yaw:.2f}, {calib_point.head_pitch:.2f}) "
                        f"head_xyz=({calib_point.head_x:.3f}, {calib_point.head_y:.3f}, {calib_point.head_z:.3f}) "
                        f"eye=({calib_point.raw_eye_yaw:.2f}, {calib_point.raw_eye_pitch:.2f}) "
                        f"zeta={calib_point.zeta:.2f}",
                        flush=True,
                    )

                    self._save_calibration_datapoint(
                        calib_point=calib_point,
                        evt=evt,
                        frame=frame,
                        calibrated_evt=calibrated_evt,
                        timestamp_ms=timestamp_ms,
                    )

                if completed:
                    if len(calib_points) == 9:
                        print("All 9 calibration points collected.", flush=True)
                    else:
                        print(
                            f"Calibration sequence ended with {len(calib_points)} points.",
                            flush=True,
                        )
                    break

                time.sleep(0.001)

            calib_matrix = None
            if len(calib_points) == 9:
                print("Computing calibration matrix...", flush=True)
                display = getattr(self, "display", None) or {}
                width_px = float(display.get("width", 0) or 0)
                height_px = float(display.get("height", 0) or 0)
                width_mm = float(display.get("width_mm", 0) or 0)
                height_mm = float(display.get("height_mm", 0) or 0)
                px_per_mm_x = width_px / width_mm if width_px > 0 and width_mm > 0 else None
                px_per_mm_y = height_px / height_mm if height_px > 0 and height_mm > 0 else None
                if px_per_mm_x and px_per_mm_y:
                    print(
                        f"Display physical size: {width_px:.0f}x{height_px:.0f}px / "
                        f"{width_mm:.0f}x{height_mm:.0f}mm => "
                        f"scales=({px_per_mm_x:.3f}, {px_per_mm_y:.3f}) px/mm",
                        flush=True,
                    )
                else:
                    print(
                        "Display physical size unavailable; falling back to unconstrained fit",
                        flush=True,
                    )
                calib_matrix = compute_calibration_matrix(
                    calib_points,
                    px_per_mm_x=px_per_mm_x,
                    px_per_mm_y=px_per_mm_y,
                )

                profile_name = (
                    getattr(self.args, "calibration_profile", "") or "default"
                )
                calib_path = save_calibration(calib_matrix, calib_points, profile_name)

                print(f"Calibration saved to: {calib_path}", flush=True)

                origin_x = safe_float(
                    getattr(
                        next((p for p in calib_points if p.name == "C"), calib_points[0]),
                        "screen_x",
                        0.0,
                    ),
                    0.0,
                )
                origin_y = safe_float(
                    getattr(
                        next((p for p in calib_points if p.name == "C"), calib_points[0]),
                        "screen_y",
                        0.0,
                    ),
                    0.0,
                )
                print("Calibration round-trip (eye target vs apply):", flush=True)
                pixel_errors: List[float] = []
                for point in calib_points:
                    result = apply_calibration_model(
                        raw_eye_yaw=point.raw_eye_yaw,
                        raw_eye_pitch=point.raw_eye_pitch,
                        head_yaw=point.head_yaw,
                        head_pitch=point.head_pitch,
                        head_x=point.head_x,
                        head_y=point.head_y,
                        head_z=point.head_z,
                        center_eye_yaw=calib_matrix.center_yaw,
                        center_eye_pitch=calib_matrix.center_pitch,
                        face_center_yaw=calib_matrix.face_center_yaw,
                        face_center_pitch=calib_matrix.face_center_pitch,
                        yaw_coefficient_positive=calib_matrix.yaw_coefficient_positive,
                        yaw_coefficient_negative=calib_matrix.yaw_coefficient_negative,
                        pitch_coefficient_positive=calib_matrix.pitch_coefficient_positive,
                        pitch_coefficient_negative=calib_matrix.pitch_coefficient_negative,
                        yaw_from_pitch_coupling=calib_matrix.yaw_from_pitch_coupling,
                        pitch_from_yaw_coupling=calib_matrix.pitch_from_yaw_coupling,
                        eye_yaw_min=calib_matrix.eye_yaw_min,
                        eye_yaw_max=calib_matrix.eye_yaw_max,
                        eye_pitch_min=calib_matrix.eye_pitch_min,
                        eye_pitch_max=calib_matrix.eye_pitch_max,
                        center_zeta=calib_matrix.center_zeta,
                        face_center_x=calib_matrix.face_center_x,
                        face_center_y=calib_matrix.face_center_y,
                        face_center_z=calib_matrix.face_center_z,
                        screen_center_cam_x=calib_matrix.screen_center_cam_x,
                        screen_center_cam_y=calib_matrix.screen_center_cam_y,
                        screen_center_cam_z=calib_matrix.screen_center_cam_z,
                        screen_axis_x_x=calib_matrix.screen_axis_x_x,
                        screen_axis_x_y=calib_matrix.screen_axis_x_y,
                        screen_axis_x_z=calib_matrix.screen_axis_x_z,
                        screen_axis_y_x=calib_matrix.screen_axis_y_x,
                        screen_axis_y_y=calib_matrix.screen_axis_y_y,
                        screen_axis_y_z=calib_matrix.screen_axis_y_z,
                        screen_scale_x=calib_matrix.screen_scale_x,
                        screen_scale_y=calib_matrix.screen_scale_y,
                        screen_fit_rmse=calib_matrix.screen_fit_rmse,
                        origin_x=origin_x,
                        origin_y=origin_y,
                    )
                    got_x = result.get("corrected_screen_x")
                    got_y = result.get("corrected_screen_y")
                    target_x = (
                        point.eye_target_x
                        if point.eye_target_x is not None
                        else point.screen_x
                    )
                    target_y = (
                        point.eye_target_y
                        if point.eye_target_y is not None
                        else point.screen_y
                    )
                    if got_x is None or got_y is None:
                        print(f"  {point.name:>3s}: projection failed", flush=True)
                        continue
                    err_x = float(got_x) - float(target_x)
                    err_y = float(got_y) - float(target_y)
                    pixel_errors.append(math.hypot(err_x, err_y))
                    print(
                        f"  {point.name:>3s}: target=({float(target_x):7.1f},{float(target_y):7.1f}) "
                        f"apply=({float(got_x):7.1f},{float(got_y):7.1f}) "
                        f"err=({err_x:+7.1f},{err_y:+7.1f})px",
                        flush=True,
                    )
                if pixel_errors:
                    max_err = max(pixel_errors)
                    mean_err = sum(pixel_errors) / len(pixel_errors)
                    print(
                        f"  round-trip pixel error: mean={mean_err:.2f}px max={max_err:.2f}px",
                        flush=True,
                    )

                print(
                    "Calibration matrix: "
                    f"eye_zero=({calib_matrix.center_yaw:.4f}, {calib_matrix.center_pitch:.4f}) "
                    f"face_zero=({calib_matrix.face_center_yaw:.4f}, {calib_matrix.face_center_pitch:.4f}) "
                    f"yaw_coeff=({calib_matrix.yaw_coefficient_negative:.4f}, {calib_matrix.yaw_coefficient_positive:.4f}) "
                    f"pitch_coeff=({calib_matrix.pitch_coefficient_negative:.4f}, {calib_matrix.pitch_coefficient_positive:.4f}) "
                    f"cross=({calib_matrix.yaw_from_pitch_coupling:.4f}, {calib_matrix.pitch_from_yaw_coupling:.4f}) "
                    f"eye_yaw_range=({calib_matrix.eye_yaw_min:.4f}, {calib_matrix.eye_yaw_max:.4f}) "
                    f"eye_pitch_range=({calib_matrix.eye_pitch_min:.4f}, {calib_matrix.eye_pitch_max:.4f}) "
                    f"screen_scale=({calib_matrix.screen_scale_x:.4f}, {calib_matrix.screen_scale_y:.4f}) "
                    f"zeta={calib_matrix.center_zeta:.4f} "
                    f"screen_fit_rmse={calib_matrix.screen_fit_rmse:.4f} "
                    f"samples={calib_matrix.sample_count}",
                    flush=True,
                )
            else:
                print(
                    f"Insufficient calibration points ({len(calib_points)}/9). Cannot compute calibration matrix.",
                    flush=True,
                )

            if calib_matrix is not None:
                self.set_calibration(calib_matrix)
                self.start_operational()

            calibration_data_path = self._save_calibration_session_data(
                session_timestamp_ms=session_timestamp_ms,
                samples=calibration_samples,
                points=calib_points,
                calib_matrix=calib_matrix,
            )
            print(f"Calibration diagnostics saved to: {calibration_data_path}", flush=True)

            return calib_matrix, calib_points

        except Exception as e:
            print(f"Error during calibration: {e}", flush=True)
            logger.exception("Calibration workflow exception")
            raise
        finally:
            if self.overlay_manager:
                self.overlay_manager.shutdown()
                print("Calibration overlay shutdown complete.", flush=True)

    def set_calibration(self, calibration: CalibrationMatrix) -> None:
        """Apply a new calibration matrix to the dispatcher."""
        self.calibration = calibration
        if self.calibration_adapter_step is not None:
            self.calibration_adapter_step.update_calibration(
                pitch=calibration.center_pitch,
                yaw=calibration.center_yaw,
                roll=0.0,
                face_center_yaw=calibration.face_center_yaw,
                face_center_pitch=calibration.face_center_pitch,
                center_zeta=calibration.center_zeta,
                yaw_coefficient_positive=calibration.yaw_coefficient_positive,
                yaw_coefficient_negative=calibration.yaw_coefficient_negative,
                pitch_coefficient_positive=calibration.pitch_coefficient_positive,
                pitch_coefficient_negative=calibration.pitch_coefficient_negative,
                yaw_from_pitch_coupling=calibration.yaw_from_pitch_coupling,
                pitch_from_yaw_coupling=calibration.pitch_from_yaw_coupling,
                eye_yaw_min=calibration.eye_yaw_min,
                eye_yaw_max=calibration.eye_yaw_max,
                eye_pitch_min=calibration.eye_pitch_min,
                eye_pitch_max=calibration.eye_pitch_max,
                face_center_x=calibration.face_center_x,
                face_center_y=calibration.face_center_y,
                face_center_z=calibration.face_center_z,
                screen_center_cam_x=calibration.screen_center_cam_x,
                screen_center_cam_y=calibration.screen_center_cam_y,
                screen_center_cam_z=calibration.screen_center_cam_z,
                screen_axis_x_x=calibration.screen_axis_x_x,
                screen_axis_x_y=calibration.screen_axis_x_y,
                screen_axis_x_z=calibration.screen_axis_x_z,
                screen_axis_y_x=calibration.screen_axis_y_x,
                screen_axis_y_y=calibration.screen_axis_y_y,
                screen_axis_y_z=calibration.screen_axis_y_z,
                screen_scale_x=calibration.screen_scale_x,
                screen_scale_y=calibration.screen_scale_y,
                screen_fit_rmse=calibration.screen_fit_rmse,
            )

    def get_latest_event(self) -> Optional[FaceMeshEvent]:
        """Return the most recent FaceMeshEvent."""
        return self._latest_evt

    def is_running(self) -> bool:
        """Check if the dispatcher is actively processing."""
        return self.running

    def _handle_calibration_complete(self, calibration_result: dict) -> None:
        """Apply completed calibration result and transition to OPERATIONAL state."""
        pitch = calibration_result.get("pitch", 0.0)
        yaw = calibration_result.get("yaw", 0.0)
        roll = calibration_result.get("roll", 0.0)

        self.calibration_adapter_step.update_calibration(
            pitch=pitch, yaw=yaw, roll=roll
        )
        self._transition_state(DispatcherState.OPERATIONAL)

    def start_calibration(self) -> None:
        """Transition to CALIBRATION state."""
        self._transition_state(DispatcherState.CALIBRATION)

    def start_operational(self) -> None:
        """Transition to OPERATIONAL state."""
        self._transition_state(DispatcherState.OPERATIONAL)

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable or disable the capture pipeline step."""
        if self.capture_step is not None:
            self.capture_step.set_enabled(enabled)

    def set_overlay_enabled(self, enabled: bool) -> None:
        """Enable or disable the overlay pipeline step."""
        if self.overlay_step is not None:
            self.overlay_step.set_enabled(enabled)

    def set_overlay_show_hud(self, show_hud: bool) -> None:
        """Toggle HUD display on the overlay step."""
        if self.overlay_step is not None:
            self.overlay_step.set_show_hud(show_hud)

    def set_udp_forwarding_enabled(self, enabled: bool) -> None:
        """Enable or disable UDP data forwarding."""
        if self.udp_forward_step is not None:
            self.udp_forward_step.set_enabled(enabled)

    def update_calibration(self, pitch: float, yaw: float, roll: float) -> None:
        """Update the calibration adapter's pitch/yaw/roll offsets."""
        self.calibration_adapter_step.update_calibration(
            pitch=pitch, yaw=yaw, roll=roll
        )

    def update_display_geometry(
        self, width: int, height: int, origin_x: float, origin_y: float
    ) -> None:
        """Propagate new display dimensions to the calibration adapter."""
        self.display_width = width
        self.display_height = height
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.calibration_adapter_step.update_display_geometry(
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
        )

    def get_state(self) -> DispatcherState:
        """Return the current dispatcher state."""
        return self.state_machine.get_state()

    def set_state_transition_callback(
        self, callback: Callable[[DispatcherState, DispatcherState], None]
    ) -> None:
        """Register a callback for state machine transitions."""
        self.state_machine.set_transition_callback(callback)

    def clear_state_transition_callback(self) -> None:
        """Remove any registered state transition callback."""
        self.state_machine.clear_transition_callback()
