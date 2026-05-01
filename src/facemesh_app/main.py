#!/usr/bin/env python3
"""
FaceMesh Data Capture Application
Main entry point for raw face mesh data capture.
"""

import argparse
import logging
import os
import sys

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from facemesh_app.calibration import load_calibration
from facemesh_app.camera_reader import CameraReader
from facemesh_app.frame_dispatcher import FrameDispatcher, ensure_model, MODEL_PATH
from facemesh_app.overlay_common import get_display_geo
from facemesh_app.pipeline_steps import (
    FaceMeshStep,
    CalibrationAdapterStep,
    CaptureStep,
    OverlayStep,
    UDPForwardStep,
)
from facemesh_app.state_machine import StateMachine

logger = logging.getLogger(__name__)


def _backend_string_to_int(backend_str: str) -> int:
    """Convert backend string to cv2 CAP_* constant.

    Args:
        backend_str: Backend string ('auto', 'msmf', 'dshow', 'any')

    Returns:
        Corresponding cv2 CAP_* constant (defaults to CAP_ANY)
    """
    backend_map = {
        "auto": cv2.CAP_ANY,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "any": cv2.CAP_ANY,
    }
    return backend_map.get(backend_str.lower(), cv2.CAP_ANY)


def _env_int(key: str, default: str) -> int:
    raw = os.getenv(key, default)
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"Environment variable {key}='{raw}' is not a valid integer, using default {default}"
        )
        return int(default)


def _env_float(key: str, default: str) -> float:
    raw = os.getenv(key, default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            f"Environment variable {key}='{raw}' is not a valid float, using default {default}"
        )
        return float(default)


def parse_args():
    parser = argparse.ArgumentParser(description="FaceMesh data capture app")

    parser.add_argument(
        "--overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show transparent overlay window",
    )
    parser.add_argument(
        "--capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture mode",
    )
    parser.add_argument(
        "--capture-live",
        "--live",
        dest="capture_live",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live camera+mesh content in capture window",
    )
    parser.add_argument(
        "--udp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward calibrated output via UDP",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--log-interval", type=float, default=2.0, help="Log interval in seconds"
    )
    parser.add_argument(
        "--overlay-fps", type=int, default=60, help="Overlay refresh rate"
    )

    parser.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run 9-point calibration workflow",
    )
    parser.add_argument(
        "--calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --calibrate",
    )
    parser.add_argument(
        "--calibration-profile",
        type=str,
        default="",
        help="Calibration profile name",
    )
    parser.add_argument(
        "--force-recalibrate",
        action="store_true",
        help="Ignore existing calibration and recalibrate",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=5,
        help="Minimum number of calibration samples to collect (default: 5)",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=_env_int("CAMERA_INDEX", "0"),
        help="Camera device index",
    )
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "msmf", "dshow", "any"],
        default=os.getenv("CAMERA_BACKEND", "dshow").lower(),
        help="Camera backend",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=_env_int("CAMERA_WIDTH", "1920"),
        help="Camera width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=_env_int("CAMERA_HEIGHT", "1080"),
        help="Camera height",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=_env_float("CAMERA_FPS", "30"),
        help="Camera FPS",
    )
    parser.add_argument(
        "--camera-fourcc",
        type=str,
        default=os.getenv("CAMERA_FOURCC", "MJPG"),
        help="Camera codec",
    )

    parser.add_argument(
        "--udp-host",
        type=str,
        default=os.getenv("UDP_HOST", "127.0.0.1"),
        help="UDP forward target host",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=_env_int("UDP_PORT", "4242"),
        help="UDP forward target port",
    )

    return parser.parse_args()


def main():
    if os.getenv("FACEMESH_PROFILE") == "1" and not os.getenv("_FACEMESH_PROFILE_ACTIVE"):
        os.environ["_FACEMESH_PROFILE_ACTIVE"] = "1"
        _run_with_yappi()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    args = parse_args()
    if args.capture_live:
        args.capture = True

    no_explicit_mode = not (
        args.overlay
        or args.capture
        or args.capture_live
        or args.udp
        or args.calibrate
        or args.calibration
        or args.force_recalibrate
    )

    calibration = None
    if not args.force_recalibrate and not args.calibrate and not args.calibration:
        try:
            calibration, _ = load_calibration(args.calibration_profile)
        except Exception as e:
            logger.warning(
                f"Failed to load calibration profile '{args.calibration_profile}': {e}"
            )
            calibration = None

        if calibration is not None and calibration.sample_count > 0:
            profile_name = args.calibration_profile or "default"
            logger.info(
                f"Loaded calibration from profile '{profile_name}': "
                f"eye_zero=({calibration.center_yaw:.4f}, {calibration.center_pitch:.4f}) "
                f"face_zero=({calibration.face_center_yaw:.4f}, {calibration.face_center_pitch:.4f}) "
                f"yaw_coeff=({calibration.yaw_coefficient_negative:.4f}, {calibration.yaw_coefficient_positive:.4f}) "
                f"pitch_coeff=({calibration.pitch_coefficient_negative:.4f}, {calibration.pitch_coefficient_positive:.4f}) "
                f"cross=({calibration.yaw_from_pitch_coupling:.4f}, {calibration.pitch_from_yaw_coupling:.4f}) "
                f"eye_yaw_range=({calibration.eye_yaw_min:.4f}, {calibration.eye_yaw_max:.4f}) "
                f"eye_pitch_range=({calibration.eye_pitch_min:.4f}, {calibration.eye_pitch_max:.4f}) "
                f"screen_scale=({calibration.screen_scale_x:.4f}, {calibration.screen_scale_y:.4f}) "
                f"zeta={calibration.center_zeta:.4f} "
                f"screen_fit_rmse={calibration.screen_fit_rmse:.4f} "
                f"samples={calibration.sample_count}",
            )
        else:
            logger.info("No existing calibration found. Running in uncalibrated mode.")

    auto_transition_to_udp = False
    if no_explicit_mode:
        if calibration is not None and calibration.sample_count > 0:
            args.udp = True
            logger.info(
                "No mode specified; existing calibration found. Starting UDP forwarder."
            )
        else:
            args.calibrate = True
            auto_transition_to_udp = True
            logger.info(
                "No mode specified and no calibration on disk. "
                "Running calibration, then UDP forwarder."
            )

    state_machine = StateMachine()

    display = get_display_geo()

    try:
        ensure_model()
    except Exception as e:
        logger.error(f"Failed to download FaceMesh model: {e}")
        raise

    try:
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            output_face_blendshapes=bool(args.capture),
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(opts)
    except Exception as e:
        logger.error(f"Failed to initialize FaceLandmarker: {e}")
        raise

    face_mesh_step = FaceMeshStep(face_landmarker)

    pitch_calib = calibration.center_pitch if calibration else 0.0
    yaw_calib = calibration.center_yaw if calibration else 0.0
    roll_calib = 0.0
    face_center_yaw = calibration.face_center_yaw if calibration else 0.0
    face_center_pitch = calibration.face_center_pitch if calibration else 0.0
    center_zeta = calibration.center_zeta if calibration else 1200.0
    yaw_coefficient_positive = calibration.yaw_coefficient_positive if calibration else 1.0
    yaw_coefficient_negative = calibration.yaw_coefficient_negative if calibration else 1.0
    pitch_coefficient_positive = (
        calibration.pitch_coefficient_positive if calibration else 1.0
    )
    pitch_coefficient_negative = (
        calibration.pitch_coefficient_negative if calibration else 1.0
    )
    yaw_from_pitch_coupling = calibration.yaw_from_pitch_coupling if calibration else 0.0
    pitch_from_yaw_coupling = calibration.pitch_from_yaw_coupling if calibration else 0.0
    eye_yaw_min = calibration.eye_yaw_min if calibration else -1.0
    eye_yaw_max = calibration.eye_yaw_max if calibration else 1.0
    eye_pitch_min = calibration.eye_pitch_min if calibration else -1.0
    eye_pitch_max = calibration.eye_pitch_max if calibration else 1.0
    face_center_x = calibration.face_center_x if calibration else 0.0
    face_center_y = calibration.face_center_y if calibration else 0.0
    face_center_z = calibration.face_center_z if calibration else center_zeta
    screen_center_cam_x = calibration.screen_center_cam_x if calibration else 0.0
    screen_center_cam_y = calibration.screen_center_cam_y if calibration else 0.0
    screen_center_cam_z = calibration.screen_center_cam_z if calibration else center_zeta
    screen_axis_x_x = calibration.screen_axis_x_x if calibration else 1.0
    screen_axis_x_y = calibration.screen_axis_x_y if calibration else 0.0
    screen_axis_x_z = calibration.screen_axis_x_z if calibration else 0.0
    screen_axis_y_x = calibration.screen_axis_y_x if calibration else 0.0
    screen_axis_y_y = calibration.screen_axis_y_y if calibration else 1.0
    screen_axis_y_z = calibration.screen_axis_y_z if calibration else 0.0
    screen_scale_x = calibration.screen_scale_x if calibration else 1.0
    screen_scale_y = calibration.screen_scale_y if calibration else 1.0
    screen_fit_rmse = calibration.screen_fit_rmse if calibration else -1.0
    calibration_adapter_step = CalibrationAdapterStep(
        pitch_calibration=pitch_calib,
        yaw_calibration=yaw_calib,
        roll_calibration=roll_calib,
        face_center_yaw=face_center_yaw,
        face_center_pitch=face_center_pitch,
        center_zeta=center_zeta,
        yaw_coefficient_positive=yaw_coefficient_positive,
        yaw_coefficient_negative=yaw_coefficient_negative,
        pitch_coefficient_positive=pitch_coefficient_positive,
        pitch_coefficient_negative=pitch_coefficient_negative,
        yaw_from_pitch_coupling=yaw_from_pitch_coupling,
        pitch_from_yaw_coupling=pitch_from_yaw_coupling,
        eye_yaw_min=eye_yaw_min,
        eye_yaw_max=eye_yaw_max,
        eye_pitch_min=eye_pitch_min,
        eye_pitch_max=eye_pitch_max,
        face_center_x=face_center_x,
        face_center_y=face_center_y,
        face_center_z=face_center_z,
        screen_center_cam_x=screen_center_cam_x,
        screen_center_cam_y=screen_center_cam_y,
        screen_center_cam_z=screen_center_cam_z,
        screen_axis_x_x=screen_axis_x_x,
        screen_axis_x_y=screen_axis_x_y,
        screen_axis_x_z=screen_axis_x_z,
        screen_axis_y_x=screen_axis_y_x,
        screen_axis_y_y=screen_axis_y_y,
        screen_axis_y_z=screen_axis_y_z,
        screen_scale_x=screen_scale_x,
        screen_scale_y=screen_scale_y,
        screen_fit_rmse=screen_fit_rmse,
        display_width=display["width"],
        display_height=display["height"],
        origin_x=float(display["width"]) / 2.0,
        origin_y=float(display["height"]) / 2.0,
    )

    capture_step = CaptureStep(enabled=False)

    overlay_step = OverlayStep(enabled=False, show_hud=False)

    udp_forward_step = UDPForwardStep(
        host=args.udp_host,
        port=args.udp_port,
        enabled=args.udp,
    )

    frame_dispatcher = FrameDispatcher(
        args,
        calibration=calibration,
        overlay_manager=None,
        state_machine=state_machine,
        face_mesh_step=face_mesh_step,
        calibration_adapter_step=calibration_adapter_step,
        capture_step=capture_step,
        overlay_step=overlay_step,
        udp_forward_step=udp_forward_step,
    )
    camera_reader = CameraReader(
        args.camera_index,
        _backend_string_to_int(args.camera_backend),
        args.camera_fourcc,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )

    try:
        frame_dispatcher.start()
        camera_reader.open()

        if args.calibrate or args.calibration:
            frame_dispatcher.start_calibration()
            logger.info("Running calibration workflow...")
            calib_matrix, _ = frame_dispatcher.run_calibration_workflow(camera_reader)
            if auto_transition_to_udp:
                if calib_matrix is None:
                    logger.info(
                        "Calibration did not complete; UDP forwarder will not start."
                    )
                else:
                    logger.info(
                        "Calibration complete. Starting UDP forwarder on "
                        f"{args.udp_host}:{args.udp_port}."
                    )
                    frame_dispatcher.set_udp_forwarding_enabled(True)
                    frame_dispatcher.run_capture_loop(camera_reader)
        else:
            frame_dispatcher.start_operational()
            active_modes = []
            if args.capture:
                active_modes.append("capture")
            if args.overlay:
                active_modes.append("overlay")
            if args.udp:
                active_modes.append("udp")
            if not active_modes:
                active_modes.append("tracking")
            logger.info(f"Running in mode(s): {', '.join(active_modes)}")
            frame_dispatcher.run_capture_loop(camera_reader)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise
    finally:
        logger.info("Shutting down...")
        camera_reader.release()
        frame_dispatcher.stop()
        logger.info("Shutdown complete")


def _run_with_yappi():
    import yappi

    yappi.set_clock_type("wall")
    yappi.start(builtins=True)
    try:
        main()
    finally:
        yappi.stop()
        out_dir = os.getenv("FACEMESH_PROFILE_DIR", "profile_out")
        os.makedirs(out_dir, exist_ok=True)

        func_stats = yappi.get_func_stats()
        func_stats.save(os.path.join(out_dir, "yappi.pstat"), type="pstat")
        func_stats.save(os.path.join(out_dir, "yappi.callgrind"), type="callgrind")

        with open(os.path.join(out_dir, "yappi_top.txt"), "w") as f:
            func_stats.sort("tsub", "desc").print_all(
                out=f,
                columns={
                    0: ("name", 80),
                    1: ("ncall", 10),
                    2: ("tsub", 10),
                    3: ("ttot", 10),
                    4: ("tavg", 10),
                },
            )

        thread_stats = yappi.get_thread_stats()
        with open(os.path.join(out_dir, "yappi_threads.txt"), "w") as f:
            thread_stats.print_all(out=f)

        print(f"[yappi] profile written to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
