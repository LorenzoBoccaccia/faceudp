from dataclasses import dataclass
from functools import cached_property
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation

from .facemesh_dao import FaceMeshEvent, safe_float
from .gaze_primitives import (
    _calibrated_screen_geometry,
    project_head_angles_to_screen_xy,
    screen_xy_to_head_angles,
)


logger = logging.getLogger(__name__)

DEFAULT_CENTER_ZETA = 1200.0
CALIBRATION_MODEL_VERSION = 10


@dataclass(frozen=True)
class CalibrationMatrix:
    center_yaw: float = 0.0
    center_pitch: float = 0.0
    face_center_yaw: float = 0.0
    face_center_pitch: float = 0.0
    center_zeta: float = DEFAULT_CENTER_ZETA
    yaw_coefficient_positive: float = 1.0
    yaw_coefficient_negative: float = 1.0
    pitch_coefficient_positive: float = 1.0
    pitch_coefficient_negative: float = 1.0
    yaw_from_pitch_coupling: float = 0.0
    pitch_from_yaw_coupling: float = 0.0
    eye_yaw_min: float = -1.0
    eye_yaw_max: float = 1.0
    eye_pitch_min: float = -1.0
    eye_pitch_max: float = 1.0
    face_center_x: float = 0.0
    face_center_y: float = 0.0
    face_center_z: float = DEFAULT_CENTER_ZETA
    screen_center_cam_x: float = 0.0
    screen_center_cam_y: float = 0.0
    screen_center_cam_z: float = DEFAULT_CENTER_ZETA
    screen_axis_x_x: float = 1.0
    screen_axis_x_y: float = 0.0
    screen_axis_x_z: float = 0.0
    screen_axis_y_x: float = 0.0
    screen_axis_y_y: float = 1.0
    screen_axis_y_z: float = 0.0
    screen_scale_x: float = 1.0
    screen_scale_y: float = 1.0
    screen_fit_rmse: float = -1.0
    sample_count: int = 0
    timestamp_ms: int = 0


@dataclass(frozen=True)
class CalibrationPoint:
    name: str
    screen_x: float
    screen_y: float
    raw_eye_yaw: float
    raw_eye_pitch: float
    raw_left_eye_yaw: float
    raw_left_eye_pitch: float
    raw_right_eye_yaw: float
    raw_right_eye_pitch: float
    sample_count: int
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    zeta: float = DEFAULT_CENTER_ZETA
    head_x: float = 0.0
    head_y: float = 0.0
    head_z: float = DEFAULT_CENTER_ZETA
    nose_target_x: Optional[float] = None
    nose_target_y: Optional[float] = None
    eye_target_x: Optional[float] = None
    eye_target_y: Optional[float] = None


def _positive_or(v: Any, fallback: float) -> float:
    x = safe_float(v, fallback)
    return x if x > 1e-9 else fallback


def _positive_coefficient(v: Any, fallback: float) -> float:
    x = abs(safe_float(v, fallback))
    return x if math.isfinite(x) and x > 1e-9 else fallback


def _head_forward_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit vector in camera frame for a head pointing at yaw/pitch."""
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cp = math.cos(pitch)
    return np.array(
        [math.sin(yaw) * cp, -math.sin(pitch), -math.cos(yaw) * cp],
        dtype=float,
    )


def _fit_screen_geometry(
    points: List[CalibrationPoint],
    center_point: CalibrationPoint,
    px_per_mm_x: Optional[float] = None,
    px_per_mm_y: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Solve for screen pose from click-confirmed nose-aim samples.

    Each sample gives ``h_i + t_i · d_i = c + u_i · A + v_i · B`` where
    (u_i, v_i) is the red-dot pixel offset from the screen-centre pixel,
    (h_i, d_i) is the head position and forward direction, and (c, A, B) is
    the unknown screen origin with per-pixel axis vectors. An unconstrained
    lstsq gives a warm start; when physical pixel scales are known, a
    nonlinear refinement pins |A|, |B| to the OS-reported values and enforces
    A ⊥ B so the plane is a physically realizable rectangular monitor.
    """
    origin_x = safe_float(getattr(center_point, "nose_target_x", None), float("nan"))
    if not math.isfinite(origin_x):
        origin_x = safe_float(center_point.screen_x, 0.0)
    origin_y = safe_float(getattr(center_point, "nose_target_y", None), float("nan"))
    if not math.isfinite(origin_y):
        origin_y = safe_float(center_point.screen_y, 0.0)

    rows: List[Tuple[np.ndarray, float, float, float]] = []
    for point in points:
        nose_x_raw = point.nose_target_x
        nose_y_raw = point.nose_target_y
        if nose_x_raw is None or nose_y_raw is None:
            nose_x = safe_float(point.screen_x, origin_x)
            nose_y = safe_float(point.screen_y, origin_y)
        else:
            nose_x = safe_float(nose_x_raw, origin_x)
            nose_y = safe_float(nose_y_raw, origin_y)
        head_x = safe_float(getattr(point, "head_x", 0.0), 0.0)
        head_y = safe_float(getattr(point, "head_y", 0.0), 0.0)
        head_z = _positive_or(
            getattr(point, "head_z", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
        )
        head = np.array([head_x, head_y, head_z], dtype=float)
        direction = _head_forward_direction(
            safe_float(getattr(point, "head_yaw", 0.0), 0.0),
            safe_float(getattr(point, "head_pitch", 0.0), 0.0),
        )
        u = nose_x - origin_x
        v = nose_y - origin_y
        rows.append((head, direction, u, v))

    n = len(rows)
    if n < 9:
        raise ValueError(f"Screen geometry fit requires 9 samples, got {n}")

    matrix = np.zeros((3 * n, 9 + n), dtype=float)
    target = np.zeros(3 * n, dtype=float)
    for i, (head, direction, u, v) in enumerate(rows):
        for axis in range(3):
            row = 3 * i + axis
            matrix[row, axis] = 1.0
            matrix[row, 3 + axis] = u
            matrix[row, 6 + axis] = v
            matrix[row, 9 + i] = -direction[axis]
            target[row] = head[axis]

    rank = int(np.linalg.matrix_rank(matrix))
    expected_rank = matrix.shape[1]
    if rank < expected_rank:
        raise ValueError(
            f"Screen geometry fit is rank-deficient (rank={rank}, expected {expected_rank}); "
            "calibration samples do not constrain the screen plane — "
            "head must shift in Z across targets for the plane normal to be observable"
        )

    t_min = 1.0  # mm; picks the half-space where the nose ray hits screen
    lower = np.full(9 + n, -np.inf, dtype=float)
    upper = np.full(9 + n, np.inf, dtype=float)
    lower[9:] = t_min

    warm = lsq_linear(matrix, target, bounds=(lower, upper), method="bvls")
    if not warm.success:
        raise ValueError(
            f"Screen geometry fit failed to converge: {warm.message}"
        )
    solution = warm.x
    screen_center = solution[0:3]
    a_vec = solution[3:6]
    b_vec = solution[6:9]

    a_norm = float(np.linalg.norm(a_vec))
    if a_norm <= 1e-6:
        raise ValueError("Screen geometry fit produced a degenerate X axis")
    axis_x = a_vec / a_norm
    scale_x = 1.0 / a_norm

    b_ortho = b_vec - float(np.dot(b_vec, axis_x)) * axis_x
    b_ortho_norm = float(np.linalg.norm(b_ortho))
    if b_ortho_norm <= 1e-6:
        raise ValueError("Screen geometry fit produced a degenerate Y axis")
    axis_y = b_ortho / b_ortho_norm
    scale_y = 1.0 / b_ortho_norm

    if (
        px_per_mm_x is not None
        and px_per_mm_y is not None
        and px_per_mm_x > 0
        and px_per_mm_y > 0
    ):
        mm_per_px_x = 1.0 / float(px_per_mm_x)
        mm_per_px_y = 1.0 / float(px_per_mm_y)

        heads = np.stack([head for head, _d, _u, _v in rows])
        dirs = np.stack([direction for _h, direction, _u, _v in rows])
        uvs = np.array([(u, v) for _h, _d, u, v in rows])

        # Seed rotation from warm-start axes, then re-orthonormalize.
        seed_axis_x = axis_x
        seed_axis_y = axis_y - float(np.dot(axis_y, seed_axis_x)) * seed_axis_x
        seed_axis_y_norm = float(np.linalg.norm(seed_axis_y))
        if seed_axis_y_norm > 1e-6:
            seed_axis_y = seed_axis_y / seed_axis_y_norm
        else:
            seed_axis_y = np.array([0.0, 1.0, 0.0])
        seed_axis_z = np.cross(seed_axis_x, seed_axis_y)
        seed_rot = np.column_stack([seed_axis_x, seed_axis_y, seed_axis_z])
        # Orthonormalize via SVD to guarantee a valid rotation matrix.
        u_svd, _s_svd, vt_svd = np.linalg.svd(seed_rot)
        r_init = u_svd @ vt_svd
        if np.linalg.det(r_init) < 0:
            u_svd[:, -1] *= -1.0
            r_init = u_svd @ vt_svd
        rot_vec0 = Rotation.from_matrix(r_init).as_rotvec()

        t_warm = solution[9:]
        t0 = np.clip(t_warm, t_min, None)

        x0 = np.concatenate([screen_center, rot_vec0, t0])

        def unpack(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            c = params[0:3]
            rotvec = params[3:6]
            t_vals = params[6:]
            rot = Rotation.from_rotvec(rotvec).as_matrix()
            ax_x = rot[:, 0] * mm_per_px_x
            ax_y = rot[:, 1] * mm_per_px_y
            return c, ax_x, ax_y, t_vals

        def residuals(params: np.ndarray) -> np.ndarray:
            c, ax_x, ax_y, t_vals = unpack(params)
            lhs = heads + t_vals[:, None] * dirs
            rhs = c[None, :] + uvs[:, 0:1] * ax_x[None, :] + uvs[:, 1:2] * ax_y[None, :]
            return (lhs - rhs).reshape(-1)

        lb = np.concatenate(
            [np.full(6, -np.inf), np.full(n, t_min)]
        )
        ub = np.full(6 + n, np.inf)

        refined = least_squares(
            residuals,
            x0,
            bounds=(lb, ub),
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
            max_nfev=500,
        )
        if not refined.success and refined.status <= 0:
            raise ValueError(
                f"Constrained screen geometry refinement failed: {refined.message}"
            )

        c_ref, ax_x_ref, ax_y_ref, _t_ref = unpack(refined.x)
        screen_center = c_ref
        axis_x = ax_x_ref / np.linalg.norm(ax_x_ref)
        axis_y = ax_y_ref / np.linalg.norm(ax_y_ref)
        scale_x = float(px_per_mm_x)
        scale_y = float(px_per_mm_y)
        residual_vec = refined.fun
    else:
        residual_vec = matrix @ solution - target

    per_point_sq = residual_vec.reshape(n, 3)
    per_point_distance = np.sqrt(np.sum(per_point_sq * per_point_sq, axis=1))
    fit_rmse = float(np.sqrt(float(np.mean(per_point_distance * per_point_distance))))

    return screen_center, axis_x, axis_y, scale_x, scale_y, fit_rmse


def _profile_token(raw_profile: str) -> str:
    if not raw_profile:
        return "default"
    out: List[str] = []
    for ch in raw_profile:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("-")
    sanitized = "".join(out).strip("._-")
    return sanitized if sanitized else "default"


def _profile_filename(profile_token: str) -> str:
    return f"calibration-{profile_token}.json"


def _profile_load_candidates(profile_token: str) -> List[str]:
    filename = _profile_filename(profile_token)
    if profile_token != "default":
        return [filename]
    return [filename, "calibration.json"]


def _coefficient_from_samples(samples: List[float]) -> Optional[float]:
    if not samples:
        return None
    arr = np.array(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _resolve_axis_coefficients(
    positive_samples: List[float],
    negative_samples: List[float],
) -> Tuple[float, float]:
    positive = _coefficient_from_samples(positive_samples)
    negative = _coefficient_from_samples(negative_samples)
    if positive is None and negative is None:
        return 1.0, 1.0
    if positive is None:
        positive = negative
    if negative is None:
        negative = positive
    return _positive_coefficient(positive, 1.0), _positive_coefficient(negative, 1.0)


def _resolve_axis_extension(samples: List[float]) -> Tuple[float, float]:
    if not samples:
        return -1.0, 1.0
    arr = np.array(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    axis_min = float(np.min(arr))
    axis_max = float(np.max(arr))
    if abs(axis_max - axis_min) <= 1e-6:
        axis_min -= 1e-3
        axis_max += 1e-3
    return axis_min, axis_max


def _interpolate_coefficient(
    eye_delta: float,
    axis_min: float,
    axis_max: float,
    negative_coefficient: float,
    positive_coefficient: float,
) -> float:
    c_neg = _positive_coefficient(negative_coefficient, 1.0)
    c_pos = _positive_coefficient(positive_coefficient, 1.0)
    mid = 0.5 * (c_neg + c_pos)
    if eye_delta >= 0.0:
        span = axis_max
        if span <= 1e-9:
            return c_pos
        t = eye_delta / span
        if t > 1.0:
            t = 1.0
        return mid + (c_pos - mid) * t
    span = -axis_min
    if span <= 1e-9:
        return c_neg
    t = (-eye_delta) / span
    if t > 1.0:
        t = 1.0
    return mid + (c_neg - mid) * t


def _fit_linear_scalar(inputs: List[float], targets: List[float]) -> float:
    if not inputs or not targets:
        return 0.0
    if len(inputs) != len(targets):
        return 0.0
    x = np.array(inputs, dtype=float)
    y = np.array(targets, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return 0.0
    denom = float(np.dot(x, x))
    if denom <= 1e-9:
        return 0.0
    return float(np.dot(x, y) / denom)


def _target_eye_angles(
    point: CalibrationPoint,
    center_point: CalibrationPoint,
    center_zeta: float,
    screen_center_cam: np.ndarray,
    screen_axis_x: np.ndarray,
    screen_axis_y: np.ndarray,
    screen_scale_x: float,
    screen_scale_y: float,
) -> Optional[Tuple[float, float]]:
    eye_target_x = point.eye_target_x if point.eye_target_x is not None else point.screen_x
    eye_target_y = point.eye_target_y if point.eye_target_y is not None else point.screen_y
    head_x = safe_float(getattr(point, "head_x", 0.0), 0.0)
    head_y = safe_float(getattr(point, "head_y", 0.0), 0.0)
    head_z = _positive_or(getattr(point, "head_z", center_zeta), center_zeta)
    origin_x = safe_float(getattr(center_point, "nose_target_x", None), float("nan"))
    if not math.isfinite(origin_x):
        origin_x = safe_float(center_point.screen_x, 0.0)
    origin_y = safe_float(getattr(center_point, "nose_target_y", None), float("nan"))
    if not math.isfinite(origin_y):
        origin_y = safe_float(center_point.screen_y, 0.0)
    angles = screen_xy_to_head_angles(
        screen_x=eye_target_x,
        screen_y=eye_target_y,
        head_x=head_x,
        head_y=head_y,
        head_z=head_z,
        center_zeta=center_zeta,
        screen_center_cam_x=float(screen_center_cam[0]),
        screen_center_cam_y=float(screen_center_cam[1]),
        screen_center_cam_z=float(screen_center_cam[2]),
        screen_axis_x_x=float(screen_axis_x[0]),
        screen_axis_x_y=float(screen_axis_x[1]),
        screen_axis_x_z=float(screen_axis_x[2]),
        screen_axis_y_x=float(screen_axis_y[0]),
        screen_axis_y_y=float(screen_axis_y[1]),
        screen_axis_y_z=float(screen_axis_y[2]),
        screen_fit_rmse=0.0,
        screen_scale_x=screen_scale_x,
        screen_scale_y=screen_scale_y,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if angles is None:
        return None
    head_yaw = safe_float(getattr(point, "head_yaw", 0.0), 0.0)
    head_pitch = safe_float(getattr(point, "head_pitch", 0.0), 0.0)
    true_yaw, true_pitch = angles
    return true_yaw - head_yaw, true_pitch - head_pitch


def _signum(value: float, eps: float = 1e-6) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def compute_calibration_matrix(
    points: List[CalibrationPoint],
    px_per_mm_x: Optional[float] = None,
    px_per_mm_y: Optional[float] = None,
) -> CalibrationMatrix:
    if len(points) < 9:
        raise ValueError(f"Calibration requires 9 points, got {len(points)}")
    required_names = {"C", "T", "TL", "L", "BL", "B", "BR", "R", "TR"}
    available_names = {str(p.name) for p in points}
    if not required_names.issubset(available_names):
        missing = sorted(required_names - available_names)
        raise ValueError(f"Calibration points missing required targets: {missing}")

    center_point = next((p for p in points if p.name == "C"), None)
    if center_point is None:
        raise ValueError("Calibration points must include a center point named 'C'")

    center_zeta = _positive_or(
        getattr(center_point, "zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
    )
    center_head_x = safe_float(getattr(center_point, "head_x", 0.0), 0.0)
    center_head_y = safe_float(getattr(center_point, "head_y", 0.0), 0.0)
    center_head_z = _positive_or(getattr(center_point, "head_z", center_zeta), center_zeta)
    (
        screen_center_cam,
        screen_axis_x,
        screen_axis_y,
        screen_scale_x,
        screen_scale_y,
        screen_fit_rmse,
    ) = _fit_screen_geometry(
        points=points,
        center_point=center_point,
        px_per_mm_x=px_per_mm_x,
        px_per_mm_y=px_per_mm_y,
    )

    yaw_positive_samples: List[float] = []
    yaw_negative_samples: List[float] = []
    pitch_positive_samples: List[float] = []
    pitch_negative_samples: List[float] = []
    eye_yaw_samples: List[float] = []
    eye_pitch_samples: List[float] = []
    sign_issues: List[str] = []

    for point in points:
        if point.name == "C":
            continue
        eye_delta_yaw = safe_float(point.raw_eye_yaw) - safe_float(center_point.raw_eye_yaw)
        eye_delta_pitch = safe_float(point.raw_eye_pitch) - safe_float(center_point.raw_eye_pitch)
        face_delta_yaw = safe_float(getattr(point, "head_yaw", 0.0)) - safe_float(
            getattr(center_point, "head_yaw", 0.0)
        )
        face_delta_pitch = safe_float(getattr(point, "head_pitch", 0.0)) - safe_float(
            getattr(center_point, "head_pitch", 0.0)
        )
        expected_nose_x = safe_float(
            point.nose_target_x if point.nose_target_x is not None else point.screen_x,
            center_point.screen_x,
        )
        expected_nose_y = safe_float(
            point.nose_target_y if point.nose_target_y is not None else point.screen_y,
            center_point.screen_y,
        )
        expected_eye_x = safe_float(
            point.eye_target_x if point.eye_target_x is not None else point.screen_x,
            center_point.screen_x,
        )
        expected_eye_y = safe_float(
            point.eye_target_y if point.eye_target_y is not None else point.screen_y,
            center_point.screen_y,
        )
        expected_face_yaw_sign = _signum(expected_nose_x - center_point.screen_x)
        expected_face_pitch_sign = _signum(center_point.screen_y - expected_nose_y)
        expected_eye_yaw_sign = _signum(expected_eye_x - center_point.screen_x)
        expected_eye_pitch_sign = _signum(center_point.screen_y - expected_eye_y)

        eye_yaw_samples.append(eye_delta_yaw)
        eye_pitch_samples.append(eye_delta_pitch)

        target_eye = _target_eye_angles(
            point=point,
            center_point=center_point,
            center_zeta=center_zeta,
            screen_center_cam=screen_center_cam,
            screen_axis_x=screen_axis_x,
            screen_axis_y=screen_axis_y,
            screen_scale_x=screen_scale_x,
            screen_scale_y=screen_scale_y,
        )
        if target_eye is not None:
            target_eye_yaw, target_eye_pitch = target_eye
            if abs(eye_delta_yaw) > 1e-6:
                ratio_yaw = target_eye_yaw / eye_delta_yaw
                if math.isfinite(ratio_yaw) and ratio_yaw > 1e-6:
                    if eye_delta_yaw >= 0.0:
                        yaw_positive_samples.append(ratio_yaw)
                    else:
                        yaw_negative_samples.append(ratio_yaw)
            if abs(eye_delta_pitch) > 1e-6:
                ratio_pitch = target_eye_pitch / eye_delta_pitch
                if math.isfinite(ratio_pitch) and ratio_pitch > 1e-6:
                    if eye_delta_pitch >= 0.0:
                        pitch_positive_samples.append(ratio_pitch)
                    else:
                        pitch_negative_samples.append(ratio_pitch)

        if (
            expected_face_yaw_sign != 0
            and abs(face_delta_yaw) > 0.5
            and _signum(face_delta_yaw) != expected_face_yaw_sign
        ):
            sign_issues.append(f"{point.name}: face yaw sign mismatch")
        if (
            expected_face_pitch_sign != 0
            and abs(face_delta_pitch) > 0.5
            and _signum(face_delta_pitch) != expected_face_pitch_sign
        ):
            sign_issues.append(f"{point.name}: face pitch sign mismatch")
        eye_tracking_reliable = (
            abs(face_delta_yaw) < 20.0 and abs(face_delta_pitch) < 20.0
        )
        if (
            eye_tracking_reliable
            and expected_eye_yaw_sign != 0
            and abs(eye_delta_yaw) > 0.5
            and _signum(eye_delta_yaw) != expected_eye_yaw_sign
        ):
            sign_issues.append(f"{point.name}: eye yaw sign mismatch")
        if (
            eye_tracking_reliable
            and expected_eye_pitch_sign != 0
            and abs(eye_delta_pitch) > 0.5
            and _signum(eye_delta_pitch) != expected_eye_pitch_sign
        ):
            sign_issues.append(f"{point.name}: eye pitch sign mismatch")

    if sign_issues:
        deduped = sorted(set(sign_issues))
        raise ValueError(
            "Calibration sign validation failed: " + "; ".join(deduped)
        )

    yaw_positive, yaw_negative = _resolve_axis_coefficients(
        yaw_positive_samples, yaw_negative_samples
    )
    pitch_positive, pitch_negative = _resolve_axis_coefficients(
        pitch_positive_samples, pitch_negative_samples
    )
    eye_yaw_min, eye_yaw_max = _resolve_axis_extension(eye_yaw_samples)
    eye_pitch_min, eye_pitch_max = _resolve_axis_extension(eye_pitch_samples)
    yaw_cross_inputs: List[float] = []
    yaw_cross_targets: List[float] = []
    pitch_cross_inputs: List[float] = []
    pitch_cross_targets: List[float] = []
    center_head_yaw = safe_float(getattr(center_point, "head_yaw", 0.0), 0.0)
    center_head_pitch = safe_float(getattr(center_point, "head_pitch", 0.0), 0.0)
    center_eye_yaw = safe_float(center_point.raw_eye_yaw, 0.0)
    center_eye_pitch = safe_float(center_point.raw_eye_pitch, 0.0)
    for point in points:
        if point.name == "C":
            continue
        eye_delta_yaw = safe_float(point.raw_eye_yaw, 0.0) - center_eye_yaw
        eye_delta_pitch = safe_float(point.raw_eye_pitch, 0.0) - center_eye_pitch
        target_eye = _target_eye_angles(
            point=point,
            center_point=center_point,
            center_zeta=center_zeta,
            screen_center_cam=screen_center_cam,
            screen_axis_x=screen_axis_x,
            screen_axis_y=screen_axis_y,
            screen_scale_x=screen_scale_x,
            screen_scale_y=screen_scale_y,
        )
        if target_eye is None:
            continue
        target_eye_yaw, target_eye_pitch = target_eye
        yaw_coeff_point = _interpolate_coefficient(
            eye_delta=eye_delta_yaw,
            axis_min=eye_yaw_min,
            axis_max=eye_yaw_max,
            negative_coefficient=yaw_negative,
            positive_coefficient=yaw_positive,
        )
        pitch_coeff_point = _interpolate_coefficient(
            eye_delta=eye_delta_pitch,
            axis_min=eye_pitch_min,
            axis_max=eye_pitch_max,
            negative_coefficient=pitch_negative,
            positive_coefficient=pitch_positive,
        )
        yaw_cross_inputs.append(eye_delta_pitch)
        yaw_cross_targets.append(target_eye_yaw - yaw_coeff_point * eye_delta_yaw)
        pitch_cross_inputs.append(eye_delta_yaw)
        pitch_cross_targets.append(target_eye_pitch - pitch_coeff_point * eye_delta_pitch)
    yaw_from_pitch_coupling = _fit_linear_scalar(yaw_cross_inputs, yaw_cross_targets)
    pitch_from_yaw_coupling = _fit_linear_scalar(pitch_cross_inputs, pitch_cross_targets)
    total_sample_count = sum(int(p.sample_count) for p in points)

    return CalibrationMatrix(
        center_yaw=safe_float(center_point.raw_eye_yaw),
        center_pitch=safe_float(center_point.raw_eye_pitch),
        face_center_yaw=safe_float(getattr(center_point, "head_yaw", 0.0)),
        face_center_pitch=safe_float(getattr(center_point, "head_pitch", 0.0)),
        center_zeta=center_zeta,
        yaw_coefficient_positive=yaw_positive,
        yaw_coefficient_negative=yaw_negative,
        pitch_coefficient_positive=pitch_positive,
        pitch_coefficient_negative=pitch_negative,
        yaw_from_pitch_coupling=yaw_from_pitch_coupling,
        pitch_from_yaw_coupling=pitch_from_yaw_coupling,
        eye_yaw_min=eye_yaw_min,
        eye_yaw_max=eye_yaw_max,
        eye_pitch_min=eye_pitch_min,
        eye_pitch_max=eye_pitch_max,
        face_center_x=center_head_x,
        face_center_y=center_head_y,
        face_center_z=center_head_z,
        screen_center_cam_x=float(screen_center_cam[0]),
        screen_center_cam_y=float(screen_center_cam[1]),
        screen_center_cam_z=float(screen_center_cam[2]),
        screen_axis_x_x=float(screen_axis_x[0]),
        screen_axis_x_y=float(screen_axis_x[1]),
        screen_axis_x_z=float(screen_axis_x[2]),
        screen_axis_y_x=float(screen_axis_y[0]),
        screen_axis_y_y=float(screen_axis_y[1]),
        screen_axis_y_z=float(screen_axis_y[2]),
        screen_scale_x=screen_scale_x,
        screen_scale_y=screen_scale_y,
        screen_fit_rmse=screen_fit_rmse,
        sample_count=total_sample_count,
        timestamp_ms=int(time.time() * 1000),
    )


def save_calibration(
    calib: CalibrationMatrix, points: List[CalibrationPoint], profile: str = ""
) -> Path:
    profile_token = _profile_token(profile)
    filename = _profile_filename(profile_token)

    payload = {
        "timestamp": int(time.time() * 1000),
        "profile": profile_token,
        "calibration": {
            "modelVersion": int(CALIBRATION_MODEL_VERSION),
            "centerYaw": float(calib.center_yaw),
            "centerPitch": float(calib.center_pitch),
            "faceCenterYaw": float(calib.face_center_yaw),
            "faceCenterPitch": float(calib.face_center_pitch),
            "centerZeta": float(calib.center_zeta),
            "yawCoefficientPositive": float(calib.yaw_coefficient_positive),
            "yawCoefficientNegative": float(calib.yaw_coefficient_negative),
            "pitchCoefficientPositive": float(calib.pitch_coefficient_positive),
            "pitchCoefficientNegative": float(calib.pitch_coefficient_negative),
            "yawFromPitchCoupling": float(calib.yaw_from_pitch_coupling),
            "pitchFromYawCoupling": float(calib.pitch_from_yaw_coupling),
            "eyeYawMin": float(calib.eye_yaw_min),
            "eyeYawMax": float(calib.eye_yaw_max),
            "eyePitchMin": float(calib.eye_pitch_min),
            "eyePitchMax": float(calib.eye_pitch_max),
            "faceCenterX": float(calib.face_center_x),
            "faceCenterY": float(calib.face_center_y),
            "faceCenterZ": float(calib.face_center_z),
            "screenCenterCamX": float(calib.screen_center_cam_x),
            "screenCenterCamY": float(calib.screen_center_cam_y),
            "screenCenterCamZ": float(calib.screen_center_cam_z),
            "screenAxisXX": float(calib.screen_axis_x_x),
            "screenAxisXY": float(calib.screen_axis_x_y),
            "screenAxisXZ": float(calib.screen_axis_x_z),
            "screenAxisYX": float(calib.screen_axis_y_x),
            "screenAxisYY": float(calib.screen_axis_y_y),
            "screenAxisYZ": float(calib.screen_axis_y_z),
            "screenScaleX": float(calib.screen_scale_x),
            "screenScaleY": float(calib.screen_scale_y),
            "screenFitRmse": float(calib.screen_fit_rmse),
            "sampleCount": int(calib.sample_count),
        },
        "points": [
            {
                "name": str(point.name),
                "screenX": float(point.screen_x),
                "screenY": float(point.screen_y),
                "rawEyeYaw": float(point.raw_eye_yaw),
                "rawEyePitch": float(point.raw_eye_pitch),
                "rawLeftEyeYaw": float(point.raw_left_eye_yaw),
                "rawLeftEyePitch": float(point.raw_left_eye_pitch),
                "rawRightEyeYaw": float(point.raw_right_eye_yaw),
                "rawRightEyePitch": float(point.raw_right_eye_pitch),
                "headYaw": float(getattr(point, "head_yaw", 0.0)),
                "headPitch": float(getattr(point, "head_pitch", 0.0)),
                "zeta": float(getattr(point, "zeta", DEFAULT_CENTER_ZETA)),
                "headX": float(getattr(point, "head_x", 0.0)),
                "headY": float(getattr(point, "head_y", 0.0)),
                "headZ": float(getattr(point, "head_z", DEFAULT_CENTER_ZETA)),
                "noseTargetX": (
                    float(point.nose_target_x) if point.nose_target_x is not None else None
                ),
                "noseTargetY": (
                    float(point.nose_target_y) if point.nose_target_y is not None else None
                ),
                "eyeTargetX": (
                    float(point.eye_target_x) if point.eye_target_x is not None else None
                ),
                "eyeTargetY": (
                    float(point.eye_target_y) if point.eye_target_y is not None else None
                ),
                "sampleCount": int(point.sample_count),
            }
            for point in points
        ],
    }

    file_path = Path(filename)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return file_path


def load_calibration(
    profile: str = "",
) -> Tuple[CalibrationMatrix, List[CalibrationPoint]]:
    profile_token = _profile_token(profile)
    filenames = _profile_load_candidates(profile_token)
    file_path: Optional[Path] = None
    for filename in filenames:
        candidate = Path(filename)
        if candidate.exists():
            file_path = candidate
            break
    if file_path is None:
        file_path = Path(filenames[0])

    empty = CalibrationMatrix(
        center_yaw=0.0,
        center_pitch=0.0,
        face_center_yaw=0.0,
        face_center_pitch=0.0,
        center_zeta=DEFAULT_CENTER_ZETA,
        yaw_coefficient_positive=1.0,
        yaw_coefficient_negative=1.0,
        pitch_coefficient_positive=1.0,
        pitch_coefficient_negative=1.0,
        yaw_from_pitch_coupling=0.0,
        pitch_from_yaw_coupling=0.0,
        eye_yaw_min=-1.0,
        eye_yaw_max=1.0,
        eye_pitch_min=-1.0,
        eye_pitch_max=1.0,
        face_center_x=0.0,
        face_center_y=0.0,
        face_center_z=DEFAULT_CENTER_ZETA,
        screen_center_cam_x=0.0,
        screen_center_cam_y=0.0,
        screen_center_cam_z=DEFAULT_CENTER_ZETA,
        screen_axis_x_x=1.0,
        screen_axis_x_y=0.0,
        screen_axis_x_z=0.0,
        screen_axis_y_x=0.0,
        screen_axis_y_y=1.0,
        screen_axis_y_z=0.0,
        screen_scale_x=1.0,
        screen_scale_y=1.0,
        screen_fit_rmse=-1.0,
        sample_count=0,
        timestamp_ms=int(time.time() * 1000),
    )

    if not file_path.exists():
        return empty, []

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning(f"Failed to load calibration from {file_path.name}: {exc}")
        return empty, []

    calib_data = data.get("calibration", {})
    model_version = int(safe_float(calib_data.get("modelVersion", 1), 1))
    if model_version != CALIBRATION_MODEL_VERSION:
        logger.info(
            f"Calibration file {file_path.name} has model version {model_version}, expected {CALIBRATION_MODEL_VERSION}. Discarding."
        )
        return empty, []

    calib = CalibrationMatrix(
        center_yaw=safe_float(calib_data.get("centerYaw", 0.0)),
        center_pitch=safe_float(calib_data.get("centerPitch", 0.0)),
        face_center_yaw=safe_float(calib_data.get("faceCenterYaw", 0.0)),
        face_center_pitch=safe_float(calib_data.get("faceCenterPitch", 0.0)),
        center_zeta=_positive_or(
            calib_data.get("centerZeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
        ),
        yaw_coefficient_positive=_positive_coefficient(
            calib_data.get("yawCoefficientPositive", 1.0), 1.0
        ),
        yaw_coefficient_negative=_positive_coefficient(
            calib_data.get("yawCoefficientNegative", 1.0), 1.0
        ),
        pitch_coefficient_positive=_positive_coefficient(
            calib_data.get("pitchCoefficientPositive", 1.0), 1.0
        ),
        pitch_coefficient_negative=_positive_coefficient(
            calib_data.get("pitchCoefficientNegative", 1.0), 1.0
        ),
        yaw_from_pitch_coupling=safe_float(calib_data.get("yawFromPitchCoupling", 0.0), 0.0),
        pitch_from_yaw_coupling=safe_float(calib_data.get("pitchFromYawCoupling", 0.0), 0.0),
        eye_yaw_min=safe_float(calib_data.get("eyeYawMin", -1.0)),
        eye_yaw_max=safe_float(calib_data.get("eyeYawMax", 1.0)),
        eye_pitch_min=safe_float(calib_data.get("eyePitchMin", -1.0)),
        eye_pitch_max=safe_float(calib_data.get("eyePitchMax", 1.0)),
        face_center_x=safe_float(calib_data.get("faceCenterX", 0.0)),
        face_center_y=safe_float(calib_data.get("faceCenterY", 0.0)),
        face_center_z=_positive_or(
            calib_data.get("faceCenterZ", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
        ),
        screen_center_cam_x=safe_float(calib_data.get("screenCenterCamX", 0.0)),
        screen_center_cam_y=safe_float(calib_data.get("screenCenterCamY", 0.0)),
        screen_center_cam_z=safe_float(
            calib_data.get("screenCenterCamZ", DEFAULT_CENTER_ZETA),
            DEFAULT_CENTER_ZETA,
        ),
        screen_axis_x_x=safe_float(calib_data.get("screenAxisXX", 1.0)),
        screen_axis_x_y=safe_float(calib_data.get("screenAxisXY", 0.0)),
        screen_axis_x_z=safe_float(calib_data.get("screenAxisXZ", 0.0)),
        screen_axis_y_x=safe_float(calib_data.get("screenAxisYX", 0.0)),
        screen_axis_y_y=safe_float(calib_data.get("screenAxisYY", 1.0)),
        screen_axis_y_z=safe_float(calib_data.get("screenAxisYZ", 0.0)),
        screen_scale_x=_positive_or(calib_data.get("screenScaleX", 1.0), 1.0),
        screen_scale_y=_positive_or(calib_data.get("screenScaleY", 1.0), 1.0),
        screen_fit_rmse=safe_float(calib_data.get("screenFitRmse", -1.0)),
        sample_count=int(calib_data.get("sampleCount", 0)),
        timestamp_ms=int(data.get("timestamp", time.time() * 1000)),
    )

    points: List[CalibrationPoint] = []
    for point_data in data.get("points", []):
        points.append(
            CalibrationPoint(
                name=str(point_data.get("name", "")),
                screen_x=safe_float(point_data.get("screenX", 0.0)),
                screen_y=safe_float(point_data.get("screenY", 0.0)),
                raw_eye_yaw=safe_float(point_data.get("rawEyeYaw", 0.0)),
                raw_eye_pitch=safe_float(point_data.get("rawEyePitch", 0.0)),
                raw_left_eye_yaw=safe_float(point_data.get("rawLeftEyeYaw", 0.0)),
                raw_left_eye_pitch=safe_float(point_data.get("rawLeftEyePitch", 0.0)),
                raw_right_eye_yaw=safe_float(point_data.get("rawRightEyeYaw", 0.0)),
                raw_right_eye_pitch=safe_float(point_data.get("rawRightEyePitch", 0.0)),
                sample_count=int(point_data.get("sampleCount", 0)),
                head_yaw=safe_float(point_data.get("headYaw", 0.0)),
                head_pitch=safe_float(point_data.get("headPitch", 0.0)),
                zeta=_positive_or(
                    point_data.get("zeta", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
                ),
                head_x=safe_float(point_data.get("headX", 0.0)),
                head_y=safe_float(point_data.get("headY", 0.0)),
                head_z=_positive_or(
                    point_data.get("headZ", DEFAULT_CENTER_ZETA), DEFAULT_CENTER_ZETA
                ),
                nose_target_x=safe_float(point_data.get("noseTargetX"), float("nan"))
                if point_data.get("noseTargetX") is not None
                else None,
                nose_target_y=safe_float(point_data.get("noseTargetY"), float("nan"))
                if point_data.get("noseTargetY") is not None
                else None,
                eye_target_x=safe_float(point_data.get("eyeTargetX"), float("nan"))
                if point_data.get("eyeTargetX") is not None
                else None,
                eye_target_y=safe_float(point_data.get("eyeTargetY"), float("nan"))
                if point_data.get("eyeTargetY") is not None
                else None,
            )
        )

    return calib, points


def apply_calibration_model(
    raw_eye_yaw: Optional[float],
    raw_eye_pitch: Optional[float],
    head_yaw: Optional[float],
    head_pitch: Optional[float],
    *,
    head_x: Optional[float] = None,
    head_y: Optional[float] = None,
    head_z: Optional[float] = None,
    center_eye_yaw: float = 0.0,
    center_eye_pitch: float = 0.0,
    face_center_yaw: float = 0.0,
    face_center_pitch: float = 0.0,
    yaw_coefficient_positive: float = 1.0,
    yaw_coefficient_negative: float = 1.0,
    pitch_coefficient_positive: float = 1.0,
    pitch_coefficient_negative: float = 1.0,
    yaw_from_pitch_coupling: float = 0.0,
    pitch_from_yaw_coupling: float = 0.0,
    eye_yaw_min: float = -1.0,
    eye_yaw_max: float = 1.0,
    eye_pitch_min: float = -1.0,
    eye_pitch_max: float = 1.0,
    center_zeta: float = DEFAULT_CENTER_ZETA,
    face_center_x: float = 0.0,
    face_center_y: float = 0.0,
    face_center_z: float = DEFAULT_CENTER_ZETA,
    screen_center_cam_x: float = 0.0,
    screen_center_cam_y: float = 0.0,
    screen_center_cam_z: float = DEFAULT_CENTER_ZETA,
    screen_axis_x_x: float = 1.0,
    screen_axis_x_y: float = 0.0,
    screen_axis_x_z: float = 0.0,
    screen_axis_y_x: float = 0.0,
    screen_axis_y_y: float = 1.0,
    screen_axis_y_z: float = 0.0,
    screen_scale_x: float = 1.0,
    screen_scale_y: float = 1.0,
    screen_fit_rmse: float = -1.0,
    origin_x: Optional[float] = None,
    origin_y: Optional[float] = None,
) -> Dict[str, Any]:
    if raw_eye_yaw is None or raw_eye_pitch is None:
        raise ValueError("Raw eye angles are required")
    if head_yaw is None or head_pitch is None:
        raise ValueError("Head angles are required")
    if head_x is None or head_y is None or head_z is None:
        raise ValueError("Head position is required")

    center_eye_yaw_value = safe_float(center_eye_yaw, 0.0)
    center_eye_pitch_value = safe_float(center_eye_pitch, 0.0)
    face_center_yaw_value = safe_float(face_center_yaw, 0.0)
    face_center_pitch_value = safe_float(face_center_pitch, 0.0)

    raw_eye_yaw_value = safe_float(raw_eye_yaw, float("nan"))
    raw_eye_pitch_value = safe_float(raw_eye_pitch, float("nan"))
    head_yaw_value = safe_float(head_yaw, float("nan"))
    head_pitch_value = safe_float(head_pitch, float("nan"))
    head_x_value = safe_float(head_x, float("nan"))
    head_y_value = safe_float(head_y, float("nan"))
    head_z_value = safe_float(head_z, float("nan"))

    if not (
        math.isfinite(raw_eye_yaw_value)
        and math.isfinite(raw_eye_pitch_value)
        and math.isfinite(head_yaw_value)
        and math.isfinite(head_pitch_value)
        and math.isfinite(head_x_value)
        and math.isfinite(head_y_value)
        and math.isfinite(head_z_value)
    ):
        raise ValueError("Calibration input values must be finite")
    if head_z_value <= 1e-9:
        raise ValueError("Head Z position must be positive")

    _, screen_axis_x, screen_axis_y, screen_normal = _calibrated_screen_geometry(
        safe_float(screen_center_cam_x, 0.0),
        safe_float(screen_center_cam_y, 0.0),
        safe_float(screen_center_cam_z, _positive_or(center_zeta, DEFAULT_CENTER_ZETA)),
        safe_float(screen_axis_x_x, 1.0),
        safe_float(screen_axis_x_y, 0.0),
        safe_float(screen_axis_x_z, 0.0),
        safe_float(screen_axis_y_x, 0.0),
        safe_float(screen_axis_y_y, 1.0),
        safe_float(screen_axis_y_z, 0.0),
    )

    head_origin = np.array([head_x_value, head_y_value, head_z_value], dtype=float)
    face_origin = np.array(
        [
            safe_float(face_center_x, 0.0),
            safe_float(face_center_y, 0.0),
            _positive_or(face_center_z, DEFAULT_CENTER_ZETA),
        ],
        dtype=float,
    )
    head_delta = head_origin - face_origin
    head_ref_x = float(np.dot(head_delta, screen_axis_x))
    head_ref_y = float(np.dot(head_delta, screen_axis_y))
    head_ref_z = head_z_value

    uncalibrated_mode = safe_float(screen_fit_rmse, -1.0) < 0.0
    if uncalibrated_mode:
        eye_delta_yaw = raw_eye_yaw_value
        eye_delta_pitch = raw_eye_pitch_value
        face_delta_yaw = head_yaw_value
        face_delta_pitch = head_pitch_value
        corrected_eye_yaw = raw_eye_yaw_value
        corrected_eye_pitch = raw_eye_pitch_value
        applied_yaw_coefficient = 1.0
        applied_pitch_coefficient = 1.0
        applied_yaw_from_pitch_coupling = 0.0
        applied_pitch_from_yaw_coupling = 0.0
        corrected_yaw_linear = head_yaw_value + raw_eye_yaw_value
        corrected_pitch_linear = head_pitch_value + raw_eye_pitch_value
    else:
        eye_delta_yaw = raw_eye_yaw_value - center_eye_yaw_value
        eye_delta_pitch = raw_eye_pitch_value - center_eye_pitch_value
        face_delta_yaw = head_yaw_value - face_center_yaw_value
        face_delta_pitch = head_pitch_value - face_center_pitch_value
        applied_yaw_coefficient = _interpolate_coefficient(
            eye_delta=eye_delta_yaw,
            axis_min=safe_float(eye_yaw_min, -1.0),
            axis_max=safe_float(eye_yaw_max, 1.0),
            negative_coefficient=yaw_coefficient_negative,
            positive_coefficient=yaw_coefficient_positive,
        )
        applied_pitch_coefficient = _interpolate_coefficient(
            eye_delta=eye_delta_pitch,
            axis_min=safe_float(eye_pitch_min, -1.0),
            axis_max=safe_float(eye_pitch_max, 1.0),
            negative_coefficient=pitch_coefficient_negative,
            positive_coefficient=pitch_coefficient_positive,
        )
        applied_yaw_from_pitch_coupling = safe_float(yaw_from_pitch_coupling, 0.0)
        applied_pitch_from_yaw_coupling = safe_float(pitch_from_yaw_coupling, 0.0)
        corrected_eye_yaw = (
            eye_delta_yaw * applied_yaw_coefficient
            + eye_delta_pitch * applied_yaw_from_pitch_coupling
        )
        corrected_eye_pitch = (
            eye_delta_pitch * applied_pitch_coefficient
            + eye_delta_yaw * applied_pitch_from_yaw_coupling
        )
        corrected_yaw_linear = face_delta_yaw + corrected_eye_yaw
        corrected_pitch_linear = face_delta_pitch + corrected_eye_pitch

    corrected_yaw = corrected_yaw_linear
    corrected_pitch = corrected_pitch_linear
    corrected_screen_x = None
    corrected_screen_y = None
    screen_offset_x = None
    screen_offset_y = None
    screen_projection_t = None
    screen_depth = _positive_or(head_z_value, DEFAULT_CENTER_ZETA)
    absolute_gaze_yaw = head_yaw_value + corrected_eye_yaw
    absolute_gaze_pitch = head_pitch_value + corrected_eye_pitch
    projected = project_head_angles_to_screen_xy(
        yaw_deg=absolute_gaze_yaw,
        pitch_deg=absolute_gaze_pitch,
        head_x=head_x_value,
        head_y=head_y_value,
        head_z=head_z_value,
        center_zeta=center_zeta,
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
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if projected is not None:
        corrected_screen_x = float(projected["screen_x"])
        corrected_screen_y = float(projected["screen_y"])
        screen_offset_x = float(projected["offset_x"])
        screen_offset_y = float(projected["offset_y"])
        screen_projection_t = float(projected["projection_t"])

    return {
        "raw_eye_yaw": raw_eye_yaw_value,
        "raw_eye_pitch": raw_eye_pitch_value,
        "head_yaw": head_yaw_value,
        "head_pitch": head_pitch_value,
        "head_x": head_x_value,
        "head_y": head_y_value,
        "head_z": head_z_value,
        "eye_delta_yaw": eye_delta_yaw,
        "eye_delta_pitch": eye_delta_pitch,
        "face_delta_yaw": face_delta_yaw,
        "face_delta_pitch": face_delta_pitch,
        "corrected_eye_yaw": corrected_eye_yaw,
        "corrected_eye_pitch": corrected_eye_pitch,
        "corrected_yaw_linear": corrected_yaw_linear,
        "corrected_pitch_linear": corrected_pitch_linear,
        "corrected_yaw": corrected_yaw,
        "corrected_pitch": corrected_pitch,
        "applied_yaw_coefficient": applied_yaw_coefficient,
        "applied_pitch_coefficient": applied_pitch_coefficient,
        "applied_yaw_from_pitch_coupling": applied_yaw_from_pitch_coupling,
        "applied_pitch_from_yaw_coupling": applied_pitch_from_yaw_coupling,
        "corrected_screen_x": corrected_screen_x,
        "corrected_screen_y": corrected_screen_y,
        "screen_offset_x": screen_offset_x,
        "screen_offset_y": screen_offset_y,
        "screen_projection_t": screen_projection_t,
        "screen_depth": screen_depth,
        "screen_scale_x": _positive_or(screen_scale_x, 1.0),
        "screen_scale_y": _positive_or(screen_scale_y, 1.0),
        "head_ref_x": head_ref_x,
        "head_ref_y": head_ref_y,
        "head_ref_z": head_ref_z,
        "screen_normal_x": float(screen_normal[0]),
        "screen_normal_y": float(screen_normal[1]),
        "screen_normal_z": float(screen_normal[2]),
    }


@dataclass(frozen=True)
class CalibratedFaceAndGazeEvent:
    face_mesh_event: FaceMeshEvent
    pitch_calibration: float
    yaw_calibration: float
    roll_calibration: float
    face_center_yaw: float = 0.0
    face_center_pitch: float = 0.0
    center_zeta: float = DEFAULT_CENTER_ZETA
    yaw_coefficient_positive: float = 1.0
    yaw_coefficient_negative: float = 1.0
    pitch_coefficient_positive: float = 1.0
    pitch_coefficient_negative: float = 1.0
    yaw_from_pitch_coupling: float = 0.0
    pitch_from_yaw_coupling: float = 0.0
    eye_yaw_min: float = -1.0
    eye_yaw_max: float = 1.0
    eye_pitch_min: float = -1.0
    eye_pitch_max: float = 1.0
    face_center_x: float = 0.0
    face_center_y: float = 0.0
    face_center_z: float = DEFAULT_CENTER_ZETA
    screen_center_cam_x: float = 0.0
    screen_center_cam_y: float = 0.0
    screen_center_cam_z: float = DEFAULT_CENTER_ZETA
    screen_axis_x_x: float = 1.0
    screen_axis_x_y: float = 0.0
    screen_axis_x_z: float = 0.0
    screen_axis_y_x: float = 0.0
    screen_axis_y_y: float = 1.0
    screen_axis_y_z: float = 0.0
    screen_scale_x: float = 1.0
    screen_scale_y: float = 1.0
    screen_fit_rmse: float = -1.0
    display_width: int = 1920
    display_height: int = 1080
    origin_x: float = 960.0
    origin_y: float = 540.0

    @property
    def raw_eye_yaw(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.combined_eye_gaze_yaw

    @property
    def raw_eye_pitch(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.combined_eye_gaze_pitch

    @property
    def head_yaw(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.head_yaw

    @property
    def head_pitch(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.head_pitch

    @property
    def head_x(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.camera_x

    @property
    def head_y(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.camera_y

    @property
    def head_z(self) -> Optional[float]:
        if self.face_mesh_event is None:
            return None
        return self.face_mesh_event.camera_z

    @cached_property
    def calibrated_components(self) -> Dict[str, Any]:
        return apply_calibration_model(
            raw_eye_yaw=self.raw_eye_yaw,
            raw_eye_pitch=self.raw_eye_pitch,
            head_yaw=self.head_yaw,
            head_pitch=self.head_pitch,
            head_x=self.head_x,
            head_y=self.head_y,
            head_z=self.head_z,
            center_eye_yaw=self.yaw_calibration,
            center_eye_pitch=self.pitch_calibration,
            face_center_yaw=self.face_center_yaw,
            face_center_pitch=self.face_center_pitch,
            yaw_coefficient_positive=self.yaw_coefficient_positive,
            yaw_coefficient_negative=self.yaw_coefficient_negative,
            pitch_coefficient_positive=self.pitch_coefficient_positive,
            pitch_coefficient_negative=self.pitch_coefficient_negative,
            yaw_from_pitch_coupling=self.yaw_from_pitch_coupling,
            pitch_from_yaw_coupling=self.pitch_from_yaw_coupling,
            eye_yaw_min=self.eye_yaw_min,
            eye_yaw_max=self.eye_yaw_max,
            eye_pitch_min=self.eye_pitch_min,
            eye_pitch_max=self.eye_pitch_max,
            center_zeta=self.center_zeta,
            face_center_x=self.face_center_x,
            face_center_y=self.face_center_y,
            face_center_z=self.face_center_z,
            screen_center_cam_x=self.screen_center_cam_x,
            screen_center_cam_y=self.screen_center_cam_y,
            screen_center_cam_z=self.screen_center_cam_z,
            screen_axis_x_x=self.screen_axis_x_x,
            screen_axis_x_y=self.screen_axis_x_y,
            screen_axis_x_z=self.screen_axis_x_z,
            screen_axis_y_x=self.screen_axis_y_x,
            screen_axis_y_y=self.screen_axis_y_y,
            screen_axis_y_z=self.screen_axis_y_z,
            screen_scale_x=self.screen_scale_x,
            screen_scale_y=self.screen_scale_y,
            screen_fit_rmse=self.screen_fit_rmse,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )

    @property
    def face_delta_yaw(self) -> float:
        return self.calibrated_components["face_delta_yaw"]

    @property
    def face_delta_pitch(self) -> float:
        return self.calibrated_components["face_delta_pitch"]

    @property
    def corrected_eye_yaw(self) -> float:
        return self.calibrated_components["corrected_eye_yaw"]

    @property
    def corrected_eye_pitch(self) -> float:
        return self.calibrated_components["corrected_eye_pitch"]

    @property
    def corrected_yaw(self) -> float:
        return self.calibrated_components["corrected_yaw"]

    @property
    def corrected_pitch(self) -> float:
        return self.calibrated_components["corrected_pitch"]

    @property
    def corrected_screen_x(self) -> Optional[float]:
        return self.calibrated_components.get("corrected_screen_x")

    @property
    def corrected_screen_y(self) -> Optional[float]:
        return self.calibrated_components.get("corrected_screen_y")

    @property
    def corrected_yaw_linear(self) -> float:
        return self.calibrated_components["corrected_yaw_linear"]

    @property
    def corrected_pitch_linear(self) -> float:
        return self.calibrated_components["corrected_pitch_linear"]

    @property
    def head_ref_x(self) -> float:
        return self.calibrated_components["head_ref_x"]

    @property
    def head_ref_y(self) -> float:
        return self.calibrated_components["head_ref_y"]

    @property
    def head_ref_z(self) -> float:
        return self.calibrated_components["head_ref_z"]
