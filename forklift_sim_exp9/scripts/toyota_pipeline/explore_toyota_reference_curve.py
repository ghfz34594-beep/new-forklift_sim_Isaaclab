#!/usr/bin/env python3
"""Explore Toyota-paper-style reference curves without Isaac or third-party deps.

The Toyota forklift paper only says the fixed reward reference trajectory is
"based on the approximation of a clothoid curve".  This script makes that idea
concrete for our near-field pallet approach task:

- one curve is generated at reset from the episode start pose to the pallet
- the curve is fixed afterwards
- rewards can query distance to the curve and heading error to its tangent

Outputs:
- manifest.json with per-case metrics
- summary.md with a compact recommendation
- SVG overlays and selected per-case drawings
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Iterable


PAPER_URL = "https://arxiv.org/abs/2412.11503"
FORK_CENTER_BACKOFF_M = 0.6


@dataclass(frozen=True)
class Pose2:
    x: float
    y: float
    yaw: float


@dataclass
class CurvePoint:
    x: float
    y: float
    yaw: float
    phase: str


@dataclass
class CurveMetrics:
    case_id: str
    model: str
    length_m: float
    max_abs_lateral_m: float
    max_abs_curvature_1pm: float
    curvature_start_1pm: float
    curvature_end_1pm: float
    heading_change_deg: float
    min_forward_step_m: float
    start_pos_err_m: float
    end_pos_err_m: float
    start_yaw_err_deg: float
    end_yaw_err_deg: float
    curvature_limit_1pm: float
    feasible: bool
    terminal_corridor_ok: bool
    training_candidate: bool
    failure_reasons: list[str]
    endpoint_ok: bool
    monotone_forward: bool
    notes: str


@dataclass
class CasePayload:
    case_id: str
    start: Pose2
    goal: Pose2
    pre: tuple[float, float]
    curves: dict[str, list[CurvePoint]]
    metrics: dict[str, CurveMetrics]


def load_module_from_path(module_name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def failed_metric(
    *,
    case_id: str,
    model: str,
    curvature_limit_1pm: float,
    reason: str,
    notes: str,
) -> CurveMetrics:
    return CurveMetrics(
        case_id=case_id,
        model=model,
        length_m=0.0,
        max_abs_lateral_m=0.0,
        max_abs_curvature_1pm=1.0e9,
        curvature_start_1pm=0.0,
        curvature_end_1pm=0.0,
        heading_change_deg=0.0,
        min_forward_step_m=0.0,
        start_pos_err_m=1.0e9,
        end_pos_err_m=1.0e9,
        start_yaw_err_deg=1.0e9,
        end_yaw_err_deg=1.0e9,
        curvature_limit_1pm=curvature_limit_1pm,
        feasible=False,
        terminal_corridor_ok=False,
        training_candidate=False,
        failure_reasons=[reason],
        endpoint_ok=False,
        monotone_forward=False,
        notes=notes,
    )


def wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def unit(yaw: float) -> tuple[float, float]:
    return math.cos(yaw), math.sin(yaw)


def left_unit(yaw: float) -> tuple[float, float]:
    return -math.sin(yaw), math.cos(yaw)


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def mul(a: tuple[float, float], s: float) -> tuple[float, float]:
    return a[0] * s, a[1] * s


def norm(a: tuple[float, float]) -> float:
    return math.hypot(a[0], a[1])


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def transform_to_pallet(
    xy: tuple[float, float],
    *,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
) -> tuple[float, float]:
    u = unit(pallet_yaw)
    v = left_unit(pallet_yaw)
    rel = sub(xy, pallet_xy)
    return dot(rel, u), dot(rel, v)


def transform_from_pallet(
    s: float,
    lat: float,
    *,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
) -> tuple[float, float]:
    u = unit(pallet_yaw)
    v = left_unit(pallet_yaw)
    return add(pallet_xy, add(mul(u, s), mul(v, lat)))


def poly3_coeffs(x0: float, y0: float, slope0: float, x1: float) -> tuple[float, float, float, float]:
    """Cubic y(q)=a q^3+b q^2+c q+d from q=0 to q=L.

    Boundary conditions:
    y(0)=y0, y'(0)=slope0, y(L)=0, y'(L)=0.
    For small slopes, y'' is linear, so curvature is approximately linear:
    this is the minimum "clothoid approximation" interpretation.
    """
    length = max(x1 - x0, 1e-6)
    d = y0
    c = slope0
    a = (slope0 * length + 2.0 * y0) / (length**3)
    b = (-2.0 * slope0 * length - 3.0 * y0) / (length**2)
    return a, b, c, d


def eval_poly3(q: float, coeffs: tuple[float, float, float, float]) -> tuple[float, float, float]:
    a, b, c, d = coeffs
    y = a * q**3 + b * q**2 + c * q + d
    dy = 3.0 * a * q**2 + 2.0 * b * q + c
    ddy = 6.0 * a * q + 2.0 * b
    return y, dy, ddy


def quintic_g2_coeffs(x0: float, y0: float, slope0: float, x1: float) -> tuple[float, float, float, float, float, float]:
    """Quintic y(t), t in [0,1], with zero curvature proxy at both ends.

    Boundary conditions:
    y(0)=y0, dy/dx(0)=slope0, d2y/dx2(0)=0,
    y(1)=0, dy/dx(1)=0, d2y/dx2(1)=0.
    This is a smoother G2-ish lane-change candidate, not a true clothoid.
    """
    length = max(x1 - x0, 1e-6)
    m0 = slope0 * length
    # y(t)=c0+c1*t+c2*t^2+c3*t^3+c4*t^4+c5*t^5
    c0 = y0
    c1 = m0
    c2 = 0.0
    a_sum = -y0 - m0
    b_sum = -m0
    c3 = 10.0 * a_sum - 4.0 * b_sum
    c4 = 7.0 * b_sum - 15.0 * a_sum
    c5 = 6.0 * a_sum - 3.0 * b_sum
    return c0, c1, c2, c3, c4, c5


def eval_quintic_g2(t: float, coeffs: tuple[float, float, float, float, float, float], length: float) -> tuple[float, float, float]:
    c0, c1, c2, c3, c4, c5 = coeffs
    y = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    dy_dt = c1 + 2.0 * c2 * t + 3.0 * c3 * t**2 + 4.0 * c4 * t**3 + 5.0 * c5 * t**4
    ddy_dt2 = 2.0 * c2 + 6.0 * c3 * t + 12.0 * c4 * t**2 + 20.0 * c5 * t**3
    dy_dx = dy_dt / max(length, 1e-6)
    ddy_dx2 = ddy_dt2 / max(length * length, 1e-6)
    return y, dy_dx, ddy_dx2


def append_point_local(
    out: list[CurvePoint],
    s: float,
    lat: float,
    yaw_rel: float,
    phase: str,
    *,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
    skip_duplicate: bool = True,
) -> None:
    x, y = transform_from_pallet(s, lat, pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    yaw = wrap_angle(yaw_rel + pallet_yaw)
    if skip_duplicate and out:
        prev = out[-1]
        if math.hypot(prev.x - x, prev.y - y) < 1e-9:
            out[-1] = CurvePoint(x=x, y=y, yaw=yaw, phase=phase)
            return
    out.append(CurvePoint(x=x, y=y, yaw=yaw, phase=phase))


def build_axis_poly_curve(
    *,
    start: Pose2,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
    goal_s: float,
    pre_dist_m: float,
    curve_min_span_m: float,
    final_straight_min_m: float,
    initial_straight_m: float,
    num_samples: int,
    model: str,
) -> tuple[list[CurvePoint], tuple[float, float]]:
    start_s, start_y = transform_to_pallet((start.x, start.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    yaw_rel0 = wrap_angle(start.yaw - pallet_yaw)
    slope0 = math.tan(yaw_rel0)

    pre_nominal_s = goal_s - pre_dist_m
    pre_s = max(pre_nominal_s, start_s + curve_min_span_m)
    pre_s = min(pre_s, goal_s - final_straight_min_m)
    pre_s = max(pre_s, start_s + 1e-4)

    out: list[CurvePoint] = []
    append_point_local(
        out,
        start_s,
        start_y,
        yaw_rel0,
        "start",
        pallet_xy=pallet_xy,
        pallet_yaw=pallet_yaw,
        skip_duplicate=False,
    )

    curve_start_s = start_s
    curve_start_y = start_y
    curve_start_yaw = yaw_rel0
    if initial_straight_m > 1e-6:
        max_line = max(0.0, 0.45 * (pre_s - start_s))
        line_len = min(initial_straight_m, max_line)
        curve_start_s = start_s + line_len * math.cos(yaw_rel0)
        curve_start_y = start_y + line_len * math.sin(yaw_rel0)
        if curve_start_s > pre_s - curve_min_span_m * 0.25:
            curve_start_s = start_s
            curve_start_y = start_y
            line_len = 0.0
        line_steps = max(1, int(num_samples * min(line_len / max(goal_s - start_s, 1e-6), 0.2))) if line_len > 0 else 0
        for i in range(1, line_steps + 1):
            t = i / max(line_steps, 1)
            s = lerp(start_s, curve_start_s, t)
            lat = lerp(start_y, curve_start_y, t)
            append_point_local(out, s, lat, yaw_rel0, "initial_straight", pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)

    curve_span = max(pre_s - curve_start_s, 1e-6)
    curve_steps = max(2, int(num_samples * 0.70))
    if model == "poly3":
        coeffs3 = poly3_coeffs(curve_start_s, curve_start_y, math.tan(curve_start_yaw), pre_s)
        for i in range(1, curve_steps + 1):
            t = i / curve_steps
            q = curve_span * t
            y_val, dy, _ = eval_poly3(q, coeffs3)
            append_point_local(
                out,
                curve_start_s + q,
                y_val,
                math.atan(dy),
                "cubic_transition",
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
            )
    elif model == "g2_quintic":
        coeffs5 = quintic_g2_coeffs(curve_start_s, curve_start_y, math.tan(curve_start_yaw), pre_s)
        for i in range(1, curve_steps + 1):
            t = i / curve_steps
            y_val, dy, _ = eval_quintic_g2(t, coeffs5, curve_span)
            append_point_local(
                out,
                curve_start_s + curve_span * t,
                y_val,
                math.atan(dy),
                "g2_transition",
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
            )
    else:
        raise ValueError(f"unsupported axis-poly model: {model}")

    line_steps = max(2, num_samples - len(out) + 1)
    for i in range(1, line_steps + 1):
        t = i / line_steps
        s = lerp(pre_s, goal_s, t)
        append_point_local(out, s, 0.0, 0.0, "final_straight", pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)

    return out, transform_from_pallet(pre_s, 0.0, pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)


def build_hermite_direct_curve(
    *,
    start: Pose2,
    goal: Pose2,
    num_samples: int,
    tangent_gain: float,
) -> list[CurvePoint]:
    t0 = unit(start.yaw)
    t1 = unit(goal.yaw)
    dist = math.hypot(goal.x - start.x, goal.y - start.y)
    length = max(dist * tangent_gain, 1e-6)
    m0 = mul(t0, length)
    m1 = mul(t1, length)
    out: list[CurvePoint] = []
    for i in range(num_samples):
        t = i / max(num_samples - 1, 1)
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        x = h00 * start.x + h10 * m0[0] + h01 * goal.x + h11 * m1[0]
        y = h00 * start.y + h10 * m0[1] + h01 * goal.y + h11 * m1[1]

        dh00 = 6.0 * t2 - 6.0 * t
        dh10 = 3.0 * t2 - 4.0 * t + 1.0
        dh01 = -6.0 * t2 + 6.0 * t
        dh11 = 3.0 * t2 - 2.0 * t
        dx = dh00 * start.x + dh10 * m0[0] + dh01 * goal.x + dh11 * m1[0]
        dy = dh00 * start.y + dh10 * m0[1] + dh01 * goal.y + dh11 * m1[1]
        yaw = math.atan2(dy, dx)
        out.append(CurvePoint(x=x, y=y, yaw=yaw, phase="direct_hermite"))
    return out


def build_single_arc_pre_align_curve(
    *,
    start: Pose2,
    goal: Pose2,
    num_samples: int,
) -> tuple[list[CurvePoint] | None, str | None]:
    """Single circular-arc pose connector.

    A single arc can only satisfy both endpoint pose and tangent for special
    geometries. Returning a failed candidate is useful for this ablation because
    it separates "simple arc is not expressive enough" from PPO behavior.
    """
    dx = goal.x - start.x
    dy = goal.y - start.y
    chord_len = math.hypot(dx, dy)
    yaw_delta = wrap_angle(goal.yaw - start.yaw)

    if chord_len < 1e-9:
        return None, "single arc failed: start and goal positions coincide"

    if abs(yaw_delta) < 1e-6:
        fwd = unit(start.yaw)
        lateral = abs(dx * (-fwd[1]) + dy * fwd[0])
        forward = dx * fwd[0] + dy * fwd[1]
        if lateral <= 1e-5 and forward > 0.0:
            out = []
            for i in range(num_samples):
                t = i / max(num_samples - 1, 1)
                out.append(
                    CurvePoint(
                        x=lerp(start.x, goal.x, t),
                        y=lerp(start.y, goal.y, t),
                        yaw=start.yaw,
                        phase="single_arc_degenerate_straight",
                    )
                )
            out[0] = CurvePoint(x=start.x, y=start.y, yaw=start.yaw, phase=out[0].phase)
            out[-1] = CurvePoint(x=goal.x, y=goal.y, yaw=goal.yaw, phase=out[-1].phase)
            return out, "single arc degenerates to a straight line"
        return None, (
            "single arc failed: equal endpoint yaw but endpoint is not colinear "
            f"(lateral_residual={lateral:.4g}m)"
        )

    best: tuple[float, int, float, tuple[float, float]] | None = None
    p0 = (start.x, start.y)
    p1 = (goal.x, goal.y)
    l0 = left_unit(start.yaw)
    l1 = left_unit(goal.yaw)
    for sign in (1, -1):
        d = (sign * (l0[0] - l1[0]), sign * (l0[1] - l1[1]))
        den = d[0] * d[0] + d[1] * d[1]
        if den < 1e-12:
            continue
        rel = (p0[0] - p1[0], p0[1] - p1[1])
        radius = -(rel[0] * d[0] + rel[1] * d[1]) / den
        if radius <= 1e-6:
            continue
        center0 = (p0[0] + sign * radius * l0[0], p0[1] + sign * radius * l0[1])
        center1 = (p1[0] + sign * radius * l1[0], p1[1] + sign * radius * l1[1])
        residual = math.hypot(center0[0] - center1[0], center0[1] - center1[1])
        signed_delta = yaw_delta if sign > 0 else -yaw_delta
        if signed_delta <= 1e-6:
            continue
        item = (residual, sign, radius, center0)
        if best is None or item[0] < best[0]:
            best = item

    if best is None:
        return None, "single arc failed: no positive-radius arc has the required yaw direction"
    residual, sign, radius, center = best
    if residual > 1e-3:
        return None, f"single arc failed: center consistency residual={residual:.4g}m"

    total_heading = abs(yaw_delta)
    out: list[CurvePoint] = []
    for i in range(num_samples):
        t = i / max(num_samples - 1, 1)
        yaw = wrap_angle(start.yaw + sign * total_heading * t)
        lx, ly = left_unit(yaw)
        out.append(
            CurvePoint(
                x=center[0] - sign * radius * lx,
                y=center[1] - sign * radius * ly,
                yaw=yaw,
                phase=f"single_arc_R{radius:.3f}",
            )
        )
    out[0] = CurvePoint(x=start.x, y=start.y, yaw=start.yaw, phase=out[0].phase)
    out[-1] = CurvePoint(x=goal.x, y=goal.y, yaw=goal.yaw, phase=out[-1].phase)
    return out, f"single circular arc; radius={radius:.3f}m"


def build_two_segment_spline_pre_align_curve(
    *,
    start: Pose2,
    goal: Pose2,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
    num_samples: int,
    tangent_gain: float,
) -> list[CurvePoint]:
    """Two-stage Hermite guide: coarse lateral reduction then pre-align."""
    start_s, start_lat = transform_to_pallet((start.x, start.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    goal_s, goal_lat = transform_to_pallet((goal.x, goal.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    mid_s = start_s + 0.62 * (goal_s - start_s)
    mid_lat = 0.28 * start_lat + 0.72 * goal_lat
    terminal_ds = max(goal_s - mid_s, 1e-6)
    mid_yaw_rel = math.atan2(goal_lat - mid_lat, terminal_ds)
    mid_xy = transform_from_pallet(mid_s, mid_lat, pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    mid = Pose2(x=mid_xy[0], y=mid_xy[1], yaw=wrap_angle(pallet_yaw + mid_yaw_rel))

    n1 = max(3, int(round(num_samples * 0.56)))
    n2 = max(3, num_samples - n1 + 1)
    seg1 = build_hermite_direct_curve(start=start, goal=mid, num_samples=n1, tangent_gain=tangent_gain)
    seg2 = build_hermite_direct_curve(start=mid, goal=goal, num_samples=n2, tangent_gain=tangent_gain)
    out = seg1 + seg2[1:]
    if len(out) > num_samples:
        poses = [Pose2(p.x, p.y, p.yaw) for p in out]
        resampled = _resample_pose_list(poses, num_samples=num_samples)
        out = [CurvePoint(x=p.x, y=p.y, yaw=p.yaw, phase="two_segment_spline_pre_align") for p in resampled]
    out[0] = CurvePoint(x=start.x, y=start.y, yaw=start.yaw, phase=out[0].phase)
    out[-1] = CurvePoint(x=goal.x, y=goal.y, yaw=goal.yaw, phase=out[-1].phase)
    return out


def _pre_pose_for_axis_curve(
    *,
    start: Pose2,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
    goal_s: float,
    pre_dist_m: float,
    curve_min_span_m: float,
    final_straight_min_m: float,
) -> Pose2:
    start_s, _ = transform_to_pallet((start.x, start.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    pre_nominal_s = goal_s - pre_dist_m
    pre_s = max(pre_nominal_s, start_s + curve_min_span_m)
    pre_s = min(pre_s, goal_s - final_straight_min_m)
    pre_s = max(pre_s, start_s + 1e-4)
    pre_xy = transform_from_pallet(pre_s, 0.0, pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    return Pose2(x=pre_xy[0], y=pre_xy[1], yaw=pallet_yaw)


def _append_final_straight(
    out: list[CurvePoint],
    *,
    goal: Pose2,
    num_samples: int,
) -> list[CurvePoint]:
    if not out:
        return out
    start = out[-1]
    line_steps = max(2, num_samples - len(out) + 1)
    for i in range(1, line_steps + 1):
        t = i / line_steps
        out.append(
            CurvePoint(
                x=lerp(start.x, goal.x, t),
                y=lerp(start.y, goal.y, t),
                yaw=goal.yaw,
                phase="final_straight",
            )
        )
    return out


def _integrate_piecewise_constant_curvature(
    *,
    start: Pose2,
    segments: list[tuple[float, float, str]],
    samples_per_segment: int,
) -> list[CurvePoint]:
    out = [CurvePoint(x=start.x, y=start.y, yaw=start.yaw, phase="start")]
    x = start.x
    y = start.y
    yaw = start.yaw
    for length, curvature, phase in segments:
        steps = max(1, int(samples_per_segment))
        ds = float(length) / steps
        for _ in range(steps):
            dtheta = float(curvature) * ds
            mid_yaw = yaw + 0.5 * dtheta
            x += ds * math.cos(mid_yaw)
            y += ds * math.sin(mid_yaw)
            yaw = wrap_angle(yaw + dtheta)
            out.append(CurvePoint(x=x, y=y, yaw=yaw, phase=phase))
    return out


def build_clothoid_final_straight_curve(
    *,
    start: Pose2,
    goal: Pose2,
    pre: Pose2,
    num_samples: int,
    curvature_limit_1pm: float,
) -> tuple[list[CurvePoint] | None, str | None]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - depends on runtime env
        return None, f"clothoid solver unavailable: scipy import failed ({exc})"

    target = (pre.x, pre.y, pre.yaw)
    dx = pre.x - start.x
    dy = pre.y - start.y
    chord = max(math.hypot(dx, dy), 1e-3)
    yaw_delta = wrap_angle(pre.yaw - start.yaw)

    def integrate(length: float, k0: float, k1: float, n: int = 160) -> tuple[float, float, float]:
        x = start.x
        y = start.y
        yaw = start.yaw
        ds = length / max(n, 1)
        for i in range(n):
            s_mid = (i + 0.5) * ds
            k_mid = k0 + (k1 - k0) * s_mid / max(length, 1e-9)
            dtheta = k_mid * ds
            x += ds * math.cos(yaw + 0.5 * dtheta)
            y += ds * math.sin(yaw + 0.5 * dtheta)
            yaw = wrap_angle(yaw + dtheta)
        return x, y, yaw

    def residual(params: list[float]) -> list[float]:
        length = max(float(params[0]), 1e-3)
        k0 = float(params[1])
        k1 = float(params[2])
        x, y, yaw = integrate(length, k0, k1)
        return [
            x - target[0],
            y - target[1],
            wrap_angle(yaw - target[2]) * 0.75,
        ]

    avg_k = 2.0 * yaw_delta / max(chord, 1e-6)
    bound_k = max(4.0 * curvature_limit_1pm, abs(avg_k) + 1.0, 1.0)
    result = least_squares(
        residual,
        x0=[chord * 1.05, avg_k, avg_k],
        bounds=([0.25 * chord, -bound_k, -bound_k], [chord + 8.0, bound_k, bound_k]),
        max_nfev=300,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    err = math.sqrt(sum(v * v for v in residual(result.x)))
    if (not result.success) or err > 5e-3:
        return None, f"clothoid solve failed: success={result.success}, residual={err:.4g}"

    length, k0, k1 = (float(v) for v in result.x)
    curve_steps = max(12, int(num_samples * 0.72))
    out = [CurvePoint(x=start.x, y=start.y, yaw=start.yaw, phase="start")]
    x = start.x
    y = start.y
    yaw = start.yaw
    ds = length / max(curve_steps, 1)
    for i in range(1, curve_steps + 1):
        s_mid = (i - 0.5) * ds
        k_mid = k0 + (k1 - k0) * s_mid / max(length, 1e-9)
        dtheta = k_mid * ds
        x += ds * math.cos(yaw + 0.5 * dtheta)
        y += ds * math.sin(yaw + 0.5 * dtheta)
        yaw = wrap_angle(yaw + dtheta)
        out.append(CurvePoint(x=x, y=y, yaw=yaw, phase="true_clothoid"))
    out[-1] = CurvePoint(x=pre.x, y=pre.y, yaw=pre.yaw, phase="clothoid_join")
    return _append_final_straight(out, goal=goal, num_samples=num_samples), None


def build_biarc_final_straight_curve(
    *,
    start: Pose2,
    goal: Pose2,
    pre: Pose2,
    num_samples: int,
    curvature_limit_1pm: float,
) -> tuple[list[CurvePoint] | None, str | None]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # pragma: no cover - depends on runtime env
        return None, f"biarc solver unavailable: scipy import failed ({exc})"

    chord = max(math.hypot(pre.x - start.x, pre.y - start.y), 1e-3)
    yaw_delta = wrap_angle(pre.yaw - start.yaw)

    def integrate(params: list[float], samples: int = 96) -> list[CurvePoint]:
        l1 = max(float(params[0]), 1e-4)
        l2 = max(float(params[1]), 1e-4)
        k1 = float(params[2])
        k2 = float(params[3])
        return _integrate_piecewise_constant_curvature(
            start=start,
            segments=[(l1, k1, "biarc_arc1"), (l2, k2, "biarc_arc2")],
            samples_per_segment=max(2, samples // 2),
        )

    def residual(params: list[float]) -> list[float]:
        pts = integrate(params, samples=80)
        end = pts[-1]
        return [
            end.x - pre.x,
            end.y - pre.y,
            wrap_angle(end.yaw - pre.yaw) * 0.75,
            0.015 * (float(params[2]) - float(params[3])),
        ]

    avg_k = yaw_delta / max(chord, 1e-6)
    bound_k = max(4.0 * curvature_limit_1pm, abs(avg_k) + 1.0, 1.0)
    result = least_squares(
        residual,
        x0=[0.5 * chord, 0.5 * chord, avg_k, avg_k],
        bounds=([0.02, 0.02, -bound_k, -bound_k], [chord + 5.0, chord + 5.0, bound_k, bound_k]),
        max_nfev=400,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    err = math.sqrt(sum(v * v for v in residual(result.x)[:3]))
    if (not result.success) or err > 5e-3:
        return None, f"biarc solve failed: success={result.success}, residual={err:.4g}"
    arc_steps = max(16, int(num_samples * 0.72))
    out = integrate([float(v) for v in result.x], samples=arc_steps)
    out[-1] = CurvePoint(x=pre.x, y=pre.y, yaw=pre.yaw, phase="biarc_join")
    return _append_final_straight(out, goal=goal, num_samples=num_samples), None


def _rs_local_goal(start: Pose2, goal: Pose2) -> tuple[float, float, float]:
    dx = goal.x - start.x
    dy = goal.y - start.y
    cos_t = math.cos(start.yaw)
    sin_t = math.sin(start.yaw)
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    lphi = wrap_angle(goal.yaw - start.yaw)
    return -lx, -ly, lphi


def _rs_candidate_stats(segs: list[tuple[str, float]]) -> dict[str, float | int | bool]:
    total = float(sum(abs(seg_len) for _, seg_len in segs))
    reverse = float(sum(abs(seg_len) for _, seg_len in segs if seg_len < 0.0))
    switches = sum((segs[i][1] >= 0.0) != (segs[i - 1][1] >= 0.0) for i in range(1, len(segs)))
    final_forward = bool(segs and segs[-1][1] > 0.0)
    return {
        "total_length_m": total,
        "reverse_length_m": reverse,
        "reverse_frac": reverse / max(total, 1e-9),
        "direction_switches": int(switches),
        "final_forward": final_forward,
    }


def _sample_rs_segments_world(
    *,
    start: Pose2,
    segs: list[tuple[str, float]],
    turning_radius: float,
    step: float,
) -> list[Pose2]:
    cx = 0.0
    cy = 0.0
    cth = 0.0
    rs_pts = [(cx, cy, cth)]
    for stype, length_m in segs:
        direction = 1 if length_m >= 0.0 else -1
        dist_left = abs(length_m)
        while dist_left > 1e-9:
            ds = min(step, dist_left)
            dist_left -= ds
            if stype == "S":
                cx += direction * ds * math.cos(cth)
                cy += direction * ds * math.sin(cth)
            elif stype == "L":
                dth = direction * ds / turning_radius
                cx = cx + turning_radius * (math.sin(cth + dth) - math.sin(cth))
                cy = cy + turning_radius * (math.cos(cth) - math.cos(cth + dth))
                cth = wrap_angle(cth + dth)
            else:
                dth = -direction * ds / turning_radius
                cx = cx + turning_radius * (math.sin(cth) - math.sin(cth + dth))
                cy = cy + turning_radius * (math.cos(cth + dth) - math.cos(cth))
                cth = wrap_angle(cth + dth)
            rs_pts.append((cx, cy, cth))

    cos_t = math.cos(start.yaw)
    sin_t = math.sin(start.yaw)
    world: list[Pose2] = []
    for x_rs, y_rs, th_rs in rs_pts:
        lx = -x_rs
        ly = -y_rs
        wx = start.x + lx * cos_t - ly * sin_t
        wy = start.y + lx * sin_t + ly * cos_t
        world.append(Pose2(x=wx, y=wy, yaw=wrap_angle(start.yaw + th_rs)))
    return world


def _resample_pose_list(poses: list[Pose2], *, num_samples: int) -> list[Pose2]:
    if len(poses) <= 1:
        return poses
    distances = [0.0]
    for a, b in zip(poses[:-1], poses[1:]):
        distances.append(distances[-1] + math.hypot(b.x - a.x, b.y - a.y))
    total = distances[-1]
    if total < 1e-9:
        return [poses[0] for _ in range(num_samples)]
    yaws = unwrap_sequence([p.yaw for p in poses])
    out: list[Pose2] = []
    j = 0
    for i in range(num_samples):
        s = total * i / max(num_samples - 1, 1)
        while j < len(distances) - 2 and distances[j + 1] < s:
            j += 1
        span = max(distances[j + 1] - distances[j], 1e-9)
        t = (s - distances[j]) / span
        out.append(
            Pose2(
                x=lerp(poses[j].x, poses[j + 1].x, t),
                y=lerp(poses[j].y, poses[j + 1].y, t),
                yaw=wrap_angle(lerp(yaws[j], yaws[j + 1], t)),
            )
        )
    return out


def build_rs_family_curve(
    *,
    start: Pose2,
    goal: Pose2,
    model: str,
    num_samples: int,
    rs_module,
    min_turn_radius_m: float,
    sample_step_m: float,
    fork_forward_offset_m: float,
    fork_center_backoff_m: float,
    max_candidates: int,
    max_extra_length_m: float,
    max_reverse_frac: float,
    max_direction_switches: int,
    require_final_forward: bool,
) -> tuple[list[CurvePoint] | None, str | None]:
    root_to_fc = max(float(fork_forward_offset_m) - float(fork_center_backoff_m), 0.0)
    root_start = Pose2(
        x=start.x - root_to_fc * math.cos(start.yaw),
        y=start.y - root_to_fc * math.sin(start.yaw),
        yaw=start.yaw,
    )
    root_goal = Pose2(
        x=goal.x - root_to_fc * math.cos(goal.yaw),
        y=goal.y - root_to_fc * math.sin(goal.yaw),
        yaw=goal.yaw,
    )
    x_rs, y_rs, th_rs = _rs_local_goal(root_start, root_goal)
    all_segs = rs_module.rs_all_paths(x_rs, y_rs, th_rs, float(min_turn_radius_m))
    if not all_segs:
        return None, "RS solver returned no candidate paths"

    shortest_total = float(sum(abs(seg_len) for _, seg_len in all_segs[0]))
    candidates: list[tuple[float, list[tuple[str, float]], dict[str, float | int | bool]]] = []
    for segs in all_segs[: max(1, min(int(max_candidates), len(all_segs)))]:
        stats = _rs_candidate_stats(segs)
        if model == "dubins_forward" and any(seg_len <= 0.0 for _, seg_len in segs):
            continue
        if model == "rs_forward_preferred":
            if float(stats["total_length_m"]) > shortest_total + float(max_extra_length_m):
                continue
            if float(stats["reverse_frac"]) > float(max_reverse_frac):
                continue
            if int(stats["direction_switches"]) > int(max_direction_switches):
                continue
            if bool(require_final_forward) and not bool(stats["final_forward"]):
                continue
        score = (
            float(stats["total_length_m"])
            + 3.0 * float(stats["reverse_length_m"])
            + 0.8 * int(stats["direction_switches"])
            + (0.0 if bool(stats["final_forward"]) else 2.0)
        )
        candidates.append((score, segs, stats))
    if not candidates:
        if model == "dubins_forward":
            return None, "no all-forward Dubins-style RS candidate among searched paths"
        return None, (
            "no RS candidate satisfied forward-preferred filters "
            f"(max_extra_length={max_extra_length_m}, max_reverse_frac={max_reverse_frac}, "
            f"max_switches={max_direction_switches}, require_final_forward={require_final_forward})"
        )

    _, best_segs, stats = min(candidates, key=lambda item: item[0])
    root_poses = _sample_rs_segments_world(
        start=root_start,
        segs=best_segs,
        turning_radius=float(min_turn_radius_m),
        step=float(sample_step_m),
    )
    root_poses = _resample_pose_list(root_poses, num_samples=num_samples)
    out: list[CurvePoint] = []
    for pose in root_poses:
        out.append(
            CurvePoint(
                x=pose.x + root_to_fc * math.cos(pose.yaw),
                y=pose.y + root_to_fc * math.sin(pose.yaw),
                yaw=pose.yaw,
                phase=f"{model}:{','.join(f'{s}{l:+.2f}' for s, l in best_segs)}",
            )
        )
    # Force exact endpoint values after resampling; metrics still reveal interior behavior.
    out[0] = CurvePoint(x=start.x, y=start.y, yaw=start.yaw, phase=out[0].phase)
    out[-1] = CurvePoint(x=goal.x, y=goal.y, yaw=goal.yaw, phase=out[-1].phase)
    return out, (
        f"segments={','.join(f'{s}{l:+.3f}' for s, l in best_segs)}; "
        f"reverse_frac={float(stats['reverse_frac']):.3f}; "
        f"switches={int(stats['direction_switches'])}; "
        f"root_R_min={min_turn_radius_m:.3f}m"
    )


def unwrap_sequence(angles: Iterable[float]) -> list[float]:
    result: list[float] = []
    prev = None
    offset = 0.0
    for angle in angles:
        if prev is not None:
            delta = angle + offset - prev
            if delta > math.pi:
                offset -= 2.0 * math.pi
            elif delta < -math.pi:
                offset += 2.0 * math.pi
        value = angle + offset
        result.append(value)
        prev = value
    return result


def compute_metrics(
    *,
    case_id: str,
    model: str,
    points: list[CurvePoint],
    start: Pose2,
    goal: Pose2,
    pallet_xy: tuple[float, float],
    pallet_yaw: float,
    curvature_limit_1pm: float,
    terminal_corridor_dist_m: float,
    terminal_corridor_yaw_deg: float,
    terminal_corridor_length_m: float,
    notes: str,
) -> CurveMetrics:
    length = 0.0
    forward_steps: list[float] = []
    segment_yaws: list[float] = []
    local_lats: list[float] = []
    for p in points:
        _, lat = transform_to_pallet((p.x, p.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
        local_lats.append(lat)
    for a, b in zip(points[:-1], points[1:]):
        dx = b.x - a.x
        dy = b.y - a.y
        seg_len = math.hypot(dx, dy)
        length += seg_len
        if seg_len > 1e-9:
            segment_yaws.append(math.atan2(dy, dx))
        sa, _ = transform_to_pallet((a.x, a.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
        sb, _ = transform_to_pallet((b.x, b.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
        forward_steps.append(sb - sa)

    curvatures: list[float] = []
    if len(segment_yaws) >= 2:
        unwrapped = unwrap_sequence(segment_yaws)
        for i in range(len(unwrapped) - 1):
            a = points[i]
            b = points[i + 1]
            c = points[i + 2]
            ds1 = math.hypot(b.x - a.x, b.y - a.y)
            ds2 = math.hypot(c.x - b.x, c.y - b.y)
            ds = max(0.5 * (ds1 + ds2), 1e-9)
            curvatures.append((unwrapped[i + 1] - unwrapped[i]) / ds)

    start_p = points[0]
    end_p = points[-1]
    start_pos_err = math.hypot(start_p.x - start.x, start_p.y - start.y)
    end_pos_err = math.hypot(end_p.x - goal.x, end_p.y - goal.y)
    start_yaw_err = abs(wrap_angle(start_p.yaw - start.yaw)) * 180.0 / math.pi
    end_yaw_err = abs(wrap_angle(end_p.yaw - goal.yaw)) * 180.0 / math.pi
    heading_change = abs(wrap_angle(end_p.yaw - start_p.yaw)) * 180.0 / math.pi
    endpoint_ok = (
        start_pos_err < 1e-6
        and end_pos_err < 1e-6
        and start_yaw_err < 1e-3
        and end_yaw_err < 1e-3
    )
    max_abs_curvature = max((abs(v) for v in curvatures), default=0.0)
    terminal_corridor_ok = True
    if terminal_corridor_dist_m > 0.0 and terminal_corridor_length_m > 0.0:
        terminal_points = [points[-1]]
        remaining = terminal_corridor_length_m
        for a, b in zip(reversed(points[:-1]), reversed(points[1:])):
            if remaining <= 0.0:
                break
            terminal_points.append(a)
            remaining -= math.hypot(b.x - a.x, b.y - a.y)
        terminal_corridor_ok = all(
            abs(transform_to_pallet((p.x, p.y), pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)[1])
            <= terminal_corridor_dist_m
            and abs(wrap_angle(p.yaw - pallet_yaw)) * 180.0 / math.pi <= terminal_corridor_yaw_deg
            for p in terminal_points
        )
    feasible = max_abs_curvature <= curvature_limit_1pm + 1e-9
    failure_reasons: list[str] = []
    if not endpoint_ok:
        failure_reasons.append(
            "endpoint mismatch "
            f"(start_pos={start_pos_err:.3g}m, end_pos={end_pos_err:.3g}m, "
            f"start_yaw={start_yaw_err:.3g}deg, end_yaw={end_yaw_err:.3g}deg)"
        )
    if not feasible:
        failure_reasons.append(
            f"curvature infeasible: max_abs_curvature={max_abs_curvature:.3f} 1/m "
            f"> limit={curvature_limit_1pm:.3f} 1/m"
        )
    if not terminal_corridor_ok:
        failure_reasons.append(
            f"terminal insertion corridor failed: final {terminal_corridor_length_m:.2f}m must stay within "
            f"{terminal_corridor_dist_m:.3f}m lateral and {terminal_corridor_yaw_deg:.1f}deg yaw"
        )
    if not all(step >= -1e-6 for step in forward_steps):
        failure_reasons.append("non-monotone in pallet-forward coordinate")
    return CurveMetrics(
        case_id=case_id,
        model=model,
        length_m=length,
        max_abs_lateral_m=max(abs(v) for v in local_lats),
        max_abs_curvature_1pm=max_abs_curvature,
        curvature_start_1pm=curvatures[0] if curvatures else 0.0,
        curvature_end_1pm=curvatures[-1] if curvatures else 0.0,
        heading_change_deg=heading_change,
        min_forward_step_m=min(forward_steps) if forward_steps else 0.0,
        start_pos_err_m=start_pos_err,
        end_pos_err_m=end_pos_err,
        start_yaw_err_deg=start_yaw_err,
        end_yaw_err_deg=end_yaw_err,
        curvature_limit_1pm=curvature_limit_1pm,
        feasible=feasible,
        terminal_corridor_ok=terminal_corridor_ok,
        training_candidate=endpoint_ok and feasible and terminal_corridor_ok,
        failure_reasons=failure_reasons,
        endpoint_ok=endpoint_ok,
        monotone_forward=all(step >= -1e-6 for step in forward_steps),
        notes=notes,
    )


def format_case_id(index: int, start: Pose2) -> str:
    def token(value: float) -> str:
        sign = "p" if value >= 0.0 else "m"
        return f"{sign}{abs(value):.3f}".replace(".", "p")

    return f"c{index:02d}_x{token(start.x)}_y{token(start.y)}_yaw{token(math.degrees(start.yaw))}"


def build_cases(args: argparse.Namespace) -> list[Pose2]:
    xs = linspace(args.x_min, args.x_max, args.grid_x)
    ys = linspace(args.y_min, args.y_max, args.grid_y)
    yaws = [math.radians(v) for v in linspace(args.yaw_min_deg, args.yaw_max_deg, args.grid_yaw)]
    return [Pose2(x=x, y=y, yaw=yaw) for x in xs for y in ys for yaw in yaws]


def linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [(a + b) * 0.5]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def build_payloads(args: argparse.Namespace) -> list[CasePayload]:
    pallet_xy = (args.pallet_x, args.pallet_y)
    pallet_yaw = math.radians(args.pallet_yaw_deg)
    curvature_limit_1pm = 1.0 / max(float(args.min_turn_radius_m), 1e-6)
    rs_module = None
    if any(model in args.models for model in ("rs_forward_preferred", "dubins_forward")):
        try:
            rs_module = load_module_from_path("toyota_reference_rs", Path(args.rs_module_path))
        except Exception as exc:
            rs_module = exc
    s_front = -0.5 * args.pallet_depth_m
    if args.goal_mode == "front":
        goal_s = s_front
    elif args.goal_mode == "pre_align":
        goal_s = s_front - args.pre_align_fork_center_backoff_m
    elif args.goal_mode == "success_center":
        goal_s = s_front + (args.insert_fraction * args.pallet_depth_m - FORK_CENTER_BACKOFF_M)
    else:
        raise ValueError(f"unsupported goal mode: {args.goal_mode}")
    goal_xy = transform_from_pallet(goal_s, 0.0, pallet_xy=pallet_xy, pallet_yaw=pallet_yaw)
    goal = Pose2(x=goal_xy[0], y=goal_xy[1], yaw=pallet_yaw)

    payloads: list[CasePayload] = []
    for idx, start in enumerate(build_cases(args), start=1):
        case_id = format_case_id(idx, start)
        curves: dict[str, list[CurvePoint]] = {}
        metrics: dict[str, CurveMetrics] = {}
        pre_xy = goal_xy
        pre_pose = _pre_pose_for_axis_curve(
            start=start,
            pallet_xy=pallet_xy,
            pallet_yaw=pallet_yaw,
            goal_s=goal_s,
            pre_dist_m=args.pre_dist_m,
            curve_min_span_m=args.curve_min_span_m,
            final_straight_min_m=args.final_straight_min_m,
        )
        pre_xy = (pre_pose.x, pre_pose.y)

        if "poly3" in args.models:
            pts, pre_xy = build_axis_poly_curve(
                start=start,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                goal_s=goal_s,
                pre_dist_m=args.pre_dist_m,
                curve_min_span_m=args.curve_min_span_m,
                final_straight_min_m=args.final_straight_min_m,
                initial_straight_m=0.0,
                num_samples=args.num_samples,
                model="poly3",
            )
            curves["poly3"] = pts
            metrics["poly3"] = compute_metrics(
                case_id=case_id,
                model="poly3",
                points=pts,
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                curvature_limit_1pm=curvature_limit_1pm,
                terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                terminal_corridor_length_m=args.terminal_corridor_length_m,
                notes="minimal cubic y(s); curvature is approximately linear only for small slopes",
            )

        if "line_poly3" in args.models:
            pts, pre_xy = build_axis_poly_curve(
                start=start,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                goal_s=goal_s,
                pre_dist_m=args.pre_dist_m,
                curve_min_span_m=args.curve_min_span_m,
                final_straight_min_m=args.final_straight_min_m,
                initial_straight_m=args.initial_straight_m,
                num_samples=args.num_samples,
                model="poly3",
            )
            curves["line_poly3"] = pts
            metrics["line_poly3"] = compute_metrics(
                case_id=case_id,
                model="line_poly3",
                points=pts,
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                curvature_limit_1pm=curvature_limit_1pm,
                terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                terminal_corridor_length_m=args.terminal_corridor_length_m,
                notes="short initial straight + cubic transition + terminal straight; close to Fig.3 sketch",
            )

        if "line_g2_quintic" in args.models:
            pts, pre_xy = build_axis_poly_curve(
                start=start,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                goal_s=goal_s,
                pre_dist_m=args.pre_dist_m,
                curve_min_span_m=args.curve_min_span_m,
                final_straight_min_m=args.final_straight_min_m,
                initial_straight_m=args.initial_straight_m,
                num_samples=args.num_samples,
                model="g2_quintic",
            )
            curves["line_g2_quintic"] = pts
            metrics["line_g2_quintic"] = compute_metrics(
                case_id=case_id,
                model="line_g2_quintic",
                points=pts,
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                curvature_limit_1pm=curvature_limit_1pm,
                terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                terminal_corridor_length_m=args.terminal_corridor_length_m,
                notes="short straight + zero-curvature-end quintic transition + terminal straight; recommended proxy",
            )

        if "toyota_straight_clothoid_terminal" in args.models:
            pts, pre_xy = build_axis_poly_curve(
                start=start,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                goal_s=goal_s,
                pre_dist_m=args.pre_dist_m,
                curve_min_span_m=args.curve_min_span_m,
                final_straight_min_m=args.final_straight_min_m,
                initial_straight_m=args.initial_straight_m,
                num_samples=args.num_samples,
                model="g2_quintic",
            )
            curves["toyota_straight_clothoid_terminal"] = pts
            metrics["toyota_straight_clothoid_terminal"] = compute_metrics(
                case_id=case_id,
                model="toyota_straight_clothoid_terminal",
                points=pts,
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                curvature_limit_1pm=curvature_limit_1pm,
                terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                terminal_corridor_length_m=args.terminal_corridor_length_m,
                notes=(
                    "Stage2B Toyota-style proxy: episode-reset guide, short initial straight, "
                    "clothoid-like G2 transition, and short terminal pre-align straight"
                ),
            )

        if "single_arc_pre_align" in args.models:
            pts, error = build_single_arc_pre_align_curve(
                start=start,
                goal=goal,
                num_samples=args.num_samples,
            )
            if pts is None:
                metrics["single_arc_pre_align"] = failed_metric(
                    case_id=case_id,
                    model="single_arc_pre_align",
                    curvature_limit_1pm=curvature_limit_1pm,
                    reason=error or "single arc generation failed",
                    notes="Stage2B single circular arc from start pose to pallet-front pre-align pose",
                )
            else:
                curves["single_arc_pre_align"] = pts
                metrics["single_arc_pre_align"] = compute_metrics(
                    case_id=case_id,
                    model="single_arc_pre_align",
                    points=pts,
                    start=start,
                    goal=goal,
                    pallet_xy=pallet_xy,
                    pallet_yaw=pallet_yaw,
                    curvature_limit_1pm=curvature_limit_1pm,
                    terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                    terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                    terminal_corridor_length_m=args.terminal_corridor_length_m,
                    notes=error or "Stage2B single circular arc from start pose to pallet-front pre-align pose",
                )

        if "two_segment_spline_pre_align" in args.models:
            pts = build_two_segment_spline_pre_align_curve(
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                num_samples=args.num_samples,
                tangent_gain=args.hermite_tangent_gain,
            )
            curves["two_segment_spline_pre_align"] = pts
            metrics["two_segment_spline_pre_align"] = compute_metrics(
                case_id=case_id,
                model="two_segment_spline_pre_align",
                points=pts,
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                curvature_limit_1pm=curvature_limit_1pm,
                terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                terminal_corridor_length_m=args.terminal_corridor_length_m,
                notes="Stage2B two-phase spline: coarse approach then pallet-front pre-align pose",
            )

        if "hermite_direct" in args.models:
            pts = build_hermite_direct_curve(
                start=start,
                goal=goal,
                num_samples=args.num_samples,
                tangent_gain=args.hermite_tangent_gain,
            )
            curves["hermite_direct"] = pts
            metrics["hermite_direct"] = compute_metrics(
                case_id=case_id,
                model="hermite_direct",
                points=pts,
                start=start,
                goal=goal,
                pallet_xy=pallet_xy,
                pallet_yaw=pallet_yaw,
                curvature_limit_1pm=curvature_limit_1pm,
                terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                terminal_corridor_length_m=args.terminal_corridor_length_m,
                notes="direct cubic Hermite pose connector; no explicit terminal approach corridor",
            )

        if "true_clothoid" in args.models:
            pts, error = build_clothoid_final_straight_curve(
                start=start,
                goal=goal,
                pre=pre_pose,
                num_samples=args.num_samples,
                curvature_limit_1pm=curvature_limit_1pm,
            )
            if pts is None:
                metrics["true_clothoid"] = failed_metric(
                    case_id=case_id,
                    model="true_clothoid",
                    curvature_limit_1pm=curvature_limit_1pm,
                    reason=error or "clothoid generation failed",
                    notes="true clothoid attempt with linearly varying curvature plus final insertion straight",
                )
            else:
                curves["true_clothoid"] = pts
                metrics["true_clothoid"] = compute_metrics(
                    case_id=case_id,
                    model="true_clothoid",
                    points=pts,
                    start=start,
                    goal=goal,
                    pallet_xy=pallet_xy,
                    pallet_yaw=pallet_yaw,
                    curvature_limit_1pm=curvature_limit_1pm,
                    terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                    terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                    terminal_corridor_length_m=args.terminal_corridor_length_m,
                    notes="numerically solved Euler/clothoid-like transition to pre-align pose plus final straight",
                )

        if "biarc" in args.models:
            pts, error = build_biarc_final_straight_curve(
                start=start,
                goal=goal,
                pre=pre_pose,
                num_samples=args.num_samples,
                curvature_limit_1pm=curvature_limit_1pm,
            )
            if pts is None:
                metrics["biarc"] = failed_metric(
                    case_id=case_id,
                    model="biarc",
                    curvature_limit_1pm=curvature_limit_1pm,
                    reason=error or "biarc generation failed",
                    notes="two constant-curvature arcs plus final insertion straight",
                )
            else:
                curves["biarc"] = pts
                metrics["biarc"] = compute_metrics(
                    case_id=case_id,
                    model="biarc",
                    points=pts,
                    start=start,
                    goal=goal,
                    pallet_xy=pallet_xy,
                    pallet_yaw=pallet_yaw,
                    curvature_limit_1pm=curvature_limit_1pm,
                    terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                    terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                    terminal_corridor_length_m=args.terminal_corridor_length_m,
                    notes="biarc G1 connector plus final straight; solver may still find too-tight radii",
                )

        for rs_model in ("rs_forward_preferred", "dubins_forward"):
            if rs_model not in args.models:
                continue
            if isinstance(rs_module, BaseException):
                metrics[rs_model] = failed_metric(
                    case_id=case_id,
                    model=rs_model,
                    curvature_limit_1pm=curvature_limit_1pm,
                    reason=f"RS module load failed: {rs_module}",
                    notes="vehicle-kinematic root path mapped to fork-center guide curve",
                )
                continue
            pts, error = build_rs_family_curve(
                start=start,
                goal=goal,
                model=rs_model,
                num_samples=args.num_samples,
                rs_module=rs_module,
                min_turn_radius_m=args.min_turn_radius_m,
                sample_step_m=args.rs_sample_step_m,
                fork_forward_offset_m=args.fork_forward_offset_m,
                fork_center_backoff_m=FORK_CENTER_BACKOFF_M,
                max_candidates=args.rs_forward_preferred_max_candidates,
                max_extra_length_m=args.rs_forward_preferred_max_extra_length_m,
                max_reverse_frac=args.rs_forward_preferred_max_reverse_frac,
                max_direction_switches=args.rs_forward_preferred_max_direction_switches,
                require_final_forward=args.rs_forward_preferred_require_final_forward,
            )
            if pts is None:
                metrics[rs_model] = failed_metric(
                    case_id=case_id,
                    model=rs_model,
                    curvature_limit_1pm=curvature_limit_1pm,
                    reason=error or f"{rs_model} generation failed",
                    notes="vehicle-kinematic root path mapped to fork-center guide curve",
                )
            else:
                curves[rs_model] = pts
                metrics[rs_model] = compute_metrics(
                    case_id=case_id,
                    model=rs_model,
                    points=pts,
                    start=start,
                    goal=goal,
                    pallet_xy=pallet_xy,
                    pallet_yaw=pallet_yaw,
                    curvature_limit_1pm=curvature_limit_1pm,
                    terminal_corridor_dist_m=args.terminal_corridor_lateral_m,
                    terminal_corridor_yaw_deg=args.terminal_corridor_yaw_deg,
                    terminal_corridor_length_m=args.terminal_corridor_length_m,
                    notes=(error or "vehicle-kinematic root path mapped to fork-center guide curve"),
                )

        payloads.append(CasePayload(case_id=case_id, start=start, goal=goal, pre=pre_xy, curves=curves, metrics=metrics))
    return payloads


SVG_COLORS = {
    "poly3": "#1f77b4",
    "line_poly3": "#ff7f0e",
    "line_g2_quintic": "#2ca02c",
    "toyota_straight_clothoid_terminal": "#2ca02c",
    "single_arc_pre_align": "#e377c2",
    "two_segment_spline_pre_align": "#9467bd",
    "hermite_direct": "#9467bd",
    "true_clothoid": "#d62728",
    "biarc": "#17becf",
    "rs_forward_preferred": "#8c564b",
    "dubins_forward": "#bcbd22",
}


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SvgCanvas:
    def __init__(self, points: list[tuple[float, float]], *, width: int = 1100, height: int = 850):
        self.width = width
        self.height = height
        self.margin = 60
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)
        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)
        scale_x = (width - 2 * self.margin) / span_x
        scale_y = (height - 2 * self.margin) / span_y
        self.scale = min(scale_x, scale_y)
        cx = 0.5 * (self.min_x + self.max_x)
        cy = 0.5 * (self.min_y + self.max_y)
        visible_w = (width - 2 * self.margin) / self.scale
        visible_h = (height - 2 * self.margin) / self.scale
        self.min_x = cx - 0.5 * visible_w
        self.max_x = cx + 0.5 * visible_w
        self.min_y = cy - 0.5 * visible_h
        self.max_y = cy + 0.5 * visible_h

    def xy(self, x: float, y: float) -> tuple[float, float]:
        sx = self.margin + (x - self.min_x) * self.scale
        sy = self.height - self.margin - (y - self.min_y) * self.scale
        return sx, sy


def polyline_svg(points: list[CurvePoint], canvas: SvgCanvas, *, color: str, width: float, opacity: float, dash: str | None = None) -> str:
    coord = " ".join(f"{canvas.xy(p.x, p.y)[0]:.1f},{canvas.xy(p.x, p.y)[1]:.1f}" for p in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{coord}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'


def arrow_svg(p: CurvePoint, canvas: SvgCanvas, *, color: str, length_m: float = 0.25) -> str:
    x0, y0 = canvas.xy(p.x, p.y)
    x1, y1 = canvas.xy(p.x + length_m * math.cos(p.yaw), p.y + length_m * math.sin(p.yaw))
    return (
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
        f'stroke="{color}" stroke-width="2.0" marker-end="url(#arrow)"/>'
    )


def collect_svg_points(payloads: list[CasePayload], pallet_xy: tuple[float, float], pallet_depth: float) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = [(0.0, 0.0), pallet_xy]
    half = 0.5 * pallet_depth
    pts.extend([(pallet_xy[0] - half, pallet_xy[1] - 0.35), (pallet_xy[0] + half, pallet_xy[1] + 0.35)])
    for payload in payloads:
        pts.append((payload.start.x, payload.start.y))
        pts.append((payload.goal.x, payload.goal.y))
        pts.append(payload.pre)
        for curve in payload.curves.values():
            pts.extend((p.x, p.y) for p in curve)
    return pts


def svg_header(canvas: SvgCanvas, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas.width}" height="{canvas.height}" viewBox="0 0 {canvas.width} {canvas.height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L8,3 z" fill="#333"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="32" font-family="monospace" font-size="18" fill="#222">{svg_escape(title)}</text>',
    ]


def draw_pallet(lines: list[str], canvas: SvgCanvas, *, pallet_xy: tuple[float, float], pallet_depth: float) -> None:
    # The default exploration assumes pallet yaw is zero.  For nonzero yaw this
    # rectangle is still a harmless center marker; paths are transformed exactly.
    x0, y0 = pallet_xy
    half_d = 0.5 * pallet_depth
    half_w = 0.35
    p1 = canvas.xy(x0 - half_d, y0 - half_w)
    p2 = canvas.xy(x0 + half_d, y0 + half_w)
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    w = abs(p2[0] - p1[0])
    h = abs(p2[1] - p1[1])
    lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#555" stroke-width="2"/>')
    cx, cy = canvas.xy(x0, y0)
    lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#555"/>')
    lines.append(f'<text x="{cx + 8:.1f}" y="{cy - 8:.1f}" font-family="monospace" font-size="13" fill="#555">pallet</text>')


def write_overlay_svg(path: Path, payloads: list[CasePayload], args: argparse.Namespace) -> None:
    pallet_xy = (args.pallet_x, args.pallet_y)
    canvas = SvgCanvas(collect_svg_points(payloads, pallet_xy, args.pallet_depth_m))
    lines = svg_header(canvas, "Toyota-like reference curve exploration: all cases")
    draw_pallet(lines, canvas, pallet_xy=pallet_xy, pallet_depth=args.pallet_depth_m)
    for payload in payloads:
        for model, curve in payload.curves.items():
            lines.append(polyline_svg(curve, canvas, color=SVG_COLORS.get(model, "#333"), width=1.5, opacity=0.22))
        sx, sy = canvas.xy(payload.start.x, payload.start.y)
        lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.8" fill="#222" opacity="0.65"/>')
    legend_x = canvas.width - 340
    legend_y = 64
    for i, model in enumerate(args.models):
        y = legend_y + i * 22
        color = SVG_COLORS.get(model, "#333")
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 42}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 52}" y="{y + 4}" font-family="monospace" font-size="13" fill="#222">{svg_escape(model)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_svg(path: Path, payload: CasePayload, args: argparse.Namespace) -> None:
    pallet_xy = (args.pallet_x, args.pallet_y)
    canvas = SvgCanvas(collect_svg_points([payload], pallet_xy, args.pallet_depth_m), width=1000, height=780)
    lines = svg_header(canvas, f"{payload.case_id}: model comparison")
    draw_pallet(lines, canvas, pallet_xy=pallet_xy, pallet_depth=args.pallet_depth_m)
    px, py = canvas.xy(payload.pre[0], payload.pre[1])
    gx, gy = canvas.xy(payload.goal.x, payload.goal.y)
    lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#ff7f0e"/>')
    lines.append(f'<text x="{px + 8:.1f}" y="{py - 8:.1f}" font-family="monospace" font-size="12" fill="#ff7f0e">pre-align</text>')
    lines.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4" fill="#2ca02c"/>')
    lines.append(f'<text x="{gx + 8:.1f}" y="{gy + 18:.1f}" font-family="monospace" font-size="12" fill="#2ca02c">goal</text>')
    for model, curve in payload.curves.items():
        lines.append(polyline_svg(curve, canvas, color=SVG_COLORS.get(model, "#333"), width=2.4, opacity=0.95, dash="7,5" if model == "hermite_direct" else None))
        lines.append(arrow_svg(curve[0], canvas, color=SVG_COLORS.get(model, "#333")))
        lines.append(arrow_svg(curve[-1], canvas, color=SVG_COLORS.get(model, "#333")))
    sx, sy = canvas.xy(payload.start.x, payload.start.y)
    lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#222"/>')
    lines.append(f'<text x="{sx + 8:.1f}" y="{sy - 8:.1f}" font-family="monospace" font-size="12" fill="#222">start</text>')
    legend_x = 28
    legend_y = canvas.height - 110
    for i, model in enumerate(payload.curves):
        y = legend_y + i * 22
        color = SVG_COLORS.get(model, "#333")
        m = payload.metrics[model]
        text = f"{model}: len={m.length_m:.2f}m, kmax={m.max_abs_curvature_1pm:.2f}, monotone={int(m.monotone_forward)}"
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 42}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 52}" y="{y + 4}" font-family="monospace" font-size="13" fill="#222">{svg_escape(text)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_model_metrics(payloads: list[CasePayload]) -> dict[str, dict[str, float | int]]:
    by_model: dict[str, list[CurveMetrics]] = {}
    for payload in payloads:
        for model, metric in payload.metrics.items():
            by_model.setdefault(model, []).append(metric)
    summary: dict[str, dict[str, float | int]] = {}
    for model, metrics in by_model.items():
        n = len(metrics)
        summary[model] = {
            "n": n,
            "endpoint_ok_count": sum(int(m.endpoint_ok) for m in metrics),
            "monotone_forward_count": sum(int(m.monotone_forward) for m in metrics),
            "feasible_count": sum(int(m.feasible) for m in metrics),
            "terminal_corridor_ok_count": sum(int(m.terminal_corridor_ok) for m in metrics),
            "training_candidate_count": sum(int(m.training_candidate) for m in metrics),
            "mean_length_m": sum(m.length_m for m in metrics) / max(n, 1),
            "max_curvature_1pm": max(m.max_abs_curvature_1pm for m in metrics),
            "mean_max_abs_lateral_m": sum(m.max_abs_lateral_m for m in metrics) / max(n, 1),
            "max_start_curvature_abs_1pm": max(abs(m.curvature_start_1pm) for m in metrics),
            "max_end_curvature_abs_1pm": max(abs(m.curvature_end_1pm) for m in metrics),
            "failure_reasons": dict(Counter(reason for metric in metrics for reason in metric.failure_reasons)),
        }
    return summary


def recommended_models(summary: dict[str, dict[str, float | int]], model_order: list[str]) -> list[str]:
    candidates = []
    for index, model in enumerate(model_order):
        if model not in summary:
            continue
        s = summary[model]
        n = int(s["n"])
        if n <= 0 or int(s["training_candidate_count"]) < n:
            continue
        candidates.append(
            (
                float(s["max_curvature_1pm"]),
                float(s["mean_length_m"]),
                index,
                model,
            )
        )
    return [item[-1] for item in sorted(candidates)]


def write_manifest(path: Path, payloads: list[CasePayload], args: argparse.Namespace) -> None:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paper_url": PAPER_URL,
        "curvature_feasibility": {
            "min_turn_radius_m": float(args.min_turn_radius_m),
            "curvature_limit_1pm": 1.0 / max(float(args.min_turn_radius_m), 1e-6),
            "source": args.min_turn_radius_source,
            "terminal_corridor_lateral_m": float(args.terminal_corridor_lateral_m),
            "terminal_corridor_yaw_deg": float(args.terminal_corridor_yaw_deg),
            "terminal_corridor_length_m": float(args.terminal_corridor_length_m),
        },
        "paper_facts": [
            "start is the forklift position at task start",
            "terminal position is the pallet",
            "reference trajectory is based on an approximation of a clothoid curve",
            "reference trajectory remains fixed throughout the task",
            "reward queries fork-center distance to the pallet, fork-center distance to the curve, and heading-vs-tangent error",
        ],
        "interpretation": [
            "fixed throughout the task means generated at episode reset and frozen during that episode",
            "Stage2B uses a pallet-front pre-align endpoint when goal_mode=pre_align, not a final deep-insertion endpoint",
            "the paper does not publish the clothoid construction, boundary yaw, solver, curvature bounds, or reward weights",
            "the most useful reproducible proxy is a reset-time fork-center guide curve, not a teacher demonstration",
        ],
        "args": args_dict,
        "model_summary": summarize_model_metrics(payloads),
        "cases": [
            {
                "case_id": payload.case_id,
                "start": {"x": payload.start.x, "y": payload.start.y, "yaw_deg": math.degrees(payload.start.yaw)},
                "goal": {"x": payload.goal.x, "y": payload.goal.y, "yaw_deg": math.degrees(payload.goal.yaw)},
                "pre": {"x": payload.pre[0], "y": payload.pre[1]},
                "metrics": {model: asdict(metric) for model, metric in payload.metrics.items()},
            }
            for payload in payloads
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(path: Path, payloads: list[CasePayload], args: argparse.Namespace) -> None:
    summary = summarize_model_metrics(payloads)
    recommendations = recommended_models(summary, list(args.models))
    curvature_limit = 1.0 / max(float(args.min_turn_radius_m), 1e-6)
    lines = [
        "# Toyota Reference Curve Exploration",
        "",
        f"Paper: {PAPER_URL}",
        "",
        "## Feasibility Gate",
        "",
        f"- R_min = {float(args.min_turn_radius_m):.3f} m.",
        f"- Curvature limit = {curvature_limit:.3f} 1/m.",
        f"- Source: {args.min_turn_radius_source}.",
        f"- Terminal insertion corridor = final {float(args.terminal_corridor_length_m):.2f} m within {float(args.terminal_corridor_lateral_m):.3f} m lateral and {float(args.terminal_corridor_yaw_deg):.1f} deg yaw.",
        f"- A training candidate must pass endpoint, curvature, and terminal insertion-corridor gates.",
        f"- Goal mode = `{args.goal_mode}`.",
        "",
        "## What the paper actually fixes",
        "",
        "- A reference trajectory is generated from the task-start forklift position to the pallet.",
        "- The trajectory is based on an approximation of a clothoid curve.",
        "- The trajectory is fixed during the task episode.",
        "- The reward uses distance to pallet, distance to this curve, and heading error to the curve tangent.",
        "",
        "The paper does not specify the actual generator, boundary yaw choices, number of segments, curvature limits, solver, or reward weights.",
        "",
        "## Tested Reproducible Proxies",
        "",
        "- `poly3`: cubic lateral polynomial. This is the smallest approximation: for small heading angles, curvature is approximately linear along the path.",
        "- `line_poly3`: short initial straight + cubic transition + final straight. This visually matches Fig.3 better, but the cubic transition can have nonzero curvature at the joins.",
        "- `line_g2_quintic`: short initial straight + quintic transition with zero second derivative at both joins + final straight.",
        "- `toyota_straight_clothoid_terminal`: Stage2B Toyota-style proxy: episode-fixed, forward guide, short initial straight + clothoid-like G2 transition + terminal pre-align straight.",
        "- `single_arc_pre_align`: Stage2B single circular arc to the pallet-front pre-align pose.",
        "- `two_segment_spline_pre_align`: Stage2B two-phase smooth guide, first coarse approach then pre-align.",
        "- `hermite_direct`: direct pose-to-goal cubic Hermite. Useful contrast, but it lacks an explicit terminal insertion corridor.",
        "- `true_clothoid`: numeric linearly varying curvature solve to a pre-align pose plus final straight.",
        "- `biarc`: two bounded-radius arc segments plus final straight.",
        "- `rs_forward_preferred`: vehicle/root Reeds-Shepp path filtered toward forward motion, mapped to fork-center.",
        "- `dubins_forward`: all-forward RS/Dubins-style candidate when available.",
        "",
        "These are Toyota-like proxy curves, not a recovered original implementation. The paper does not provide enough detail to reproduce the author's exact generator.",
        "",
        "## Aggregate Metrics",
        "",
        "| model | endpoint ok | feasible | corridor ok | training candidate | monotone | mean length m | max curvature 1/m | max abs k_start | max abs k_end |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in args.models:
        if model not in summary:
            continue
        s = summary[model]
        n = int(s["n"])
        lines.append(
            "| {model} | {ep}/{n} | {feas}/{n} | {corr}/{n} | {cand}/{n} | {mono}/{n} | {length:.3f} | {kmax:.3f} | {ks:.3f} | {ke:.3f} |".format(
                model=model,
                ep=int(s["endpoint_ok_count"]),
                n=n,
                feas=int(s["feasible_count"]),
                corr=int(s["terminal_corridor_ok_count"]),
                cand=int(s["training_candidate_count"]),
                mono=int(s["monotone_forward_count"]),
                length=float(s["mean_length_m"]),
                kmax=float(s["max_curvature_1pm"]),
                ks=float(s["max_start_curvature_abs_1pm"]),
                ke=float(s["max_end_curvature_abs_1pm"]),
            )
        )
    lines.extend(["", "## Failure Reasons", ""])
    for model in args.models:
        if model not in summary:
            continue
        reasons = summary[model].get("failure_reasons", {})
        if not reasons:
            lines.append(f"- `{model}`: no gate failures.")
            continue
        reason_text = "; ".join(f"{count}x {reason}" for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:6])
        lines.append(f"- `{model}`: {reason_text}")

    lines.extend(["", "## Recommendation", ""])
    if recommendations:
        lines.append(
            "Train feasible curves in this order: "
            + ", ".join(f"`{model}`" for model in recommendations)
            + "."
        )
        lines.append("The ordering prefers lower peak curvature, then shorter mean path length.")
    else:
        lines.append("No model passed all hard gates; do not start curve-guidance PPO from this exploration.")
        lines.append("This is a curve-exploration FAIL caused by the gate reasons above, not a training failure.")
    lines.extend(
        [
            "",
            "Implementation note: generate the curve at reset from the current episode start pose and pallet pose, cache its sampled points/tangents, and do not update it inside the episode.",
            "",
            "Generated artifacts:",
            "- `manifest.json`: complete per-case metrics",
            "- `overlay_all_cases.svg`: all sampled cases and models",
            "- `case_*.svg`: selected detailed comparisons",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toyota_reference_curve_exploration"))
    parser.add_argument("--no-stamp", action="store_true", help="write directly into output-dir instead of a timestamped subdir")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "poly3",
            "line_poly3",
            "line_g2_quintic",
            "toyota_straight_clothoid_terminal",
            "single_arc_pre_align",
            "two_segment_spline_pre_align",
            "hermite_direct",
            "true_clothoid",
            "biarc",
            "rs_forward_preferred",
            "dubins_forward",
        ],
        choices=[
            "poly3",
            "line_poly3",
            "line_g2_quintic",
            "toyota_straight_clothoid_terminal",
            "single_arc_pre_align",
            "two_segment_spline_pre_align",
            "hermite_direct",
            "true_clothoid",
            "biarc",
            "rs_forward_preferred",
            "dubins_forward",
        ],
    )
    parser.add_argument("--grid-x", type=int, default=3)
    parser.add_argument("--grid-y", type=int, default=3)
    parser.add_argument("--grid-yaw", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=-4.0)
    parser.add_argument("--x-max", type=float, default=-3.0)
    parser.add_argument("--y-min", type=float, default=-0.6)
    parser.add_argument("--y-max", type=float, default=0.6)
    parser.add_argument("--yaw-min-deg", type=float, default=-14.32394487827058)
    parser.add_argument("--yaw-max-deg", type=float, default=14.32394487827058)
    parser.add_argument("--pallet-x", type=float, default=0.0)
    parser.add_argument("--pallet-y", type=float, default=0.0)
    parser.add_argument("--pallet-yaw-deg", type=float, default=0.0)
    parser.add_argument("--pallet-depth-m", type=float, default=2.16)
    parser.add_argument("--insert-fraction", type=float, default=0.40)
    parser.add_argument("--goal-mode", choices=["front", "pre_align", "success_center"], default="front")
    parser.add_argument(
        "--pre-align-fork-center-backoff-m",
        type=float,
        default=FORK_CENTER_BACKOFF_M,
        help=(
            "For goal-mode=pre_align, place the fork-center target this far before the pallet front edge, "
            "so fork tips are near but not inside the pallet entry."
        ),
    )
    parser.add_argument("--pre-dist-m", type=float, default=1.05)
    parser.add_argument("--curve-min-span-m", type=float, default=0.35)
    parser.add_argument("--final-straight-min-m", type=float, default=0.10)
    parser.add_argument("--initial-straight-m", type=float, default=0.30)
    parser.add_argument("--num-samples", type=int, default=96)
    parser.add_argument("--hermite-tangent-gain", type=float, default=1.5)
    parser.add_argument(
        "--min-turn-radius-m",
        type=float,
        default=2.34,
        help="Forklift/Ackermann minimum turning radius used for curvature feasibility.",
    )
    parser.add_argument(
        "--min-turn-radius-source",
        type=str,
        default="IsaacLab v311 cfg: wheelbase ~= 1.6m, max steer ~= 0.6rad, R_min ~= wheelbase/tan(0.6) ~= 2.34m",
    )
    parser.add_argument("--terminal-corridor-lateral-m", type=float, default=0.08)
    parser.add_argument("--terminal-corridor-yaw-deg", type=float, default=6.0)
    parser.add_argument(
        "--terminal-corridor-length-m",
        type=float,
        default=0.30,
        help="Final path length that must remain aligned with the pallet insertion axis.",
    )
    parser.add_argument(
        "--rs-module-path",
        type=Path,
        default=Path(
            "/data/jianshi/projects/forklift_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/forklift_pallet_insert_lift/rs/rs.py"
        ),
    )
    parser.add_argument("--rs-sample-step-m", type=float, default=0.05)
    parser.add_argument(
        "--fork-forward-offset-m",
        type=float,
        default=1.87,
        help="Root to fork-tip forward offset. Fork-center offset subtracts 0.6m backoff.",
    )
    parser.add_argument("--rs-forward-preferred-max-candidates", type=int, default=12)
    parser.add_argument("--rs-forward-preferred-max-extra-length-m", type=float, default=1.50)
    parser.add_argument("--rs-forward-preferred-max-reverse-frac", type=float, default=0.35)
    parser.add_argument("--rs-forward-preferred-max-direction-switches", type=int, default=1)
    parser.add_argument("--rs-forward-preferred-require-final-forward", action="store_true", default=True)
    parser.add_argument("--rs-forward-preferred-allow-final-reverse", action="store_false", dest="rs_forward_preferred_require_final_forward")
    parser.add_argument("--selected-cases", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    if not args.no_stamp:
        out_dir = out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = build_payloads(args)
    write_manifest(out_dir / "manifest.json", payloads, args)
    write_summary(out_dir / "summary.md", payloads, args)
    write_overlay_svg(out_dir / "overlay_all_cases.svg", payloads, args)

    selected = payloads[: max(0, args.selected_cases)]
    if payloads and args.selected_cases > 0:
        # Also include a few edge cases from the end of the grid.
        for payload in payloads[-max(0, args.selected_cases // 2) :]:
            if payload.case_id not in {p.case_id for p in selected}:
                selected.append(payload)
    for payload in selected:
        write_case_svg(out_dir / f"case_{payload.case_id}.svg", payload, args)

    summary = summarize_model_metrics(payloads)
    print(f"wrote Toyota reference curve exploration to: {out_dir}")
    for model in args.models:
        if model not in summary:
            continue
        s = summary[model]
        print(
            "{model}: endpoint_ok={ep}/{n}, monotone={mono}/{n}, mean_len={length:.3f}m, max_k={kmax:.3f} 1/m".format(
                model=model,
                ep=int(s["endpoint_ok_count"]),
                n=int(s["n"]),
                mono=int(s["monotone_forward_count"]),
                length=float(s["mean_length_m"]),
                kmax=float(s["max_curvature_1pm"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
