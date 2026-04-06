#!/usr/bin/env python3
"""
Synthetic dataset generator for ping pong interception learning.

Key behavior:
1. Forward-simulate trajectory in robot frame.
2. Select the in-workspace trajectory point (after min_time_hit) that is
   closest to ROBOT_HOME. If none are in workspace, fallback to the closest
   trajectory point overall.
3. Mark that selected point as `is_reachable=1` only if it lies inside the
   workspace, else `is_reachable=0`.

Inputs (default):
    x1 y1 z1 ... xK yK zK dt12 ... dt(K-1)K

Outputs:
    x_hit y_hit z_hit vx_hit vy_hit vz_hit t_hit is_reachable
    where t_hit is time-from-last-input-point (pointK)-to-intercept [s]
    (+ metadata columns)
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESETS_YAML = SCRIPT_DIR / "synthetic_data_generated" / "synthetic_generator_presets.yaml"
DEFAULT_RUNS_ROOT = SCRIPT_DIR / "synthetic_data_generated" / "runs"


# Match top_level_control/trajectory/workspace.py defaults (mm units).
@dataclass(frozen=True)
class WorkspaceConfig:
    ellipse_a: float = 720.0
    ellipse_b: float = 470.0
    z_min: float = -1020.0
    z_max: float = -800.0
    robot_home: Tuple[float, float, float] = (0.0, 0.0, -900.0)


@dataclass(frozen=True)
class NoiseConfig:
    position_std_mm: float = 15.0
    # Optional axis-wise position noise stds [mm]. If provided, they override
    # position_std_mm per axis for x/y/z respectively.
    position_std_x_mm: Optional[float] = None
    position_std_y_mm: Optional[float] = None
    position_std_z_mm: Optional[float] = None
    bias_std_mm: float = 4.0
    outlier_prob: float = 0.02
    outlier_std_mm: float = 30.0
    timestamp_jitter_ms: float = 2.0


@dataclass(frozen=True)
class GeneratorConfig:
    n_samples: int = 10000
    k_points: int = 6
    scan_duration_s: float = 1.5
    scan_dt_s: float = 0.005
    min_time_hit_s: float = 0.10
    gravity_z_mm_s2: float = -9810.0
    gravity_jitter_frac_std: float = 0.03
    obs_dt_min_s: float = 0.01
    obs_dt_max_s: float = 0.02
    obs_pre_hit_margin_s: float = 0.030
    obs_last_time_cap_s: float = 0.250
    # Observation start strategy:
    # - "entry": point1 starts at FoV upper-y entry.
    # - "uniform": point1 start time sampled uniformly across feasible window.
    obs_start_mode: str = "entry"
    max_attempts_factor: int = 60
    preview_duration_s: float = 2.5
    preview_dt_s: float = 0.005
    preview_stop_y_mm: float = -700.0
    preview_stop_z_mm: float = -1700.0
    table_z_mm: float = -1150.74
    table_length_mm: float = 2740.0
    table_width_mm: float = 1525.0
    max_bounces: int = 2
    preview_z_exaggeration: float = 1.8
    # Optional reachable label balancing in generated dataset.
    # None -> keep natural generated distribution.
    # Float in [0,1] -> target fraction of samples with is_reachable=1.
    reachable_target_frac: Optional[float] = None


@dataclass(frozen=True)
class LaunchBoxConfig:
    # Player-side launch box in robot frame (mm), from user-provided 8 vertices.
    # This is where the physical shot starts (not where x1/y1/z1 are sampled).
    x_min: float = -1062.5
    x_max: float = 1062.5
    # Original launch depth range (before FoV-limited observation windowing).
    y_min: float = 1940.0
    y_max: float = 3240.0
    # Clarified convention: launch Z is always negative in this robot frame.
    z_min: float = -1150.74
    z_max: float = -450.74


@dataclass(frozen=True)
class ObservationWindowConfig:
    # Camera-visible y-range for observed model inputs x1..xK (mm).
    y_min: float = 760.0
    y_max: float = 2060.0
    # Optional hard bounds for observed point1 x1 (mm). When provided,
    # sample generation enforces x1 inside [x1_min, x1_max].
    x1_min: Optional[float] = None
    x1_max: Optional[float] = None
    # Optional hard bounds for observed point1 y1 (mm).
    y1_min: Optional[float] = None
    y1_max: Optional[float] = None
    # Optional hard bounds for observed point1 z1 (mm).
    z1_min: Optional[float] = None
    z1_max: Optional[float] = None


@dataclass(frozen=True)
class RealMatchedConfig:
    enabled: bool = False
    dt_mode: str = "uniform"
    dt_mean_s: float = 0.0158
    dt_std_s: float = 0.0013
    dt_min_s: float = 0.0125
    dt_max_s: float = 0.0210
    require_valid_intercept: bool = False
    require_reachable: bool = False
    # When max_bounces >= 2, probabilistically allow second bounce to better
    # match real-data bounce count distribution.
    second_bounce_prob: float = 1.0
    # Optional filter on time-from-last-input-point (pointK)-to-hit [s] for tighter matching.
    t_hit_min_s: Optional[float] = None
    t_hit_max_s: Optional[float] = None
    # Optional velocity-shaping controls (applied only when enabled=True).
    vx_scale: float = 1.0
    vy_scale: float = 1.0
    vz_scale: float = 1.0
    vz_bias_mm_s: float = 0.0
    vx_noise_std_mm_s: float = 180.0
    vy_noise_std_mm_s: float = 220.0
    vz_noise_std_mm_s: float = 180.0
    # Observation start strategy inside the FoV window:
    # - "entry": point1 starts at upper-y window entry.
    # - "uniform": point1 start time is sampled uniformly inside feasible window.
    obs_start_mode: str = "entry"


def _deep_update(dst: Dict, src: Dict) -> Dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _build_dataclass(cls, overrides: Optional[Dict] = None):
    overrides = overrides or {}
    defaults = asdict(cls())
    unknown = sorted(set(overrides.keys()) - set(defaults.keys()))
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {unknown}")
    defaults.update(overrides)
    return cls(**defaults)


def load_presets_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Preset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    presets = data.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise ValueError(f"Invalid presets YAML structure in: {path}")
    return presets


def resolve_preset_bundle(presets: Dict, preset_name: str) -> Dict:
    if preset_name not in presets:
        raise KeyError(f"Preset '{preset_name}' not found. Available: {sorted(presets.keys())}")

    preset_raw = dict(presets[preset_name] or {})
    base_name = preset_raw.pop("inherits", None)

    if base_name:
        if base_name not in presets:
            raise KeyError(
                f"Preset '{preset_name}' inherits unknown preset '{base_name}'. "
                f"Available: {sorted(presets.keys())}"
            )
        merged = dict(presets[base_name] or {})
        _deep_update(merged, preset_raw)
        return merged

    return preset_raw


def sanitize_tag(text: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text.strip())
    return safe.strip("_")


def in_workspace(x: float, y: float, z: float, ws: WorkspaceConfig) -> bool:
    return (
        ws.z_min <= z <= ws.z_max
        and (x / ws.ellipse_a) ** 2 + (y / ws.ellipse_b) ** 2 <= 1.0
    )


def _map_to_range(val: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return float(dst_min)
    alpha = (val - src_min) / (src_max - src_min)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(dst_min + alpha * (dst_max - dst_min))


def _resolve_position_std_xyz(noise: NoiseConfig) -> np.ndarray:
    sx = noise.position_std_x_mm if noise.position_std_x_mm is not None else noise.position_std_mm
    sy = noise.position_std_y_mm if noise.position_std_y_mm is not None else noise.position_std_mm
    sz = noise.position_std_z_mm if noise.position_std_z_mm is not None else noise.position_std_mm
    return np.asarray(
        [max(0.0, float(sx)), max(0.0, float(sy)), max(0.0, float(sz))],
        dtype=float,
    )


def _intersect_time_range_linear_bounds(
    t_min: float,
    t_max: float,
    p0: float,
    v: float,
    p_min: Optional[float],
    p_max: Optional[float],
) -> Optional[Tuple[float, float]]:
    lo = t_min
    hi = t_max

    if p_min is None and p_max is None:
        return lo, hi

    if p_min is not None and p_max is not None and p_min > p_max:
        p_min, p_max = p_max, p_min

    if abs(v) <= 1e-9:
        if p_min is not None and p0 < p_min:
            return None
        if p_max is not None and p0 > p_max:
            return None
        return lo, hi

    if p_min is not None and p_max is not None:
        t_a = (p_min - p0) / v
        t_b = (p_max - p0) / v
        lo = max(lo, min(t_a, t_b))
        hi = min(hi, max(t_a, t_b))
        return (lo, hi) if lo <= hi else None

    if p_min is not None:
        t_thr = (p_min - p0) / v
        if v > 0:
            lo = max(lo, t_thr)
        else:
            hi = min(hi, t_thr)
    if p_max is not None:
        t_thr = (p_max - p0) / v
        if v > 0:
            hi = min(hi, t_thr)
        else:
            lo = max(lo, t_thr)

    return (lo, hi) if lo <= hi else None


def _point1_in_constraints(point_xyz: np.ndarray, obs_window: ObservationWindowConfig) -> bool:
    x1 = float(point_xyz[0])
    y1 = float(point_xyz[1])
    z1 = float(point_xyz[2])

    if obs_window.x1_min is not None and x1 < float(obs_window.x1_min):
        return False
    if obs_window.x1_max is not None and x1 > float(obs_window.x1_max):
        return False
    if obs_window.y1_min is not None and y1 < float(obs_window.y1_min):
        return False
    if obs_window.y1_max is not None and y1 > float(obs_window.y1_max):
        return False
    if obs_window.z1_min is not None and z1 < float(obs_window.z1_min):
        return False
    if obs_window.z1_max is not None and z1 > float(obs_window.z1_max):
        return False
    return True


def simulate_no_loss_trajectory(
    x0: float,
    y0: float,
    z0: float,
    vx0: float,
    vy0: float,
    vz0: float,
    gravity_z: float,
    duration_s: float,
    dt_s: float,
    table_z_mm: Optional[float] = None,
    max_bounces: int = 0,
    stop_y_mm: Optional[float] = None,
    stop_z_mm: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_list: List[float] = []
    s_list: List[List[float]] = []
    b_list: List[int] = []

    x = float(x0)
    y = float(y0)
    z = float(z0)
    vx = float(vx0)
    vy = float(vy0)
    vz = float(vz0)
    t = 0.0
    bounce_count = 0
    while t <= duration_s + 1e-9:
        t_list.append(float(t))
        s_list.append([float(x), float(y), float(z), float(vx), float(vy), float(vz)])
        b_list.append(int(bounce_count))

        if t >= duration_s:
            break
        if stop_y_mm is not None and y <= stop_y_mm:
            break
        if stop_z_mm is not None and z <= stop_z_mm:
            break

        step = min(dt_s, duration_s - t)
        x_prev, y_prev, z_prev = x, y, z
        vz_prev = vz

        x_next = x_prev + vx * step
        y_next = y_prev + vy * step
        z_next = z_prev + vz_prev * step + 0.5 * gravity_z * step * step
        vz_next = vz_prev + gravity_z * step

        if table_z_mm is not None and bounce_count < max_bounces:
            if z_prev >= table_z_mm and z_next < table_z_mm:
                dz = z_next - z_prev
                if abs(dz) > 1e-12:
                    frac = (table_z_mm - z_prev) / dz
                    frac = max(0.0, min(1.0, float(frac)))

                    t_to_bounce = step * frac
                    xb = x_prev + vx * t_to_bounce
                    yb = y_prev + vy * t_to_bounce
                    vz_bounce = vz_prev + gravity_z * t_to_bounce

                    # No-loss bounce: flip vertical velocity only.
                    vz_after = -vz_bounce

                    remain = step - t_to_bounce
                    x_next = xb + vx * remain
                    y_next = yb + vy * remain
                    z_next = table_z_mm + vz_after * remain + 0.5 * gravity_z * remain * remain
                    vz_next = vz_after + gravity_z * remain
                    bounce_count += 1

        x, y, z, vz = x_next, y_next, z_next, vz_next
        t += step

    return (
        np.asarray(t_list, dtype=float),
        np.asarray(s_list, dtype=float),
        np.asarray(b_list, dtype=int),
    )


def select_intercept_like_robot_predictor(
    t: np.ndarray,
    states: np.ndarray,
    bounces: np.ndarray,
    ws: WorkspaceConfig,
    min_time_hit_s: float,
) -> Optional[Dict[str, float]]:
    cx, cy, cz = ws.robot_home
    best_ws: Optional[Dict[str, float]] = None
    best_ws_d2 = float("inf")
    best_any: Optional[Dict[str, float]] = None
    best_any_d2 = float("inf")

    for i in range(t.shape[0]):
        ti = float(t[i])
        if ti < min_time_hit_s:
            continue

        x, y, z, vx, vy, vz = states[i]
        b_before = int(bounces[i]) if i < bounces.shape[0] else 0
        has_bounce = 1.0 if b_before > 0 else 0.0
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2

        if in_workspace(float(x), float(y), float(z), ws) and d2 < best_ws_d2:
            best_ws_d2 = float(d2)
            best_ws = {
                "x_hit": float(x),
                "y_hit": float(y),
                "z_hit": float(z),
                "vx_hit": float(vx),
                "vy_hit": float(vy),
                "vz_hit": float(vz),
                # Absolute hit time from launch-time reference (t=0 at physical launch).
                # Converted to time-from-last-input-point (pointK) in build_row().
                "t_hit_abs": ti,
                "is_reachable": 1.0,
                "bounces_before_hit": float(b_before),
                "has_bounce_before_hit": has_bounce,
            }

        if d2 < best_any_d2:
            best_any_d2 = float(d2)
            reachable = 1.0 if in_workspace(float(x), float(y), float(z), ws) else 0.0
            best_any = {
                "x_hit": float(x),
                "y_hit": float(y),
                "z_hit": float(z),
                "vx_hit": float(vx),
                "vy_hit": float(vy),
                "vz_hit": float(vz),
                # Absolute hit time from launch-time reference (t=0 at physical launch).
                # Converted to time-from-last-input-point (pointK) in build_row().
                "t_hit_abs": ti,
                "is_reachable": reachable,
                "bounces_before_hit": float(b_before),
                "has_bounce_before_hit": has_bounce,
            }
    return best_ws if best_ws is not None else best_any


def sample_initial_state(
    rng: np.random.Generator,
    box: LaunchBoxConfig,
    ws: WorkspaceConfig,
    gravity_z: float,
    rm_cfg: Optional[RealMatchedConfig] = None,
) -> Tuple[float, float, float, float, float, float]:
    if rm_cfg is None:
        rm_cfg = RealMatchedConfig()

    # Physical launch state starts inside the player-side launch box.
    x0 = float(rng.uniform(box.x_min, box.x_max))
    y0 = float(rng.uniform(box.y_min, box.y_max))
    z0 = float(rng.uniform(box.z_min, box.z_max))

    # Paper-like sweep remapped to our frame:
    # - Along-table speed formula applied along our +y launch extent.
    # - Width speed formula applied across our x launch extent.
    # - vy directed toward robot side (-y).
    y_sweep_m = _map_to_range(y0, box.y_min, box.y_max, 0.0, 2.8)
    x_sweep_m = _map_to_range(x0, box.x_min, box.x_max, 0.0, 2.125)

    # Eq (5)-style magnitude sweep (m/s), then mapped to -y direction.
    v_len_min = 4.0 - y_sweep_m
    v_len_max = 10.0 - 2.5 * y_sweep_m
    if v_len_max < v_len_min:
        v_len_min, v_len_max = v_len_max, v_len_min
    v_len_min = max(0.2, v_len_min)
    v_len_max = max(v_len_min + 0.05, v_len_max)
    vy0 = -1000.0 * float(rng.uniform(v_len_min, v_len_max))

    # Eq (6)-style width sweep (m/s) remapped across x.
    v_wid_min = 0.0 - 2.353 * x_sweep_m
    v_wid_max = 5.0 - 2.353 * x_sweep_m
    if v_wid_max < v_wid_min:
        v_wid_min, v_wid_max = v_wid_max, v_wid_min
    vx0 = 1000.0 * float(rng.uniform(v_wid_min, v_wid_max))

    # Vertical sweep: include many flat/downward cases as requested.
    # Base range follows the paper (-5 to +2 m/s), with bias toward downward.
    if rng.random() < 0.65:
        vz_ms = float(rng.uniform(-5.0, 0.4))
    else:
        vz_ms = float(rng.uniform(-5.0, 2.0))
    vz0 = 1000.0 * vz_ms

    # Optionally reshape launch-velocity distribution for real-matched generation.
    if rm_cfg.enabled:
        vx0 *= float(rm_cfg.vx_scale)
        vy0 *= float(rm_cfg.vy_scale)
        vz0 = vz0 * float(rm_cfg.vz_scale) + float(rm_cfg.vz_bias_mm_s)

        vx_noise_std = float(rm_cfg.vx_noise_std_mm_s)
        vy_noise_std = float(rm_cfg.vy_noise_std_mm_s)
        vz_noise_std = float(rm_cfg.vz_noise_std_mm_s)
    else:
        vx_noise_std = 180.0
        vy_noise_std = 220.0
        vz_noise_std = 180.0

    # Mild randomization to avoid hard-edged synthetic bands.
    vx0 += float(rng.normal(0.0, vx_noise_std))
    vy0 += float(rng.normal(0.0, vy_noise_std))
    vz0 += float(rng.normal(0.0, vz_noise_std))

    # Keep speeds in a realistic numeric range.
    vx0 = float(np.clip(vx0, -5500.0, 5500.0))
    vy0 = float(np.clip(vy0, -10000.0, -150.0))
    vz0 = float(np.clip(vz0, -6000.0, 3000.0))

    return x0, y0, z0, vx0, vy0, vz0


def sample_observation_times(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    t_hit_abs: Optional[float],
    launch_state: Tuple[float, float, float, float, float, float],
    obs_window: ObservationWindowConfig,
    rm_cfg: Optional[RealMatchedConfig] = None,
) -> Optional[np.ndarray]:
    if rm_cfg is not None and rm_cfg.enabled and rm_cfg.dt_mode.lower() == "normal":
        dts = rng.normal(rm_cfg.dt_mean_s, rm_cfg.dt_std_s, size=cfg.k_points - 1)
        dts = np.clip(dts, rm_cfg.dt_min_s, rm_cfg.dt_max_s)
    else:
        dts = rng.uniform(cfg.obs_dt_min_s, cfg.obs_dt_max_s, size=cfg.k_points - 1)
    total_dt = float(np.sum(dts))

    if t_hit_abs is None:
        last_obs_max = cfg.obs_last_time_cap_s
    else:
        last_obs_max = min(cfg.obs_last_time_cap_s, t_hit_abs - cfg.obs_pre_hit_margin_s)

    if total_dt > last_obs_max:
        return None

    x0, y0, _, vx0, vy0, _ = launch_state
    if vy0 >= -1e-6:
        return None

    # Require launch to start beyond the visible upper-y boundary so point1
    # is sampled at FoV entry instead of an already-visible state.
    if y0 <= obs_window.y_max:
        return None

    # Choose observation start so x1..xK stay inside camera FoV in y:
    #   obs_window.y_min <= y(t) <= obs_window.y_max.
    # For vy0<0 and linear y(t)=y0+vy0*t:
    #   t_enter_max = time when y first reaches y_max (FoV entry from far side),
    #   t_exit_min  = time when y reaches y_min (FoV exit toward robot side).
    t_enter_max = (obs_window.y_max - y0) / vy0
    t_exit_min = (obs_window.y_min - y0) / vy0

    # Start point at FoV entry (or as early as possible if already in-window).
    t_start_min = max(0.0, t_enter_max)
    t_start_max = min(last_obs_max - total_dt, t_exit_min - total_dt)
    if t_start_min > t_start_max:
        return None

    # Optionally constrain observed point1 x1/y1 to configured bounds.
    t_range_x = _intersect_time_range_linear_bounds(
        t_start_min,
        t_start_max,
        x0,
        vx0,
        obs_window.x1_min,
        obs_window.x1_max,
    )
    if t_range_x is None:
        return None
    t_start_min, t_start_max = t_range_x

    t_range_y = _intersect_time_range_linear_bounds(
        t_start_min,
        t_start_max,
        y0,
        vy0,
        obs_window.y1_min,
        obs_window.y1_max,
    )
    if t_range_y is None:
        return None
    t_start_min, t_start_max = t_range_y

    start_mode = str(cfg.obs_start_mode).strip().lower() or "entry"
    if rm_cfg is not None and rm_cfg.enabled:
        rm_start_mode = str(rm_cfg.obs_start_mode).strip().lower()
        if rm_start_mode:
            start_mode = rm_start_mode

    if start_mode == "uniform":
        t_start = float(rng.uniform(t_start_min, t_start_max))
    else:
        # Default behavior keeps point1 near FoV entry.
        t_start = t_start_min
    t_obs = t_start + np.concatenate([[0.0], np.cumsum(dts)])
    return t_obs


def sample_positions_at_times(
    t_sim: np.ndarray,
    states_sim: np.ndarray,
    t_obs: np.ndarray,
) -> np.ndarray:
    # Interpolate observed positions from the same bounce-aware simulated path
    # used for intercept labeling to keep inputs/labels physically consistent.
    x = np.interp(t_obs, t_sim, states_sim[:, 0])
    y = np.interp(t_obs, t_sim, states_sim[:, 1])
    z = np.interp(t_obs, t_sim, states_sim[:, 2])
    return np.column_stack([x, y, z])


def inject_observation_noise(
    rng: np.random.Generator,
    points: np.ndarray,
    t_obs: np.ndarray,
    noise: NoiseConfig,
    obs_window: Optional[ObservationWindowConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Per-trajectory systematic camera calibration bias.
    bias_xyz = rng.normal(0.0, noise.bias_std_mm, size=3)
    pos_std_xyz = _resolve_position_std_xyz(noise)
    noisy_points = points + bias_xyz + rng.normal(0.0, pos_std_xyz, size=points.shape)

    # Sparse heavy outliers.
    for i in range(noisy_points.shape[0]):
        if rng.random() < noise.outlier_prob:
            noisy_points[i] += rng.normal(0.0, noise.outlier_std_mm, size=3)

    # Keep observed model inputs inside the camera FoV in y.
    if obs_window is not None and noisy_points.shape[0] > 0:
        noisy_points[:, 1] = np.clip(noisy_points[:, 1], obs_window.y_min, obs_window.y_max)

    # Timestamp jitter, then enforce monotonicity.
    jitter_s = noise.timestamp_jitter_ms / 1000.0
    noisy_t = t_obs + rng.normal(0.0, jitter_s, size=t_obs.shape[0])
    for i in range(1, noisy_t.shape[0]):
        min_allowed = noisy_t[i - 1] + 1e-4
        if noisy_t[i] < min_allowed:
            noisy_t[i] = min_allowed

    return noisy_points, noisy_t


def build_row(
    noisy_points: np.ndarray,
    noisy_t: np.ndarray,
    intercept: Optional[Dict[str, float]],
    launch_state: Tuple[float, float, float, float, float, float],
    gravity_z: float,
) -> Dict[str, float]:
    row: Dict[str, float] = {}

    for i in range(noisy_points.shape[0]):
        row[f"x{i+1}"] = float(noisy_points[i, 0])
        row[f"y{i+1}"] = float(noisy_points[i, 1])
        row[f"z{i+1}"] = float(noisy_points[i, 2])

    dts = np.diff(noisy_t)
    for i, dt in enumerate(dts, start=1):
        row[f"dt{i}{i+1}"] = float(dt)

    if intercept is None:
        row.update(
            {
                "x_hit": math.nan,
                "y_hit": math.nan,
                "z_hit": math.nan,
                "vx_hit": math.nan,
                "vy_hit": math.nan,
                "vz_hit": math.nan,
                "t_hit": math.nan,
                "is_reachable": 0.0,
                "bounces_before_hit": 0.0,
                "has_bounce_before_hit": 0.0,
                "intercept_valid": 0.0,
            }
        )
    else:
        # Define target t_hit consistently as time from last input point (pointK) to intercept.
        # Use pointK timestamp from noisy observation timing (same timing source as dt features).
        t_pk_abs = float(noisy_t[-1])
        t_hit_from_pk = float(intercept["t_hit_abs"] - t_pk_abs)
        row.update(
            {
                "x_hit": float(intercept["x_hit"]),
                "y_hit": float(intercept["y_hit"]),
                "z_hit": float(intercept["z_hit"]),
                "vx_hit": float(intercept["vx_hit"]),
                "vy_hit": float(intercept["vy_hit"]),
                "vz_hit": float(intercept["vz_hit"]),
                "t_hit": t_hit_from_pk,
                "is_reachable": float(intercept["is_reachable"]),
                "bounces_before_hit": float(intercept["bounces_before_hit"]),
                "has_bounce_before_hit": float(intercept["has_bounce_before_hit"]),
            }
        )
        row["intercept_valid"] = 1.0

    x0, y0, z0, vx0, vy0, vz0 = launch_state
    row["x0"] = float(x0)
    row["y0"] = float(y0)
    row["z0"] = float(z0)
    row["vx0"] = float(vx0)
    row["vy0"] = float(vy0)
    row["vz0"] = float(vz0)
    row["gravity_z_mm_s2"] = float(gravity_z)

    return row


def generate_single_sample(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    ws: WorkspaceConfig,
    box: LaunchBoxConfig,
    obs_window: ObservationWindowConfig,
    noise: NoiseConfig,
    rm_cfg: Optional[RealMatchedConfig] = None,
    want_trace: bool = False,
) -> Optional[Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]]:
    if rm_cfg is None:
        rm_cfg = RealMatchedConfig()

    gravity_z = cfg.gravity_z_mm_s2 * (1.0 + rng.normal(0.0, cfg.gravity_jitter_frac_std))
    launch_state = sample_initial_state(rng, box=box, ws=ws, gravity_z=gravity_z, rm_cfg=rm_cfg)
    x0, y0, z0, vx0, vy0, vz0 = launch_state

    max_bounces_use = cfg.max_bounces
    if rm_cfg.enabled and cfg.max_bounces >= 2:
        p_second = float(np.clip(rm_cfg.second_bounce_prob, 0.0, 1.0))
        max_bounces_use = cfg.max_bounces if rng.random() < p_second else 1

    t_scan, states, bounces_scan = simulate_no_loss_trajectory(
        x0=x0,
        y0=y0,
        z0=z0,
        vx0=vx0,
        vy0=vy0,
        vz0=vz0,
        gravity_z=gravity_z,
        duration_s=cfg.scan_duration_s,
        dt_s=cfg.scan_dt_s,
        table_z_mm=cfg.table_z_mm,
        max_bounces=max_bounces_use,
    )

    intercept = select_intercept_like_robot_predictor(
        t=t_scan,
        states=states,
        bounces=bounces_scan,
        ws=ws,
        min_time_hit_s=cfg.min_time_hit_s,
    )

    if rm_cfg.enabled:
        if rm_cfg.require_valid_intercept and intercept is None:
            return None
        if (
            rm_cfg.require_reachable
            and intercept is not None
            and float(intercept.get("is_reachable", 0.0)) <= 0.5
        ):
            return None

    t_hit_abs = intercept["t_hit_abs"] if intercept is not None else None
    t_obs = sample_observation_times(
        rng,
        cfg,
        t_hit_abs=t_hit_abs,
        launch_state=launch_state,
        obs_window=obs_window,
        rm_cfg=rm_cfg,
    )
    if t_obs is None:
        return None

    points_true = sample_positions_at_times(
        t_sim=t_scan,
        states_sim=states,
        t_obs=t_obs,
    )

    # Enforce optional point1 bounds on the clean sampled signal first.
    if points_true.shape[0] > 0 and not _point1_in_constraints(points_true[0], obs_window):
        return None

    noisy_points, noisy_t = inject_observation_noise(
        rng=rng,
        points=points_true,
        t_obs=t_obs,
        noise=noise,
        obs_window=obs_window,
    )

    # Enforce optional point1 bounds again on final observed sample (after noise).
    if noisy_points.shape[0] > 0 and not _point1_in_constraints(noisy_points[0], obs_window):
        return None

    row = build_row(
        noisy_points=noisy_points,
        noisy_t=noisy_t,
        intercept=intercept,
        launch_state=launch_state,
        gravity_z=gravity_z,
    )

    if rm_cfg.enabled and float(row.get("intercept_valid", 0.0)) > 0.5:
        t_hit_val = float(row["t_hit"])
        if rm_cfg.t_hit_min_s is not None and t_hit_val < float(rm_cfg.t_hit_min_s):
            return None
        if rm_cfg.t_hit_max_s is not None and t_hit_val > float(rm_cfg.t_hit_max_s):
            return None

    if not want_trace:
        return row, None

    t_preview, states_preview, bounces_preview = simulate_no_loss_trajectory(
        x0=x0,
        y0=y0,
        z0=z0,
        vx0=vx0,
        vy0=vy0,
        vz0=vz0,
        gravity_z=gravity_z,
        duration_s=cfg.preview_duration_s,
        dt_s=cfg.preview_dt_s,
        table_z_mm=cfg.table_z_mm,
        max_bounces=max_bounces_use,
        stop_y_mm=cfg.preview_stop_y_mm,
        stop_z_mm=cfg.preview_stop_z_mm,
    )

    trace = {
        "t_scan": t_scan,
        "states": states,
        "bounces_scan": bounces_scan,
        "t_preview": t_preview,
        "states_preview": states_preview,
        "bounces_preview": bounces_preview,
        "points_true": points_true,
        "points_noisy": noisy_points,
        "t_obs_true": t_obs,
        "t_obs_noisy": noisy_t,
    }
    return row, trace


def write_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No rows generated; nothing to write.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_examples(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    ws: WorkspaceConfig,
    box: LaunchBoxConfig,
    obs_window: ObservationWindowConfig,
    noise: NoiseConfig,
    rm_cfg: RealMatchedConfig,
    rows: List[Dict[str, float]],
    n_examples: int,
    save_path: Optional[Path] = None,
) -> None:
    if n_examples <= 0:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping plots: {exc}")
        return

    examples: List[Tuple[Dict[str, float], Dict[str, np.ndarray]]] = []
    attempts = 0
    max_attempts = max(200, n_examples * 30)
    while len(examples) < n_examples and attempts < max_attempts:
        attempts += 1
        item = generate_single_sample(
            rng,
            cfg,
            ws,
            box,
            obs_window,
            noise,
            rm_cfg=rm_cfg,
            want_trace=True,
        )
        if item is None:
            continue
        row, trace = item
        if trace is None:
            continue
        examples.append((row, trace))

    if not examples:
        print("[WARN] Could not create preview examples for plotting.")
        return

    # Figure 1: many trajectories overlaid for geometric scope.
    fig_traj = plt.figure(figsize=(11, 8))
    ax3d = fig_traj.add_subplot(111, projection="3d")
    theta = np.linspace(0.0, 2.0 * np.pi, 120)
    wx = ws.ellipse_a * np.cos(theta)
    wy = ws.ellipse_b * np.sin(theta)
    ax3d.plot(wx, wy, np.full_like(theta, ws.z_min), color="gray", alpha=0.8, linewidth=1.0)
    ax3d.plot(wx, wy, np.full_like(theta, ws.z_max), color="gray", alpha=0.8, linewidth=1.0)
    ax3d.plot(wx, wy, np.full_like(theta, cfg.table_z_mm), color="tab:brown", alpha=0.7, linewidth=1.2)
    # Table top outline: z fixed at table height, x centered on 0, y starts at 0.
    half_table_w = 0.5 * cfg.table_width_mm
    table_x = np.array(
        [-half_table_w, half_table_w, half_table_w, -half_table_w, -half_table_w],
        dtype=float,
    )
    table_y = np.array([0.0, 0.0, cfg.table_length_mm, cfg.table_length_mm, 0.0], dtype=float)
    table_z = np.full_like(table_x, cfg.table_z_mm)
    ax3d.plot(table_x, table_y, table_z, color="tab:green", linewidth=2.0, label="table outline")
    # Centerline (x=0) for orientation.
    ax3d.plot(
        [0.0, 0.0],
        [0.0, cfg.table_length_mm],
        [cfg.table_z_mm, cfg.table_z_mm],
        color="tab:green",
        linewidth=1.0,
        linestyle="--",
    )
    # Orientation guides (not rotated): x-major radius and y-minor radius.
    ax3d.plot(
        [-ws.ellipse_a, ws.ellipse_a],
        [0.0, 0.0],
        [cfg.table_z_mm, cfg.table_z_mm],
        color="tab:red",
        linewidth=1.3,
        label="x radius (720)",
    )
    ax3d.plot(
        [0.0, 0.0],
        [-ws.ellipse_b, ws.ellipse_b],
        [cfg.table_z_mm, cfg.table_z_mm],
        color="tab:blue",
        linewidth=1.3,
        label="y radius (470)",
    )

    for idx, (row, trace) in enumerate(examples):
        states = trace.get("states_preview", trace["states"])
        pts_noisy = trace["points_noisy"]
        ax3d.plot(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            alpha=0.35,
            linewidth=1.2,
            color=("tab:blue" if idx % 2 == 0 else "tab:purple"),
        )
        ax3d.scatter(
            pts_noisy[:, 0],
            pts_noisy[:, 1],
            pts_noisy[:, 2],
            s=10,
            alpha=0.5,
            color="black",
        )
        if row["intercept_valid"] > 0.5:
            ic = "tab:green" if row["is_reachable"] > 0.5 else "tab:orange"
            ax3d.scatter([row["x_hit"]], [row["y_hit"]], [row["z_hit"]], s=26, color=ic)

    # Auto-scale 3D bounds from actual trajectory content so the plot is readable.
    xyz_blocks: List[np.ndarray] = []
    for row, trace in examples:
        st = trace.get("states_preview", trace["states"])
        if st.size > 0:
            xyz_blocks.append(st[:, :3])
        pn = trace.get("points_noisy")
        if pn is not None and pn.size > 0:
            xyz_blocks.append(pn[:, :3])
        if row.get("intercept_valid", 0.0) > 0.5:
            xyz_blocks.append(np.array([[row["x_hit"], row["y_hit"], row["z_hit"]]], dtype=float))

    if xyz_blocks:
        xyz_blocks.append(
            np.column_stack(
                [
                    table_x,
                    table_y,
                    table_z,
                ]
            )
        )
        xyz_all = np.concatenate(xyz_blocks, axis=0)
        xyz_min = np.min(xyz_all, axis=0)
        xyz_max = np.max(xyz_all, axis=0)
        span = np.maximum(xyz_max - xyz_min, 1.0)
        margin = 0.08 * span + 25.0

        x_lo, y_lo, z_lo = xyz_min - margin
        x_hi, y_hi, z_hi = xyz_max + margin
        ax3d.set_xlim(float(x_lo), float(x_hi))
        ax3d.set_ylim(float(y_lo), float(y_hi))
        ax3d.set_zlim(float(z_lo), float(z_hi))

        x_span = float(x_hi - x_lo)
        y_span = float(y_hi - y_lo)
        z_span = float(z_hi - z_lo) * max(0.8, cfg.preview_z_exaggeration)
        ax3d.set_box_aspect((x_span, y_span, z_span))

    ax3d.scatter(
        [ws.robot_home[0]],
        [ws.robot_home[1]],
        [ws.robot_home[2]],
        marker="*",
        s=140,
        color="red",
        label="workspace center",
    )
    ax3d.set_title(f"Trajectory Scope Preview ({len(examples)} random trajectories)")
    ax3d.set_xlabel("x [mm]")
    ax3d.set_ylabel("y [mm]")
    ax3d.set_zlabel("z [mm]")
    ax3d.view_init(elev=26, azim=-58)
    ax3d.legend(loc="best")
    fig_traj.tight_layout()

    # Figure 2: dataset-level distributions from generated rows.
    def col(name: str, default: float = math.nan) -> np.ndarray:
        return np.asarray([float(r.get(name, default)) for r in rows], dtype=float)

    x1 = col("x1")
    y1 = col("y1")
    z1 = col("z1")
    x_hit = col("x_hit")
    y_hit = col("y_hit")
    t_hit = col("t_hit")
    vy0 = col("vy0")
    is_reachable = col("is_reachable", 0.0) > 0.5
    valid_hit = col("intercept_valid", 0.0) > 0.5

    fig_scope, axs = plt.subplots(2, 3, figsize=(15, 9))
    ax = axs[0, 0]
    ax.scatter(x1, y1, s=6, alpha=0.35, color="tab:blue")
    ax.set_title("Observed Point1 Scope (x1 vs y1)")
    ax.set_xlabel("x1 [mm]")
    ax.set_ylabel("y1 [mm]")
    ax.set_xlim(box.x_min - 100, box.x_max + 100)
    ax.set_ylim(obs_window.y_min - 100, obs_window.y_max + 100)

    ax = axs[0, 1]
    ax.scatter(y1, z1, s=6, alpha=0.35, color="tab:purple")
    ax.set_title("Observed Point1 Scope (y1 vs z1)")
    ax.set_xlabel("y1 [mm]")
    ax.set_ylabel("z1 [mm]")
    ax.set_xlim(obs_window.y_min - 100, obs_window.y_max + 100)
    ax.set_ylim(box.z_min - 100, box.z_max + 100)

    ax = axs[0, 2]
    if np.any(valid_hit):
        ax.scatter(
            x_hit[valid_hit & is_reachable],
            y_hit[valid_hit & is_reachable],
            s=8,
            alpha=0.5,
            label="reachable",
        )
    th = np.linspace(0, 2 * np.pi, 180)
    ax.plot(ws.ellipse_a * np.cos(th), ws.ellipse_b * np.sin(th), color="black", linewidth=1.2, label="workspace XY")
    # Axis guides verify orientation: X radius = 720 (horizontal), Y radius = 470 (vertical).
    ax.plot([-ws.ellipse_a, ws.ellipse_a], [0.0, 0.0], color="tab:red", linewidth=1.0, alpha=0.8, label="x-axis radius")
    ax.plot([0.0, 0.0], [-ws.ellipse_b, ws.ellipse_b], color="tab:blue", linewidth=1.0, alpha=0.8, label="y-axis radius")
    ax.set_title("Hit Scope (x_hit vs y_hit)")
    ax.set_xlabel("x_hit [mm]")
    ax.set_ylabel("y_hit [mm]")
    ax.set_xlim(-ws.ellipse_a * 1.1, ws.ellipse_a * 1.1)
    ax.set_ylim(-ws.ellipse_b * 1.1, ws.ellipse_b * 1.1)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    ax = axs[1, 0]
    if np.any(valid_hit):
        ax.hist(t_hit[valid_hit], bins=30, color="tab:green", alpha=0.75)
    ax.set_title("t_hit Distribution")
    ax.set_xlabel("t_hit [s]")
    ax.set_ylabel("count")

    ax = axs[1, 1]
    ax.hist(vy0, bins=30, color="tab:orange", alpha=0.75)
    ax.set_title("Initial vy0 Distribution")
    ax.set_xlabel("vy0 [mm/s]")
    ax.set_ylabel("count")

    ax = axs[1, 2]
    n_total = float(len(rows))
    n_reach = float(np.sum(is_reachable))
    n_invalid = float(np.sum(~valid_hit))
    n_bounce = float(np.sum(col("has_bounce_before_hit", 0.0) > 0.5))
    bars = ["reachable", "invalid", "bounce_before_hit"]
    vals = [n_reach, n_invalid, n_bounce]
    ax.bar(bars, vals, color=["tab:green", "tab:gray", "tab:blue"], alpha=0.8)
    ax.set_title("Label/Bounce Mix")
    ax.set_ylabel("count")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01 * max(1.0, n_total), f"{100.0 * v / max(1.0, n_total):.1f}%", ha="center", fontsize=8)

    fig_scope.tight_layout()

    scope_save_path: Optional[Path] = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig_traj.savefig(save_path, dpi=160)
        scope_save_path = save_path.with_name(f"{save_path.stem}_scope{save_path.suffix}")
        fig_scope.savefig(scope_save_path, dpi=160)
        print(f"[INFO] Saved trajectory preview to: {save_path}")
        print(f"[INFO] Saved dataset-scope preview to: {scope_save_path}")

    backend = plt.get_backend().lower()
    try:
        from matplotlib.backends import BackendFilter, backend_registry

        interactive_backends = {
            b.lower() for b in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
    except Exception:
        # Compatibility fallback for older matplotlib versions.
        interactive_backends = {
            "gtk3agg",
            "gtk3cairo",
            "gtk4agg",
            "gtk4cairo",
            "macosx",
            "nbagg",
            "notebook",
            "qtagg",
            "qt5agg",
            "qt5cairo",
            "tkagg",
            "tkcairo",
            "webagg",
            "wx",
            "wxagg",
            "wxcairo",
        }
    if backend in interactive_backends:
        plt.show()
    else:
        print(
            "[INFO] Matplotlib backend is non-interactive "
            f"('{plt.get_backend()}'); skipping on-screen plot display."
        )
        plt.close(fig_traj)
        plt.close(fig_scope)

    print("\nPreview examples:")
    max_print = min(12, len(examples))
    for idx, (row, _) in enumerate(examples[:max_print], start=1):
        print(
            f"  Example {idx}: "
            f"x1,y1,z1=({row['x1']:.1f},{row['y1']:.1f},{row['z1']:.1f}), "
            f"x2,y2,z2=({row['x2']:.1f},{row['y2']:.1f},{row['z2']:.1f}), "
            f"is_reachable={int(row['is_reachable'])}, "
            f"bounces={int(row.get('bounces_before_hit', 0.0))}, "
            f"t_hit={row['t_hit'] if row['intercept_valid'] > 0.5 else float('nan'):.4f}"
        )
    if len(examples) > max_print:
        print(f"  ... {len(examples) - max_print} more examples not printed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic transfer-learning data for ping pong interception."
    )
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument(
        "--preset",
        type=str,
        default="broad",
        help="Preset name from YAML (e.g., broad, real_matched).",
    )
    parser.add_argument(
        "--presets-yaml",
        type=str,
        default=str(DEFAULT_PRESETS_YAML),
        help="Path to presets YAML file.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Root folder where timestamped run folders are created.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional short tag appended to run folder name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional explicit CSV output path. Empty -> timestamped run folder output.",
    )

    # Optional overrides (if provided, override the selected preset).
    parser.add_argument("--scan-duration", type=float, default=None, help="Override: trajectory scan duration [s].")
    parser.add_argument("--scan-dt", type=float, default=None, help="Override: trajectory scan step [s].")
    parser.add_argument("--min-time-hit", type=float, default=None, help="Override: minimum intercept time [s].")
    parser.add_argument("--table-z", type=float, default=None, help="Override: table plane Z in robot frame [mm].")
    parser.add_argument("--table-length", type=float, default=None, help="Override: table length along +y [mm].")
    parser.add_argument("--table-width", type=float, default=None, help="Override: table width across x [mm].")
    parser.add_argument(
        "--max-bounces",
        type=int,
        default=None,
        help="Override: maximum no-loss bounces to simulate.",
    )
    parser.add_argument(
        "--preview-duration",
        type=float,
        default=None,
        help="Override: preview trajectory max duration [s] (visualization only).",
    )
    parser.add_argument(
        "--preview-z-exaggeration",
        type=float,
        default=None,
        help="Override: vertical exaggeration factor for 3D preview readability.",
    )
    parser.add_argument(
        "--pos-noise-std",
        type=float,
        default=None,
        help="Override: isotropic Gaussian position noise std [mm] (also clears axis-wise std overrides).",
    )
    parser.add_argument("--bias-noise-std", type=float, default=None, help="Override: per-trajectory bias std [mm].")
    parser.add_argument("--outlier-prob", type=float, default=None, help="Override: outlier probability per point.")
    parser.add_argument("--outlier-std", type=float, default=None, help="Override: outlier noise std [mm].")
    parser.add_argument("--time-jitter-ms", type=float, default=None, help="Override: timestamp jitter std [ms].")
    parser.add_argument(
        "--gravity-jitter-frac-std",
        type=float,
        default=None,
        help="Override: std of multiplicative gravity randomization fraction.",
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=-1,
        help=(
            "How many random trajectories to draw in preview. "
            "-1 means auto (more coverage), 0 disables."
        ),
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="Optional explicit image path to save sanity-check plot.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    presets = load_presets_yaml(Path(args.presets_yaml))
    preset_bundle = resolve_preset_bundle(presets, args.preset)

    cfg = _build_dataclass(GeneratorConfig, preset_bundle.get("generator", {}))
    noise = _build_dataclass(NoiseConfig, preset_bundle.get("noise", {}))
    ws = _build_dataclass(WorkspaceConfig, preset_bundle.get("workspace", {}))
    box = _build_dataclass(LaunchBoxConfig, preset_bundle.get("launch_box", {}))
    obs_window = _build_dataclass(ObservationWindowConfig, preset_bundle.get("observation_window", {}))
    rm_cfg = _build_dataclass(RealMatchedConfig, preset_bundle.get("real_match", {}))

    # Top-level per-run overrides
    cfg = replace(cfg, n_samples=args.num_samples)
    if args.scan_duration is not None:
        cfg = replace(cfg, scan_duration_s=float(args.scan_duration))
    if args.scan_dt is not None:
        cfg = replace(cfg, scan_dt_s=float(args.scan_dt))
    if args.min_time_hit is not None:
        cfg = replace(cfg, min_time_hit_s=float(args.min_time_hit))
    if args.gravity_jitter_frac_std is not None:
        cfg = replace(cfg, gravity_jitter_frac_std=float(args.gravity_jitter_frac_std))
    if args.preview_duration is not None:
        cfg = replace(cfg, preview_duration_s=float(args.preview_duration))
    if args.preview_z_exaggeration is not None:
        cfg = replace(cfg, preview_z_exaggeration=max(0.5, float(args.preview_z_exaggeration)))
    if args.table_z is not None:
        cfg = replace(cfg, table_z_mm=float(args.table_z))
    if args.table_length is not None:
        cfg = replace(cfg, table_length_mm=float(args.table_length))
    if args.table_width is not None:
        cfg = replace(cfg, table_width_mm=float(args.table_width))
    if args.max_bounces is not None:
        cfg = replace(cfg, max_bounces=max(0, int(args.max_bounces)))

    if args.pos_noise_std is not None:
        noise = replace(
            noise,
            position_std_mm=float(args.pos_noise_std),
            position_std_x_mm=None,
            position_std_y_mm=None,
            position_std_z_mm=None,
        )
    if args.bias_noise_std is not None:
        noise = replace(noise, bias_std_mm=float(args.bias_noise_std))
    if args.outlier_prob is not None:
        noise = replace(noise, outlier_prob=float(args.outlier_prob))
    if args.outlier_std is not None:
        noise = replace(noise, outlier_std_mm=float(args.outlier_std))
    if args.time_jitter_ms is not None:
        noise = replace(noise, timestamp_jitter_ms=float(args.time_jitter_ms))

    runs_root = Path(args.runs_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = sanitize_tag(args.run_tag)
    run_name = f"{timestamp}_{args.preset}" + (f"_{tag}" if tag else "")
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else run_dir / "synthetic_intercept_dataset.csv"
    resolved_cfg_path = run_dir / "resolved_config.yaml"

    rows: List[Dict[str, float]] = []
    max_attempts = cfg.n_samples * cfg.max_attempts_factor
    attempts = 0
    target_reach_n: Optional[int] = None
    target_unreach_n: Optional[int] = None
    accepted_reach_n = 0
    accepted_unreach_n = 0

    if cfg.reachable_target_frac is not None:
        frac = float(np.clip(cfg.reachable_target_frac, 0.0, 1.0))
        target_reach_n = int(round(cfg.n_samples * frac))
        target_unreach_n = cfg.n_samples - target_reach_n

    while len(rows) < cfg.n_samples and attempts < max_attempts:
        attempts += 1
        item = generate_single_sample(
            rng,
            cfg,
            ws,
            box,
            obs_window,
            noise,
            rm_cfg=rm_cfg,
            want_trace=False,
        )
        if item is None:
            continue
        row, _ = item

        if target_reach_n is not None and target_unreach_n is not None:
            is_reach = float(row.get("is_reachable", 0.0)) > 0.5
            if is_reach:
                if accepted_reach_n >= target_reach_n:
                    continue
            else:
                if accepted_unreach_n >= target_unreach_n:
                    continue

            rows.append(row)
            if is_reach:
                accepted_reach_n += 1
            else:
                accepted_unreach_n += 1
            continue

        rows.append(row)

    if len(rows) < cfg.n_samples:
        extra = ""
        if target_reach_n is not None and target_unreach_n is not None:
            extra = (
                f" Reachable accepted {accepted_reach_n}/{target_reach_n}, "
                f"unreachable accepted {accepted_unreach_n}/{target_unreach_n}."
            )
        raise RuntimeError(
            f"Only generated {len(rows)} / {cfg.n_samples} rows after {attempts} attempts. "
            "Try loosening generation ranges or constraints." + extra
        )

    write_csv(rows, output_path)

    resolved = {
        "timestamp": timestamp,
        "preset": args.preset,
        "seed": int(args.seed),
        "presets_yaml": str(Path(args.presets_yaml).resolve()),
        "generator": asdict(cfg),
        "noise": asdict(noise),
        "workspace": asdict(ws),
        "launch_box": asdict(box),
        "observation_window": asdict(obs_window),
        "real_match": asdict(rm_cfg),
        "output_csv": str(output_path.resolve()),
    }
    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    reachable = sum(1 for r in rows if r["is_reachable"] > 0.5)
    invalid = sum(1 for r in rows if r["intercept_valid"] < 0.5)
    bounced = sum(1 for r in rows if r.get("has_bounce_before_hit", 0.0) > 0.5)

    print(f"[INFO] Preset: {args.preset}")
    print(f"[INFO] Run folder: {run_dir}")
    print(f"[INFO] Generated {len(rows)} samples -> {output_path}")
    print(f"[INFO] Saved resolved config -> {resolved_cfg_path}")
    print(
        "[INFO] Label mix: "
        f"reachable={reachable} ({100.0*reachable/len(rows):.1f}%), "
        f"invalid={invalid} ({100.0*invalid/len(rows):.1f}%), "
        f"bounce_before_hit={bounced} ({100.0*bounced/len(rows):.1f}%)"
    )
    if cfg.reachable_target_frac is not None:
        print(
            "[INFO] Reachable target: "
            f"{100.0*float(np.clip(cfg.reachable_target_frac, 0.0, 1.0)):.1f}%"
        )

    reach_rows = [r for r in rows if r["intercept_valid"] > 0.5 and r["is_reachable"] > 0.5]
    if reach_rows:
        max_abs_x = max(abs(float(r["x_hit"])) for r in reach_rows)
        max_abs_y = max(abs(float(r["y_hit"])) for r in reach_rows)
        print(
            "[INFO] Reachable hit bounds check: "
            f"max|x_hit|={max_abs_x:.1f} (<= {ws.ellipse_a:.1f}), "
            f"max|y_hit|={max_abs_y:.1f} (<= {ws.ellipse_b:.1f})"
        )

    if args.show_examples < 0:
        n_examples = int(min(24, max(8, args.num_samples // 25)))
    else:
        n_examples = max(0, int(args.show_examples))

    if args.save_plot:
        save_plot_path = Path(args.save_plot)
    else:
        save_plot_path = (run_dir / "sanity_preview.png") if n_examples > 0 else None

    plot_examples(
        rng=rng,
        cfg=cfg,
        ws=ws,
        box=box,
        obs_window=obs_window,
        noise=noise,
        rm_cfg=rm_cfg,
        rows=rows,
        n_examples=n_examples,
        save_path=save_plot_path,
    )


if __name__ == "__main__":
    main()
