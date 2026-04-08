# `collect_real_transfer_data.py` README

## Purpose
Capture real transfer-learning samples from the live stereo pipeline in continuous mode.

Each saved sample contains:
- first `N` accepted robot-frame ball points (`N = --num-points`, default 6)
- observed stereo point closest to workspace (selected after throw ends)
- labels formatted to match the synthetic dataset convention

## Key Definition
`t_hit` is defined as **time from the last input point (pointN) to selected observed hit point** (seconds).

Internally:
- script selects an observed hit point timestamp `t_hit_abs`
- and stores:
`t_hit = t_hit_abs - pointN_timestamp`

This makes real-data `t_hit` consistent with synthetic data.

## Data Flow Assumptions
- Uses `new_top_level` stack modules:
  - `ball_tracking.stereo_triangulator`
  - `comm_functions.points_based_transform`
  - `config.camera_config`
  - `estimation.workspace`
- Camera IDs + frame size come from:
  - `new_top_level/camera_params/calibration_settings.yaml`
- Camera->robot transform comes from points-based transform file in `new_top_level`.

## Capture Workflow
1. Run script.
2. Warmup background model.
3. Press `s` to start/arm capture.
4. Throw one ball.
5. Script appends one row to the dataset CSV.
6. Script auto-rearms for the next throw after a short cooldown.

Keys:
- `s`: start/arm capture
- `r`: stop current throw and reset buffers (stays armed)
- `x` or `p`: pause/disarm
- `d`: delete last saved sample row from CSV
- `v`: plot last saved sample (3D + plane views `XY/XZ/YZ` with `p1..pN`, `x_hit/y_hit/z_hit`, table, workspace)
- `b`: relearn background
- `q`: quit

## Acceptance Logic (Per Throw)
- Triangulated 3D point must exist (`found_3d`).
- Reprojection error must be <= `--reproj-max`.
- If both checks pass, the point is kept in this throw's observed trajectory.
- The first `N` accepted points are recorded as features (`--num-points`, default 6).
- Throw continues until tracking is lost (or timeout).
- At throw end, script picks the observed robot-frame point (from accepted points, `t>=pointN`)
  using the same rule as synthetic generation:
  - choose in-workspace point closest to workspace center, if any
  - otherwise choose overall point closest to workspace center
- That observed point becomes `x_hit/y_hit/z_hit`.
- `vx_hit/vy_hit/vz_hit` are estimated by local time-fit around selected point.
- Sample is saved and capture automatically continues for next throw.

## Output Files
- Single dataset CSV (Excel-friendly):
  - `real_data_collected/real_transfer_dataset.csv` (default)
  - one row per throw, appended online
- Per-sample preview plots (when `v` is pressed):
  - `real_data_collected/sample_plots/sample_<id>_<timestamp>_3d.png`
  - `real_data_collected/sample_plots/sample_<id>_<timestamp>_planes.png`

## Output Columns (Per Sample)
- Inputs:
  - `x1 y1 z1 ... xN yN zN` (`N = --num-points`)
  - `dt12 ... dt(N-1)N`
- Targets:
  - `x_hit y_hit z_hit vx_hit vy_hit vz_hit`
  - `t_hit` (from point `N`)
  - `is_reachable`
  - `intercept_valid`
  - `bounces_before_hit`
- Metadata:
  - `sample_id`
  - `captured_at`

## Typical Usage
```powershell
python .\collect_real_transfer_data.py --stack-root ..\new_top_level --output-dir .\real_data_collected --output-file real_transfer_dataset.csv
```

For 6-point inputs explicitly:
```powershell
python .\collect_real_transfer_data.py --stack-root ..\new_top_level --output-dir .\real_data_collected --output-file real_transfer_dataset.csv --num-points 6
```

Recommended tuning during capture:
- keep reprojection strict enough for quality:
```powershell
python .\collect_real_transfer_data.py --reproj-max 5 --cooldown-s 0.35
```

If far-away tracking drops, relax detection/triangulation gates:
```powershell
python .\collect_real_transfer_data.py --num-points 6 --det-min-area 25 --tri-min-disparity 2 --tri-max-reproj 12 --tri-max-z 700
```

The overlay now shows likely rejection causes:
- `Tri reject: ...` for triangulation filter failures (`low_disp`, `reproj`, `z_far`, etc.)
- `Det reject L:... R:...` for detector contour filtering (`SMALL`, `SHAPE`, etc.)
