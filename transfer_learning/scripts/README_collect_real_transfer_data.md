# `collect_real_transfer_data.py` README

## Purpose
Capture real transfer-learning samples from the live stereo pipeline in continuous mode.

Each saved sample contains:
- first 4 accepted robot-frame ball points
- intercept prediction from `RobotPredictor`
- labels formatted to match the synthetic dataset convention

## Key Definition
`t_hit` is defined as **time from point 4 to intercept** (seconds).

Internally:
- predictor returns `intercept["time"]` as time-from-prediction-now
- script converts it before saving:

`t_hit = (prediction_timestamp + intercept["time"]) - point4_timestamp`

This makes real-data `t_hit` consistent with synthetic data.

## Data Flow Assumptions
- Uses `new_top_level` stack modules:
  - `ball_tracking.stereo_triangulator`
  - `comm_functions.points_based_transform`
  - `config.camera_config`
  - `estimation.robot_predictor`
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
- `x`: pause/disarm
- `b`: relearn background
- `q`: quit

## Acceptance Logic (Per Throw)
- Triangulated 3D point must exist (`found_3d`).
- Reprojection error must be <= `--reproj-max`.
- Point must be accepted by predictor (`add_position`).
- First 4 accepted points are recorded as features.
- After predictor is ready, first valid in-workspace non-clamped intercept is used as label.
- Then sample is saved and capture automatically continues for next throw.

## Output Files
- Single dataset CSV (Excel-friendly):
  - `real_transfer_data/real_transfer_dataset.csv` (default)
  - one row per throw, appended online

## Output Columns (Per Sample)
- Inputs:
  - `x1 y1 z1 ... x4 y4 z4`
  - `dt12 dt23 dt34`
- Targets:
  - `x_hit y_hit z_hit vx_hit vy_hit vz_hit`
  - `t_hit` (from point 4)
  - `is_reachable`
  - `intercept_valid`
  - `bounces_before_hit`
- Metadata:
  - `sample_id`
  - `captured_at`

## Typical Usage
```powershell
python .\collect_real_transfer_data.py --stack-root ..\new_top_level --output-dir .\real_transfer_data --output-file real_transfer_dataset.csv
```

Recommended tuning during capture:
- keep reprojection strict enough for quality:
```powershell
python .\collect_real_transfer_data.py --reproj-max 5 --cooldown-s 0.35
```
