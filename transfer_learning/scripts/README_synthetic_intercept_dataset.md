# `synthetic_dataset_generator.py` README

## Purpose
Generate synthetic interception data with selectable YAML presets:
- `broad` (legacy broad distribution)
- `clean_broad` (broad coverage with near-zero measurement/model noise)
- `real_matched` (narrowed to resemble real captures)

## Presets YAML
File:
- `scripts/synthetic_data_generated/synthetic_generator_presets.yaml`

Run selection:
```powershell
python .\scripts\synthetic_dataset_generator.py --preset broad
python .\scripts\synthetic_dataset_generator.py --preset clean_broad
python .\scripts\synthetic_dataset_generator.py --preset real_matched
```

`broad` is configured to keep prior behavior (same physics and default noise profile as before preset support).

## Output Organization
By default, every run is saved into a timestamped folder:
- `scripts/synthetic_data_generated/runs/<timestamp>_<preset>[_tag]/`

Each run folder contains:
- `synthetic_intercept_dataset.csv`
- `resolved_config.yaml`
- `sanity_preview.png` (+ `_scope` image when previews are enabled)

Optional tag:
```powershell
python .\scripts\synthetic_dataset_generator.py --preset real_matched --run-tag transfer_stage
```

## Key Label Definition
`t_hit` is always saved as:
- **time from point 4 to intercept** (seconds)

Label fallback behavior:
- Primary: closest in-workspace trajectory point to workspace center.
- If no in-workspace point exists: closest trajectory point to workspace center anyway
  (saved with `is_reachable=0`).
- This keeps labels dense (no NaN hit targets in normal runs).

## Useful Flags
- `--presets-yaml <path>`: use a different presets file
- `--runs-root <path>`: change root folder for timestamped runs
- `--output <csv_path>`: override default timestamped CSV path
- `--save-plot <img_path>`: override default timestamped plot path

All legacy numeric flags are still available as per-run overrides to preset values
(for example `--pos-noise-std`, `--scan-duration`, `--max-bounces`, etc.).

## Real-Matched Velocity Shaping
`real_match` now supports optional launch-velocity shaping knobs in the preset YAML:
- `vx_scale`, `vy_scale`, `vz_scale`
- `vz_bias_mm_s`
- `vx_noise_std_mm_s`, `vy_noise_std_mm_s`, `vz_noise_std_mm_s`

These are only applied when `real_match.enabled: true`, and help align synthetic
`vx_hit/vy_hit/vz_hit/t_hit` distributions to real captures while keeping tunable variance.

Additional real-matching knobs:
- `second_bounce_prob`: when `max_bounces >= 2`, probabilistically allow the second bounce.
- `t_hit_min_s`, `t_hit_max_s`: optional filter on label `t_hit` (time from point 4 to intercept).
