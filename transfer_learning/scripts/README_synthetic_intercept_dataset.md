# `synthetic_dataset_generator.py` README

## Purpose
Generate synthetic interception data with selectable YAML presets:
- `broad` (legacy broad distribution)
- `real_matched` (narrowed to resemble real captures)

## Presets YAML
File:
- `scripts/synthetic_data_generated/synthetic_generator_presets.yaml`

Run selection:
```powershell
python .\scripts\synthetic_dataset_generator.py --preset broad
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
