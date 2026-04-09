# Latency Optimizations

## Baseline: 40-45 FPS

At 800x600 MJPG with two ArduCam OV9782 cameras, the pipeline was bottlenecked by serial processing of left and right frames.

## Optimizations Applied

### 1. Parallel Ball Detection (~8-12ms saved)
**File:** `tracking/stereo_triangulator.py`

Left and right ball detection (MOG2 + morphology + contour scoring) was running serially — one after the other. Each takes ~10ms, so 20ms total.

Changed to run both in parallel using the existing `ThreadPoolExecutor(max_workers=2)`. OpenCV releases the GIL during MOG2, morphology, and contour operations, so both threads run on separate CPU cores simultaneously. Each detector has its own MOG2 instance — no shared state.

Result: Detection drops from ~20ms to ~10ms.

### 2. Parallel MJPG Decode (~4-6ms saved)
**File:** `tracking/stereo_triangulator.py`

The `retrieve()` call decompresses MJPG frames from each camera. Was sequential (~5-8ms each). Now parallel using the same thread pool.

Note: `grab()` remains sequential — that's required for stereo synchronization. Only decompression is parallelized.

### 3. Display Throttling (~5-9ms saved on average)
**File:** `scripts/test_integration_simple.py`

Visualization (draw_results + resize + putText + imshow) costs ~7-10ms per frame. Now only runs every 5th frame (~12fps display). The full tracking/prediction/UART pipeline still executes on every frame.

Key: `cv2.waitKey(1)` still runs every frame so keyboard input stays responsive.

### 4. Confidence Scoring Disabled (~2-3ms saved)
**File:** `trajectory/robot_predictor.py`

The confidence sampling (4 trajectory scans from KF covariance) was log-only — it never rejected predictions. Disabled the `_compute_confidence()` call entirely. The method is still in the code for future use. Result dict still returns default confidence/spread/hit_ratio values so all logging works unchanged.

### 5. Reduced Confidence Samples 8 to 4 (applied before disabling)
**File:** `trajectory/robot_predictor.py`

Before fully disabling confidence, reduced samples from 8 to 4 (1200 fewer Euler steps per prediction).

## Result: ~60 FPS

| Optimization | Saved per frame |
|-------------|----------------|
| Parallel detection | 8-12ms |
| Parallel MJPG decode | 4-6ms |
| Display throttling (every 5th) | 5-9ms avg |
| Confidence disabled | 2-3ms |
| **Total saved** | **~19-30ms** |

## Possible Future Optimizations (not yet applied)

- Reduce display resolution (640 -> 320px)
- Skip display entirely during active throws
- Reduce SCAN_DT from 5ms to 10ms (150 vs 300 Euler steps)
- Reduce frame resolution from 800x600 to 640x480 (fewer pixels to decode/process)
