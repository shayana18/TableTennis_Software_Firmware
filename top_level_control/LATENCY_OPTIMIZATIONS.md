# Latency Optimizations — March 24, 2026

## Goal
Reduce per-frame processing time to maximize FPS and minimize latency between ball detection and intercept send. Cameras can do 90fps triggered.

---

## Change 1: Enable Threaded Detection (~5ms saved)

**File:** `tracking/stereo_triangulator.py`
**What:** Uncomment the parallel detection code (ThreadPoolExecutor). Left and right camera detection run simultaneously on separate CPU cores since OpenCV releases the GIL.
**Revert:** Comment out the parallel lines, uncomment the sequential lines.

## Change 2: Skip Visualization During Active Throw (~5-8ms saved)

**File:** `scripts/test_integration_simple.py`
**What:** When gate is ON and a throw is active (predictor has positions), skip `draw_results()`, `cv2.resize()`, `cv2.hconcat()`, `cv2.imshow()`. Only do minimal `cv2.pollKey()` for keyboard input. Show full visualization only when gate is OFF or between throws.
**Revert:** Remove the `if` guard around visualization code.

## Change 3: Replace `cv2.waitKey(1)` with `cv2.pollKey()` (~1ms saved)

**File:** `scripts/test_integration_simple.py`
**What:** `waitKey(1)` blocks for at least 1ms. `pollKey()` returns immediately. At 90fps, this saves 90ms/second.
**Revert:** Change `cv2.pollKey()` back to `cv2.waitKey(1) & 0xFF`.
**Note:** `pollKey()` returns -1 if no key pressed, same as `waitKey`. The `& 0xFF` mask is still needed.

## Change 4: Pre-compute R*scale Matrix (~0.1ms saved, cleaner)

**File:** `scripts/test_integration_simple.py`
**What:** Instead of `R @ (cam_pt * scale) + t` every frame, pre-compute `R_scaled = R * scale` once at startup. Then each frame is just `R_scaled @ cam_pt + t`.
**Revert:** Use original `cam_to_robot(R, t, scale, ...)` call.

---

## Expected Impact

| Change | Savings | Cumulative |
|--------|---------|------------|
| Threaded detection | ~5ms | 5ms |
| Skip visualization | ~5-8ms | 10-13ms |
| pollKey vs waitKey | ~1ms | 11-14ms |
| Pre-compute R*scale | ~0.1ms | 11-14ms |

At 90fps, frame interval is 11.1ms. With these changes, processing should complete well within one frame interval, meaning the system processes every frame with zero backlog.

## Before/After

| Metric | Before | After (expected) |
|--------|--------|-----------------|
| Processing per frame | ~16-20ms | ~5-7ms |
| Effective FPS | ~50-60 | ~85-90 (camera-limited) |
| Detection-to-send latency | ~20ms | ~8-10ms |
