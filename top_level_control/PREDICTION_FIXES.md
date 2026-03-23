# Prediction Pipeline — Proposed Fixes (Revised)

## Current Performance
- 19 throws tested, only 6 produced predictions (32% hit rate)
- Mean prediction error: ~300-450mm
- Best throw: ~84mm error (proves system CAN work)
- Static transform accuracy: 43mm RMS
- VY accuracy: good (~9% error). VZ accuracy: poor (~1100 mm/s noise)
- Most predictions are clamped (ball doesn't enter workspace), not direct hits

---

## ~~FIX 1: Gravity Vector Decomposition~~ — DEBUNKED

**Status:** NOT NEEDED. Robot frame Z is truly vertical (defined by physical measurements relative to robot). The R matrix handles camera pitch correctly. Gravity IS purely along robot Z. The earlier analysis was wrong — it confused camera optical axes with robot axes.

---

## FIX 2: Increase KF min_updates from 4 to 8 (HIGH)

**Problem:** `min_updates=4` means the KF reports velocity as "ready" after just 4 measurements (~100ms). Velocity from 4 noisy points is unreliable. The best throw (84mm error) used 34 frames with excellent velocity alignment. Worst throws had velocity vectors misaligned with actual motion.

**Current state:** `min_updates=4` in KF, but `MIN_SEND_BUFFER=10` in `maybe_send()` gates the actual UART send. So sends already require 10 points. But `is_ready()` returns true at 4, causing `predict_intercept()` to run with bad velocity from frame 4 onward.

**Proposed fix:** Set `min_updates=8`. Combined with `MIN_SEND_BUFFER=10`, velocity will be more stable when predictions start.

**File:** `trajectory/robot_predictor.py` line 69

---

## FIX 3: Tighten Reproj Error Filter from 20px to 5px (MODERATE)

**Problem:** The reproj filter in the integration script is 20px. The stereo triangulator already rejects at 8px internally. Points with 8-20px reproj pass through — these are noisy measurements that degrade the KF velocity estimate.

From static measurements: points with reproj < 1.5px had 43mm RMS error. Higher reproj = much worse accuracy.

**Proposed fix:** Tighten threshold to 5px. Matches the quality level of our verified transform points.

**Risk:** May reject more frames during fast motion, reducing buffer fill rate.

**File:** `test_integration_simple.py` line 714

---

## FIX 4: Test with DRAG_K = 0 (MODERATE)

**Problem:** `DRAG_K = 0.000112` is a theoretical value for a ping pong ball. At 3000 mm/s throw speed, drag deceleration is ~1000 mm/s² (10% of gravity). Over a 400ms prediction window, this shifts the intercept by ~160mm.

If the actual drag differs (ball condition, spin, etc.), this is systematic error. The best prediction (84mm) was at lower speed where drag effect was smaller.

**Proposed fix:** Set `DRAG_K = 0` temporarily. Compare 10 throws vs current value. If accuracy improves, drag model is hurting. If it worsens, keep it.

**File:** `trajectory/workspace.py` line 31

---

## FIX 5: Increase MAX_PREDICT_Y from 1000 to 1400 (MODERATE)

**Problem:** `MAX_PREDICT_Y = 1000mm` means `predict_intercept()` returns None until ball is within 1000mm of robot. Throws start at Y=2000+, so the ball must travel more than half its flight before any prediction happens.

The mid-flight region (Y=1000-1400) often has the most stable velocity because the ball has been tracked for 10+ frames and isn't yet in the noisy close-range zone.

**Proposed fix:** Increase to 1400mm. At VY=-2500mm/s, ball at Y=1400 reaches workspace in ~0.5s — enough time for robot to react.

**File:** `trajectory/robot_predictor.py` line 47

---

## FIX 6: Reduce SCAN_DURATION from 3s to 1.0s (LOW)

**Problem:** Forward sim scans 3 seconds at 5ms steps = 600 iterations per frame. A real throw takes <0.6s from Y=1400 to workspace. Scanning 3s wastes CPU and can find spurious intercepts from second/third bounces far in the future.

**Proposed fix:** Reduce to 1.0s. Still covers any realistic flight with margin.

**File:** `trajectory/robot_predictor.py` line 42

---

## FIX 7: Reduce POST_BOUNCE_MIN_UPDATES from 8 to 6 (LOW)

**Problem:** After a bounce, two independent gates delay the send:
1. KF needs 8 updates post-bounce (~200ms)
2. VZ must be negative (ball descending)

These overlap. The VZ gate is the more meaningful physical check. Having both at high values may over-delay, leaving less time for the robot to move.

**Proposed fix:** Reduce to 6. The VZ>0 gate already ensures we don't send during the noisy rising phase.

**File:** `trajectory/robot_predictor.py` line 58

---

## NEW FIX 8: Approach Filter Too Aggressive (HIGH — blocks valid throws)

**Problem:** `_ball_approaching()` requires EITHER `abs(y) < 600mm` OR `vy < -200 mm/s`. A ball at Y=800mm approaching slowly at VY=-100 mm/s is blocked. This may explain why only 6 of 19 throws (32%) produced predictions.

**Impact:** Many valid throws never get a prediction because the ball is approaching but not fast enough to pass the -200 mm/s threshold, and not close enough for the 600mm threshold.

**Proposed fix:** Change `MIN_APPROACH_VY = -200` to `MIN_APPROACH_VY = 0` (any negative VY = approaching). Or simply: `if vy < 0 or abs(y) < APPROACH_Y_THRESHOLD: return True`

**File:** `trajectory/robot_predictor.py` line 50

---

## NEW FIX 9: Most Predictions Are Clamped — Ball Doesn't Reach Workspace (HIGH — fundamental)

**Problem:** Analysis shows only ~40% of throws actually enter the workspace ellipse. The rest get "clamped" — the nearest point on the workspace boundary is sent instead. A clamped prediction can be 200-350mm away from where the ball actually goes.

**Root cause:** Either throws are genuinely missing the workspace (thrown too high/wide), OR the workspace definition is too small, OR the forward sim physics are wrong (drag too high, gravity wrong).

**Diagnosis needed:** Check `intercept_log.json` — how many sends have `clamped: true`? If >50%, the workspace may need expanding or the throws need to be aimed better.

**Proposed fix:** If mostly clamped, consider expanding workspace Z range: `Z_MAX = -730` → `Z_MAX = -600` (allows predictions higher up where ball actually passes). OR reduce DRAG_K so the sim predicts the ball reaching further.

**File:** `trajectory/workspace.py` lines 16-17

---

## Implementation Order (Revised)

| Priority | Fix | Risk | Test Method |
|----------|-----|------|-------------|
| 1st | Fix 8 (approach filter VY threshold) | None — just less restrictive | More throws should produce predictions |
| 2nd | Fix 3 (reproj 20→5px) | Low — filters more aggressively | Check frame acceptance stays >50% |
| 3rd | Fix 5 (MAX_PREDICT_Y 1000→1400) | None — earlier predictions | More sends logged |
| 4th | Fix 2 (min_updates 4→8) | Low — delays KF readiness | Smoother velocity at send |
| 5th | Fix 4 (DRAG_K=0 test) | Reversible | Compare 10 throws |
| 6th | Fix 9 (workspace Z range or diagnosis) | Medium — changes where robot moves | Check clamped ratio first |
| 7th | Fix 6 (scan 3→1s) | None | Performance only |
| 8th | Fix 7 (post-bounce 8→6) | Low | Slightly earlier post-bounce sends |
