This Directory contains all the all the code hosted on the laptop, it is responsible for the ball identification, 3d position reconstruction, trajectory planning and generated the desired joint positions to meet the all the desired position with the desired speed!



HSV BALL TRACKING AND CONTOUR DETECTION (IMPROVMENETS )

  1. Tightened HSV range for orange:
    - Old: (0, 94, 108) to (25, 255, 197)
    - New: (5, 120, 120) to (20, 255, 255)
    - Narrower hue (5-20) avoids red/yellow, higher saturation/value thresholds
  2. 5 Strict Circularity Checks - must pass ALL:
    - Min area: 200 pixels (rejects tiny noise)
    - Circularity: 70%+ (contour area vs enclosing circle)
    - Solidity: 80%+ (contour area vs convex hull - rejects irregular shapes)
    - Aspect ratio: < 1.4 (bounding box must be square-ish)
    - Radius: 8-80 pixels
  3. More aggressive noise reduction:
    - 3 erosion/dilation iterations (was 2)
    - Added Gaussian blur + threshold for smoother edges
  4. Better scoring: Prefers larger, more circular, more solid contours



PHYSICS BASED MODEL; BALL PREDICITON

  1. Velocity Calculation:
    - Takes ball center position from current frame (x_t, y_t) and previous frame (x_t-1, y_t-1)
    - Computes raw velocity: v_raw = (x_t - x_t-1, y_t - y_t-1) in pixels/frame
    - Applies Exponential Moving Average (EMA) to smooth out noise:
      v_smoothed = 0.7 * v_previous + 0.3 * v_raw
    - Result: velocity vector (vx, vy) in pixels/frame

  2. Speed Calculation:
    - Computes magnitude using Euclidean norm: speed = sqrt(vx^2 + vy^2)
    - Units: pixels per frame

  3. Position Prediction:
    - Uses constant velocity model to predict next frame position:
      x_next = x_current + vx
      y_next = y_current + vy
    - Assumes ball maintains same velocity between consecutive frames

  4. Position History Trail:
    - Stores last 32 ball positions in circular buffer
    - Visualizes trajectory on screen with fading effect
    - Older positions: thinner, darker lines
    - Newer positions: thicker, brighter lines