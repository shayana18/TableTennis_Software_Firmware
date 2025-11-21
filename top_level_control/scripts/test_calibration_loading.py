"""
Test script to verify calibration values load correctly.
Does NOT require cameras to be connected.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tracking.stereo_tracker import StereoTracker

def main():
    config_path = Path(__file__).parent.parent / "config" / "stereo_config.yaml"

    print("Testing calibration loading...\n")

    # This will print all loaded calibration values
    tracker = StereoTracker(str(config_path))

    print("\nCalibration loading test complete.")
    print(f"Stereo calibrated: {tracker.is_calibrated}")

if __name__ == "__main__":
    main()
