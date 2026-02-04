"""
Position Buffer

Stores timestamped 3D positions for velocity estimation.
Uses a circular buffer to maintain recent history.

Note: This module is camera-agnostic.
"""

import time
import numpy as np
from collections import deque


class PositionBuffer:
    """
    Circular buffer for storing timestamped 3D positions.

    Used by TrajectoryPredictor to maintain position history
    for velocity calculation.
    """

    def __init__(self, max_size=10, max_gap_seconds=0.5):
        """
        Initialize position buffer.

        Args:
            max_size: Maximum number of positions to store
            max_gap_seconds: Maximum time gap before clearing buffer (tracking loss detection)
        """
        self.max_size = max_size
        self.max_gap_seconds = max_gap_seconds
        self.buffer = deque(maxlen=max_size)

    def add(self, x, y, z, t=None):
        """
        Add a position to the buffer.

        Args:
            x, y, z: 3D coordinates
            t: Timestamp (auto-generated if None)

        Returns:
            bool: True if position was added, False if rejected
        """
        if t is None:
            t = time.perf_counter()

        # Validate timestamp is increasing
        if len(self.buffer) > 0:
            last_t = self.buffer[-1]['t']
            if t <= last_t:
                # Timestamp not increasing - reject
                return False

            # Check for large time gaps (likely tracking loss)
            if t - last_t > self.max_gap_seconds:
                # Clear buffer and start fresh after tracking loss
                self.buffer.clear()

        self.buffer.append({
            'x': x,
            'y': y,
            'z': z,
            't': t
        })
        return True

    def get_recent(self, n=None):
        """
        Get the n most recent positions.

        Args:
            n: Number of positions (None = all)

        Returns:
            List of position dicts
        """
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]

    def get_as_arrays(self):
        """
        Get positions as numpy arrays.

        Returns:
            dict with 'x', 'y', 'z', 't' arrays
        """
        if len(self.buffer) == 0:
            return {
                'x': np.array([]),
                'y': np.array([]),
                'z': np.array([]),
                't': np.array([])
            }
        
        return {
            'x': np.array([p['x'] for p in self.buffer]),
            'y': np.array([p['y'] for p in self.buffer]),
            'z': np.array([p['z'] for p in self.buffer]),
            't': np.array([p['t'] for p in self.buffer])
        }

    def is_ready(self, min_points=3):
        """
        Check if buffer has enough points for velocity calculation.

        Args:
            min_points: Minimum required points

        Returns:
            bool
        """
        return len(self.buffer) >= min_points

    def clear(self):
        """Clear all positions."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    @property
    def latest(self):
        """Get most recent position."""
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1]

    @property
    def oldest(self):
        """Get oldest position in buffer."""
        if len(self.buffer) == 0:
            return None
        return self.buffer[0]