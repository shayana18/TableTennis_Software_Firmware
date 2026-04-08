"""
Position Buffer - Timestamped 3D Position History

Stores recent ball positions with timestamps for velocity calculation.
Uses a circular buffer to maintain fixed memory usage.

Part of trajectory prediction pipeline:
  position_buffer.py  →  velocity_estimator.py  →  trajectory_predictor.py
"""

import time
import numpy as np
from collections import deque


class PositionBuffer:
    """
    Circular buffer storing timestamped 3D positions.
    
    Each entry: (X, Y, Z, timestamp)
    
    Usage:
        buffer = PositionBuffer(max_size=10)
        buffer.add(x, y, z)  # Timestamp auto-added
        positions = buffer.get_recent(n=5)
    """
    
    def __init__(self, max_size=10):
        """
        Initialize position buffer.
        
        Args:
            max_size: Maximum positions to store (older ones discarded)
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, x, y, z, timestamp=None):
        """
        Add a new position to the buffer.
        
        Args:
            x, y, z: 3D coordinates (in calibration units, e.g., cm)
            timestamp: Time in seconds (auto-generated if None)
        """
        if timestamp is None:
            timestamp = time.perf_counter()
        
        self.buffer.append({
            'x': float(x),
            'y': float(y),
            'z': float(z),
            't': float(timestamp)
        })
    
    def add_position(self, position_3d, timestamp=None):
        """
        Add position from tuple/list.
        
        Args:
            position_3d: (X, Y, Z) tuple or list
            timestamp: Time in seconds (auto-generated if None)
        """
        self.add(position_3d[0], position_3d[1], position_3d[2], timestamp)
    
    def get_recent(self, n=None):
        """
        Get most recent positions.
        
        Args:
            n: Number of positions (None = all)
        
        Returns:
            List of dicts: [{'x', 'y', 'z', 't'}, ...]
            Ordered oldest → newest
        """
        if n is None or n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def get_as_arrays(self, n=None):
        """
        Get positions as numpy arrays.
        
        Args:
            n: Number of positions (None = all)
        
        Returns:
            (positions, timestamps) where:
                positions: np.array shape (N, 3) → [[x,y,z], ...]
                timestamps: np.array shape (N,) → [t1, t2, ...]
        """
        recent = self.get_recent(n)
        
        if not recent:
            return np.array([]).reshape(0, 3), np.array([])
        
        positions = np.array([[p['x'], p['y'], p['z']] for p in recent])
        timestamps = np.array([p['t'] for p in recent])
        
        return positions, timestamps
    
    def get_latest(self):
        """
        Get most recent position.
        
        Returns:
            dict {'x', 'y', 'z', 't'} or None if empty
        """
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1]
    
    def get_oldest(self):
        """
        Get oldest position in buffer.
        
        Returns:
            dict {'x', 'y', 'z', 't'} or None if empty
        """
        if len(self.buffer) == 0:
            return None
        return self.buffer[0]
    
    def clear(self):
        """Clear all positions from buffer."""
        self.buffer.clear()
    
    def __len__(self):
        """Return number of positions in buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_points=3):
        """
        Check if buffer has enough points for velocity calculation.
        
        Args:
            min_points: Minimum required points
        
        Returns:
            True if buffer has >= min_points
        """
        return len(self.buffer) >= min_points
    
    def get_time_span(self):
        """
        Get time span covered by buffer.
        
        Returns:
            Time in seconds from oldest to newest, or 0 if < 2 points
        """
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1]['t'] - self.buffer[0]['t']
    
    def get_average_dt(self):
        """
        Get average time between samples.
        
        Returns:
            Average dt in seconds, or 0 if < 2 points
        """
        if len(self.buffer) < 2:
            return 0.0
        return self.get_time_span() / (len(self.buffer) - 1)