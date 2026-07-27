"""Kalman filter helpers for fly position prediction and correction."""

import cv2
import numpy as np


def create_kalman(dt=1.0):
    """Create a 4-state (x, y, vx, vy) Kalman filter with 2D position measurements."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def update_kalman(kf, x, y):
    """Apply a position measurement to the Kalman filter."""
    try:
        kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
    except Exception:
        pass


def clamp_predict_in_circle(px, py, cx, cy, r):
    """Clamp a predicted point to stay inside a circular arena."""
    dx = px - cx
    dy = py - cy
    dist = (dx * dx + dy * dy) ** 0.5
    if dist > r:
        px = cx + dx / dist * r
        py = cy + dy / dist * r
    return int(px), int(py)
