"""Blob detection and filtering within circular arenas."""

import cv2

from sociSleep.config import NOISE_THRESHOLD


def filter_by_circle(mask, cx, cy, r):
    """
    Find contours inside a circular arena and return centroids sorted by area (largest first).

    Returns list of (x, y, area) tuples.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < NOISE_THRESHOLD:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
            results.append((x, y, area))
    results.sort(key=lambda item: item[2], reverse=True)
    return results
