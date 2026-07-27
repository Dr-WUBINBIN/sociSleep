"""
Unified OpenCV dashboard for one or two camera feeds.

Composites live tracker frames side-by-side and routes mouse clicks
to the correct camera panel.
"""

import cv2
import numpy as np

from sociSleep.config import DASHBOARD_WINDOW


class Dashboard:
    """
    Single dashboard window displaying one or two annotated camera feeds.

    Each feed occupies a horizontal panel; mouse events are forwarded to
    the corresponding FlyTracker instance using panel-local coordinates.
    """

    def __init__(self, trackers):
        """
        Args:
            trackers: List of FlyTracker instances (1 or 2).
        """
        self.trackers = trackers
        self.window_name = DASHBOARD_WINDOW
        self._panel_width = 640
        self._panel_height = 480
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Route mouse clicks to the tracker whose panel contains (x, y)."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for i, tracker in enumerate(self.trackers):
            panel_x1 = i * self._panel_width
            panel_x2 = panel_x1 + self._panel_width
            if panel_x1 <= x < panel_x2:
                local_x = x - panel_x1
                tracker.handle_click(local_x, y)
                return

    def compose_frame(self):
        """
        Build the dashboard image from the latest annotated frames.

        Returns a numpy array suitable for cv2.imshow, or None if no frames yet.
        """
        panels = []
        for tracker in self.trackers:
            frame = tracker.get_latest_frame()
            if frame is None:
                placeholder = np.zeros(
                    (self._panel_height, self._panel_width, 3), dtype=np.uint8
                )
                cv2.putText(
                    placeholder,
                    f"Waiting for Camera {tracker.camera_id}...",
                    (80, self._panel_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )
                panels.append(placeholder)
            else:
                resized = cv2.resize(frame, (self._panel_width, self._panel_height))
                panels.append(resized)

        if len(panels) == 1:
            return panels[0]
        return np.hstack(panels)

    def show(self):
        """Display the current composite frame."""
        composite = self.compose_frame()
        if composite is not None:
            cv2.imshow(self.window_name, composite)

    def destroy(self):
        """Close the dashboard window."""
        cv2.destroyWindow(self.window_name)

    @property
    def should_stop(self):
        """True when any tracker has requested stop."""
        return any(t.should_stop for t in self.trackers)

    @property
    def width(self):
        return self._panel_width * len(self.trackers)

    @property
    def height(self):
        return self._panel_height
