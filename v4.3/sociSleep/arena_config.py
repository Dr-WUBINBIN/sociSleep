"""Load and save per-camera arena geometry to JSON files."""

import json
import os

import cv2

from sociSleep.config import ARENA_CONFIG_FILES, DEFAULT_ARENA, ARENA_WINDOW_PREFIX


def _config_path(camera_id):
    """Return the JSON path for a given camera id (1 or 2)."""
    return os.path.join(os.getcwd(), ARENA_CONFIG_FILES[camera_id])


def load_arena_config(camera_id):
    """Load arena settings from cameraN.json, falling back to defaults."""
    path = _config_path(camera_id)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**DEFAULT_ARENA, **data}
    return dict(DEFAULT_ARENA)


def save_arena_config(camera_id, arena):
    """Persist arena settings to cameraN.json."""
    path = _config_path(camera_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arena, f, indent=2)


class ArenaSettings:
    """
    OpenCV trackbar window for one camera's circular arena geometry.

    Reads slider positions each frame and supports automatic JSON persistence.
    """

    TRACKBARS = [
        ("Left_CX", "left_cx", 640),
        ("Left_CY", "left_cy", 480),
        ("Left_R", "left_r", 200),
        ("Right_CX", "right_cx", 640),
        ("Right_CY", "right_cy", 480),
        ("Right_R", "right_r", 200),
    ]

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.window_name = f"{ARENA_WINDOW_PREFIX} {camera_id}"
        self._values = load_arena_config(camera_id)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        for tb_name, key, max_val in self.TRACKBARS:
            cv2.createTrackbar(
                tb_name,
                self.window_name,
                int(self._values[key]),
                max_val,
                lambda _x: None,
            )

    def read(self):
        """Read current trackbar positions into a dict."""
        for tb_name, key, _ in self.TRACKBARS:
            self._values[key] = cv2.getTrackbarPos(tb_name, self.window_name)
        return dict(self._values)

    def save(self):
        """Write current arena settings to disk."""
        save_arena_config(self.camera_id, self._values)

    def destroy(self):
        """Save settings and close the trackbar window."""
        self.save()
        cv2.destroyWindow(self.window_name)
