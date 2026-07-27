"""Global configuration constants for sociSleep tracking."""

import os

# Tracking parameters (unchanged from v4.0)
INIT_FRAMES = 5
MAX_MERGE_FACTOR = 0.85
MOVEMENT_THRESHOLD = 3
NOISE_THRESHOLD = 20

# Output paths
RESULTS_DIR = os.path.join(os.getcwd(), "results")

# Arena settings JSON filenames (saved in working directory)
ARENA_CONFIG_FILES = {
    1: "camera1.json",
    2: "camera2.json",
}

# Default arena geometry (used when no JSON exists)
DEFAULT_ARENA = {
    "left_cx": 200,
    "left_cy": 240,
    "left_r": 100,
    "right_cx": 440,
    "right_cy": 240,
    "right_r": 100,
}

# UI
DASHBOARD_WINDOW = "sociSleep Dashboard"
ARENA_WINDOW_PREFIX = "Arena Setting-Cam"


