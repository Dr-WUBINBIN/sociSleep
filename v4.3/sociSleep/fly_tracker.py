"""
Per-camera fly tracker.

Supports configurable plate designs (solo vs solo, solo vs group, group vs group)
while preserving the core Kalman / assignment / merge-blob algorithm from v4.0.
"""

import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from sociSleep.assignment import avg_position, nearest_assignment
from sociSleep.config import INIT_FRAMES, MAX_MERGE_FACTOR, MOVEMENT_THRESHOLD
from sociSleep.csv_logger import CSVLogger
from sociSleep.detection import filter_by_circle
from sociSleep.kalman_utils import clamp_predict_in_circle, create_kalman, update_kalman
from sociSleep.camera_detector import open_camera
from sociSleep.plate_design import LABEL_COLORS, PlateDesign


class FlyTracker:
    """
    Track flies on one camera feed according to a selectable plate design.

    Each instance owns a VideoCapture, tracking state, and CSV output stream.
    """

    def __init__(self, camera_id, camera_index, arena_settings, design_selector):
        """
        Args:
            camera_id: Logical id (1 or 2) used for output filenames.
            camera_index: OpenCV DirectShow device index.
            arena_settings: ArenaSettings instance for this camera.
            design_selector: Shared DesignSelector for the session.
        """
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.arena_settings = arena_settings
        self.design_selector = design_selector

        self.cap, width, height, fps = open_camera(camera_index)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        self.fps = fps
        self.frames_per_sec = int(round(fps))

        print(
            f"✅ Camera {camera_id} opened via DirectShow (index {camera_index}). "
            f"Resolution: {width}x{height}, FPS: {fps:.2f} → {self.frames_per_sec}"
        )

        design = design_selector.read()
        self.csv_logger = CSVLogger(camera_id, design)
        self.window = self.frames_per_sec
        self.max_hist = self.frames_per_sec * 2

        self.frame_counter = 0
        self.merged_threshold_left = None
        self.merged_threshold_right = None
        self.tracking_started = False
        self._active_design = design

        self._init_tracking_state(design)

        self.stop_button = {"x1": 10, "y1": 10, "x2": 110, "y2": 50, "pressed": False}
        self.settings_button = {"x1": 10, "y1": 60, "x2": 110, "y2": 100}
        self.start_button = {"x1": 130, "y1": 10, "x2": 230, "y2": 50, "pressed": False}

        self._stop_requested = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._thread = None

    def _init_tracking_state(self, design):
        """Create empty histories and Kalman state for the given design."""
        self._active_design = design
        self.fly_histories = {label: deque(maxlen=self.max_hist) for label in design.all_labels}
        self.kalman_filters = {label: None for label in design.all_labels}
        self.last_measurement_time = {label: None for label in design.all_labels}
        self.last_log_time = datetime.now()
        self.merged_threshold_left = None
        self.merged_threshold_right = None
        self.tracking_started = False
        self.csv_logger.set_design(design)

    # ------------------------------------------------------------------ #
    # Thread lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        """Launch the tracking loop in a background thread."""
        self._thread = threading.Thread(
            target=self._run_loop, name=f"FlyTracker-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def request_stop(self):
        """Signal the tracker thread to stop."""
        self._stop_requested = True
        self.stop_button["pressed"] = True

    def join(self, timeout=None):
        """Wait for the tracker thread to finish."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def release(self):
        """Release the camera capture device."""
        self.cap.release()

    def get_latest_frame(self):
        """Return the most recent annotated frame (thread-safe copy)."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def should_stop(self):
        return self._stop_requested or self.stop_button["pressed"]

    # ------------------------------------------------------------------ #
    # UI interactions
    # ------------------------------------------------------------------ #

    def handle_click(self, x, y):
        """Handle a mouse click within this camera's panel region."""
        if (
            self.stop_button["x1"] <= x <= self.stop_button["x2"]
            and self.stop_button["y1"] <= y <= self.stop_button["y2"]
        ):
            self.stop_button["pressed"] = True
            self._stop_requested = True
            return True
        if (
            self.settings_button["x1"] <= x <= self.settings_button["x2"]
            and self.settings_button["y1"] <= y <= self.settings_button["y2"]
        ):
            self.open_camera_settings()
            return True
        if (
            self.start_button["x1"] <= x <= self.start_button["x2"]
            and self.start_button["y1"] <= y <= self.start_button["y2"]
        ):
            self.start_button["pressed"] = True
            print(f"START button pressed (Camera {self.camera_id})")
            return True
        return False

    def open_camera_settings(self):
        """Open the native DirectShow property dialog for this camera."""
        try:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Could not open property dialog via OpenCV (Camera {self.camera_id}):", e)

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def initialize_tracking(
        self, design, left_cx, left_cy, left_r, right_cx, right_cy, right_r, init_frames=INIT_FRAMES
    ):
        """
        Gather detections over several frames and average them for robust init.

        Returns (init_positions dict, init_areas dict) keyed by fly label.
        """
        print(
            f"Camera {self.camera_id} [{design.name}]: Initializing, averaging "
            f"detections for {init_frames} frames..."
        )
        label_pts = {label: [] for label in design.all_labels}
        label_areas = {label: [] for label in design.all_labels}
        frames_captured = 0

        for _ in range(init_frames):
            ret, f = self.cap.read()
            print("Init frame", _)
            if not ret:
                break
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            _, mask_local = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask_local = cv2.medianBlur(mask_local, 5)

            left_centroids = filter_by_circle(mask_local, left_cx, left_cy, left_r)
            right_centroids = filter_by_circle(mask_local, right_cx, right_cy, right_r)

            for i, label in enumerate(design.left_labels):
                if i < len(left_centroids):
                    cx, cy, area = left_centroids[i]
                    label_pts[label].append((cx, cy))
                    label_areas[label].append(area)

            for i, label in enumerate(design.right_labels):
                if i < len(right_centroids):
                    cx, cy, area = right_centroids[i]
                    label_pts[label].append((cx, cy))
                    label_areas[label].append(area)

            frames_captured += 1
            time.sleep(0.02)

        if frames_captured == 0:
            print(f"⚠ Camera {self.camera_id}: Initialization frames not captured.")
            return {}, {}

        def mean_pt(pts):
            if not pts:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (int(np.mean(xs)), int(np.mean(ys)))

        init_positions = {label: mean_pt(label_pts[label]) for label in design.all_labels}
        init_areas = {
            label: int(np.mean(label_areas[label])) if label_areas[label] else None
            for label in design.all_labels
        }

        print(
            f"Camera {self.camera_id} initialization ({design.name}): "
            + " ".join(f"{l}:{init_positions[l]} area:{init_areas[l]}" for l in design.all_labels)
        )
        return init_positions, init_areas

    def _set_merge_thresholds(self, design, init_areas):
        """Compute merged-blob area thresholds for multi-fly arenas."""
        fallback = next((a for a in init_areas.values() if a is not None), None)

        def pair_threshold(labels):
            if len(labels) < 2:
                return None
            a0 = init_areas.get(labels[0]) or fallback
            a1 = init_areas.get(labels[1]) or fallback
            if a0 is None or a1 is None:
                return None
            return (a0 + a1) * MAX_MERGE_FACTOR

        self.merged_threshold_left = pair_threshold(design.left_labels)
        self.merged_threshold_right = pair_threshold(design.right_labels)

        if self.merged_threshold_left is not None:
            print(f"Camera {self.camera_id}: left merged_threshold = {self.merged_threshold_left}")
        if self.merged_threshold_right is not None:
            print(f"Camera {self.camera_id}: right merged_threshold = {self.merged_threshold_right}")

    def init_kalman_filters(self, design, init_positions):
        """Create and seed Kalman filters for each fly with a known initial position."""
        for label in design.all_labels:
            init = init_positions.get(label)
            if init is not None:
                kf = create_kalman(1.0 / self.frames_per_sec)
                kf.statePost = np.array(
                    [
                        [np.float32(init[0])],
                        [np.float32(init[1])],
                        [np.float32(0.0)],
                        [np.float32(0.0)],
                    ],
                    dtype=np.float32,
                )
                self.kalman_filters[label] = kf
                self.fly_histories[label].append(init)
                self.last_measurement_time[label] = datetime.now()
            else:
                self.kalman_filters[label] = None

    # ------------------------------------------------------------------ #
    # Detection helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_arena_detections(centroids, max_flies, merge_threshold):
        """
        Build detection list for one arena, applying merged-blob suppression when
        two flies share an arena (same logic as v4.0 for the right arena).
        """
        if max_flies == 1:
            if centroids:
                cx, cy, area = centroids[0]
                return [(cx, cy, area)]
            return []

        if (
            len(centroids) == 1
            and merge_threshold is not None
            and centroids[0][2] >= merge_threshold
        ):
            return []

        detections = []
        for i in range(min(max_flies, len(centroids))):
            detections.append(centroids[i])
        return detections

    def _predict_all(self, design, left_cx, left_cy, left_r, right_cx, right_cy, right_r):
        """Run Kalman predict for every active label and clamp to its arena circle."""
        preds = {}
        for label in design.all_labels:
            kf = self.kalman_filters.get(label)
            if kf is None:
                preds[label] = None
                continue
            if self.last_measurement_time[label] is not None:
                if (datetime.now() - self.last_measurement_time[label]).total_seconds() > 4:
                    self.kalman_filters[label] = None
                    preds[label] = None
                    continue
            try:
                p = kf.predict()
                px = float(p[0][0]) if isinstance(p[0], (list, np.ndarray)) else float(p[0])
                py = float(p[1][0]) if isinstance(p[1], (list, np.ndarray)) else float(p[1])
                if design.label_side(label) == "left":
                    px, py = clamp_predict_in_circle(px, py, left_cx, left_cy, left_r)
                else:
                    px, py = clamp_predict_in_circle(px, py, right_cx, right_cy, right_r)
                preds[label] = (px, py)
            except Exception:
                preds[label] = None
        return preds

    def _assign_arena(self, labels, preds, detections, arena_cx, arena_cy, arena_r, max_dist):
        """
        Assign detections to fly labels within one arena.

        Preserves the per-label assignment / bootstrap logic from v4.0.
        """
        arena_preds = {l: preds[l] for l in labels if preds.get(l) is not None}
        mapping, _ = nearest_assignment(arena_preds, detections, max_dist=max_dist)

        for i, label in enumerate(labels):
            if label in mapping and mapping[label] is not None:
                update_kalman(
                    self.kalman_filters[label], mapping[label][0], mapping[label][1]
                )
                self.fly_histories[label].append(mapping[label])
                self.last_measurement_time[label] = datetime.now()
            else:
                if preds.get(label) is not None and (
                    (preds[label][0] - arena_cx) ** 2 + (preds[label][1] - arena_cy) ** 2
                    <= arena_r ** 2
                ):
                    self.fly_histories[label].append(preds[label])
                else:
                    self.fly_histories[label].append(None)
                if preds.get(label) is None:
                    if len(detections) >= len(labels) and self.kalman_filters[label] is None:
                        chosen = detections[i]
                        kf = create_kalman(1.0 / self.frames_per_sec)
                        kf.statePost = np.array(
                            [[np.float32(chosen[0])], [np.float32(chosen[1])], [0], [0]],
                            dtype=np.float32,
                        )
                        self.kalman_filters[label] = kf
                        self.fly_histories[label].append((chosen[0], chosen[1]))
                        self.last_measurement_time[label] = datetime.now()

    # ------------------------------------------------------------------ #
    # Per-frame tracking
    # ------------------------------------------------------------------ #

    def process_frame(self, frame):
        """Run one frame of detection, Kalman prediction, assignment, and logging."""
        self.frame_counter += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        design = self.design_selector.read()

        if not self.tracking_started and design.id != self._active_design.id:
            self._init_tracking_state(design)

        arena = self.arena_settings.read()
        left_cx = arena["left_cx"]
        left_cy = arena["left_cy"]
        left_r = arena["left_r"]
        right_cx = arena["right_cx"]
        right_cy = arena["right_cy"]
        right_r = arena["right_r"]

        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.medianBlur(mask, 5)

        cv2.circle(frame, (left_cx, left_cy), left_r, (0, 255, 0), 2)
        cv2.circle(frame, (right_cx, right_cy), right_r, (0, 255, 0), 2)

        if self.start_button["pressed"] and not self.tracking_started:
            design = self.design_selector.read()
            self._init_tracking_state(design)
            self.design_selector.lock()
            print(f"Camera {self.camera_id}: Starting tracking ({design.name})...")
            init_positions, init_areas = self.initialize_tracking(
                design, left_cx, left_cy, left_r, right_cx, right_cy, right_r
            )
            self._set_merge_thresholds(design, init_areas)
            self.init_kalman_filters(design, init_positions)
            self.tracking_started = True
            self.start_button["pressed"] = False

        left_centroids = filter_by_circle(mask, left_cx, left_cy, left_r)
        right_centroids = filter_by_circle(mask, right_cx, right_cy, right_r)

        detections_left = self._collect_arena_detections(
            left_centroids, design.left_count, self.merged_threshold_left
        )
        detections_right = self._collect_arena_detections(
            right_centroids, design.right_count, self.merged_threshold_right
        )

        preds = self._predict_all(
            design, left_cx, left_cy, left_r, right_cx, right_cy, right_r
        )

        self._assign_arena(
            design.left_labels, preds, detections_left, left_cx, left_cy, left_r, max_dist=200
        )
        self._assign_arena(
            design.right_labels, preds, detections_right, right_cx, right_cy, right_r, max_dist=200
        )

        for cx, cy, area in detections_left:
            if (cx - left_cx) ** 2 + (cy - left_cy) ** 2 <= left_r ** 2:
                cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)
        for cx, cy, area in detections_right:
            if (cx - right_cx) ** 2 + (cy - right_cy) ** 2 <= right_r ** 2:
                cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 1)

        display_positions = {}
        for label in design.all_labels:
            hist = self.fly_histories[label]
            if len(hist) > 0 and hist[-1] is not None:
                pos = hist[-1]
                display_positions[label] = pos
                color = LABEL_COLORS.get(label, (255, 255, 255))
                cv2.drawMarker(
                    frame, (int(pos[0]), int(pos[1])), color, cv2.MARKER_TILTED_CROSS, 20, 2
                )
                cv2.putText(
                    frame,
                    f"Fly{label} {int(pos[0])},{int(pos[1])}",
                    (int(pos[0]) + 10, int(pos[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            else:
                display_positions[label] = (None, None)

        movement_flags = {}
        for label in design.all_labels:
            hist = list(self.fly_histories[label])
            if len(hist) < 2:
                movement_flags[label] = 0
                continue
            recent = hist[-self.window :]
            past = hist[-2 * self.window : -self.window] if len(hist) >= 2 * self.window else []
            recent_valid = [p for p in recent if p is not None]
            past_valid = [p for p in past if p is not None]
            if len(recent_valid) < self.window * 0.3 or len(past_valid) < self.window * 0.3:
                movement_flags[label] = 0
                continue
            avg_recent = avg_position(recent_valid)
            avg_past = avg_position(past_valid)
            if avg_recent is None or avg_past is None:
                movement_flags[label] = 0
                continue
            dist = np.hypot(avg_recent[0] - avg_past[0], avg_recent[1] - avg_past[1])
            movement_flags[label] = 1 if dist > MOVEMENT_THRESHOLD else 0

        now = datetime.now()
        if (now - self.last_log_time).total_seconds() >= 1:
            row = [now.strftime("%Y-%m-%d %H:%M:%S")]
            for label in design.all_labels:
                row.extend([
                    display_positions[label][0],
                    display_positions[label][1],
                    movement_flags[label],
                ])
            self.csv_logger.write_row(row)
            self.last_log_time = now

        mov_text = " ".join(f"{l}_mov:{movement_flags.get(l, 0)}" for l in design.all_labels)
        cv2.putText(
            frame, mov_text, (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
        )

        self._draw_buttons(frame)
        cv2.putText(
            frame,
            f"Cam{self.camera_id} | {design.name}",
            (frame.shape[1] - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return frame

    def _draw_buttons(self, frame):
        """Draw Stop, Camera Settings, and Start buttons on the frame."""
        cv2.rectangle(
            frame,
            (self.stop_button["x1"], self.stop_button["y1"]),
            (self.stop_button["x2"], self.stop_button["y2"]),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            frame, "Stop",
            (self.stop_button["x1"] + 10, self.stop_button["y1"] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.rectangle(
            frame,
            (self.settings_button["x1"], self.settings_button["y1"]),
            (self.settings_button["x2"], self.settings_button["y2"]),
            (255, 100, 0),
            -1,
        )
        cv2.putText(
            frame, "Camera",
            (self.settings_button["x1"] + 10, self.settings_button["y1"] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.rectangle(
            frame,
            (self.start_button["x1"], self.start_button["y1"]),
            (self.start_button["x2"], self.start_button["y2"]),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame, "Start",
            (self.start_button["x1"] + 5, self.start_button["y1"] + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    def _run_loop(self):
        """Background thread: capture frames and run tracking."""
        while not self._stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                print(f"⚠️ Camera {self.camera_id}: Frame not captured.")
                time.sleep(0.5)
                continue
            annotated = self.process_frame(frame)
            with self._lock:
                self._latest_frame = annotated
