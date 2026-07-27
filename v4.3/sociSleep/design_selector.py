"""Runtime plate-design selection UI and persistence."""

import json
import os
import threading

import cv2
import numpy as np

from sociSleep.plate_design import ALL_DESIGNS, DEFAULT_DESIGN

DESIGN_WINDOW = "Plate Design"
DESIGN_CONFIG_FILE = "plate_design.json"


class DesignSelector:
    """
    Shared plate-design selector used by all cameras in one session.

    Design can be changed any time before START is pressed on any camera.
    Once tracking has started, the design is locked until the app restarts.
    """

    def __init__(self):
        self._index = self._load_saved_index()
        self._lock = threading.Lock()
        self._locked = False
        self._buttons = []
        cv2.namedWindow(DESIGN_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(DESIGN_WINDOW, self._mouse_callback)
        self.refresh()

    def _load_saved_index(self):
        path = os.path.join(os.getcwd(), DESIGN_CONFIG_FILE)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                design_id = data.get("design_id", DEFAULT_DESIGN.id)
                for i, design in enumerate(ALL_DESIGNS):
                    if design.id == design_id:
                        return i
            except (json.JSONDecodeError, OSError):
                pass
        return next(i for i, d in enumerate(ALL_DESIGNS) if d.id == DEFAULT_DESIGN.id)

    def read(self):
        """Return the currently selected PlateDesign."""
        with self._lock:
            return ALL_DESIGNS[self._index]

    def lock(self):
        """Prevent further design changes once tracking has started."""
        with self._lock:
            self._locked = True
        self.refresh()

    @property
    def is_locked(self):
        with self._lock:
            return self._locked

    def set_by_index(self, index):
        """Set design by index (0-based). No-op if locked."""
        with self._lock:
            if self._locked:
                return
            index = max(0, min(index, len(ALL_DESIGNS) - 1))
            self._index = index
        self.refresh()

    def save(self):
        """Persist the selected design to plate_design.json."""
        design = self.read()
        path = os.path.join(os.getcwd(), DESIGN_CONFIG_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"design_id": design.id, "design_name": design.name}, f, indent=2)

    def refresh(self):
        """Update the design selector window."""
        with self._lock:
            locked = self._locked
            index = self._index
            design = ALL_DESIGNS[index]
            
        width, height = 340, 320
        panel = 255 * np.ones((height, width, 3), dtype=np.uint8)

        button_height = 50
        button_width = width - 80
        margin_top = 20
        margin_left = 40
        spacing = 20

        self._buttons.clear()

        for i, d in enumerate(ALL_DESIGNS):
            y1 = margin_top + i * (button_height + spacing)
            y2 = y1 + button_height
            x1 = margin_left
            x2 = x1 + button_width
            rect = (x1, y1, x2, y2)
            self._buttons.append(rect)

            if locked:
                # Draw all buttons in gray, selected in darker gray
                if i == index:
                    color = (120, 120, 120)
                else:
                    color = (200, 200, 200)
            else:
                # Selected button green filled, others light gray
                if i == index:
                    color = (0, 200, 0)
                else:
                    color = (220, 220, 220)

            cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness=-1)
            # Draw border
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

            # Draw label centered
            label = d.name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x1 + (button_width - text_w) // 2
            text_y = y1 + (button_height + text_h) // 2
            # Text color: black if selected green, else black
            if locked:
                text_color = (255, 255, 255) if i == index else (0, 0, 0)
            else:
                text_color = (0, 0, 0) if i != index else (255, 255, 255)
            cv2.putText(panel, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Below buttons: arena info and status
        info_y = margin_top + len(ALL_DESIGNS) * (button_height + spacing) + 10
        line_spacing = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        black = (0, 0, 0)

        left = ", ".join(design.left_labels)
        right = ", ".join(design.right_labels)
        
        left_text = "fly" if design.left_count == 1 else "flies"
        right_text = "fly" if design.right_count == 1 else "flies"
        
        cv2.putText(panel, f"Left arena:  {design.left_count} {left_text} ({left})",
                    (margin_left, info_y), font, font_scale, black, thickness, cv2.LINE_AA)
        cv2.putText(panel, f"Right arena: {design.right_count} {right_text} ({right})",
                    (margin_left, info_y + line_spacing), font, font_scale, black, thickness, cv2.LINE_AA)

        status_text = "LOCKED" if locked else "READY"
        status_color = (0, 0, 255) if locked else (0, 180, 0)
        cv2.putText(panel, f"Status: {status_text}",
                    (margin_left, info_y + 2 * line_spacing), font, font_scale, status_color, thickness, cv2.LINE_AA)

        cv2.imshow(DESIGN_WINDOW, panel)

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with self._lock:
            if self._locked:
                return
            for i, (x1, y1, x2, y2) in enumerate(self._buttons):
                if x1 <= x < x2 and y1 <= y < y2:
                    self._index = i
                    break
            else:
                return
        self.refresh()


    def destroy(self):
        """Close design selector windows."""
        cv2.destroyWindow(DESIGN_WINDOW)
