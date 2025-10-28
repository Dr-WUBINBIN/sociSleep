# About social_sleep_tracker.
# social_sleep_tracker_v1.0
# developed by Binbin Wu Ph.D.
# Ja Lab, UF Scripps Institute, University of Florida
# © 2025. All rights reserved.

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from collections import deque

# === Output file ===
output_file = os.path.join(os.getcwd(), "fly_movement_dark_marker.csv")

# === Open Logitech camera ===
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
if not cap.isOpened():
    print("❌ Cannot open Logitech camera.")
    exit()
print("✅ Logitech camera opened successfully (index 1).")

# --- Disable auto exposure/white balance ---
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

# --- Measure FPS (fallback to 30 if unavailable) ---
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0
frames_per_sec = int(round(fps))
print(f"🎞️ Measured FPS: {fps:.2f}  →  {frames_per_sec} frames per second interval for movement check.")

# === Create main window and sliders ===
cv2.namedWindow("Fly Tracker")
def nothing(x): pass

# --- Camera sliders ---
cv2.createTrackbar('Brightness', 'Fly Tracker', 50, 100, nothing)
cv2.createTrackbar('Contrast', 'Fly Tracker', 50, 100, nothing)
cv2.createTrackbar('Exposure', 'Fly Tracker', 5, 13, nothing)

# --- Dark marker detection threshold ---
cv2.createTrackbar('Dark_min', 'Fly Tracker', 0, 100, nothing)  # Larger = darker
cv2.createTrackbar('Min_area', 'Fly Tracker', 5, 100, nothing)  # in pixels²

# --- Arena geometry sliders ---
cv2.createTrackbar('Rect Width', 'Fly Tracker', 300, 640, nothing)
cv2.createTrackbar('Rect Height', 'Fly Tracker', 200, 480, nothing)
cv2.createTrackbar('Center X', 'Fly Tracker', 320, 640, nothing)
cv2.createTrackbar('Center Y', 'Fly Tracker', 240, 480, nothing)

# --- Stop button ---
stop_button = {"x1": 10, "y1": 10, "x2": 110, "y2": 50, "pressed": False}
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if stop_button["x1"] <= x <= stop_button["x2"] and stop_button["y1"] <= y <= stop_button["y2"]:
            stop_button["pressed"] = True
cv2.setMouseCallback("Fly Tracker", mouse_callback)

# --- Variables ---
frame_counter = 0
centroid_history = [deque(maxlen=frames_per_sec + 1), deque(maxlen=frames_per_sec + 1)]
results = []
last_log_time = datetime.now()

print("🎥 Press 'q' or click STOP to end tracking.")

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break
    frame_counter += 1

    # --- Trackbar values ---
    brightness = cv2.getTrackbarPos('Brightness', 'Fly Tracker')
    contrast = cv2.getTrackbarPos('Contrast', 'Fly Tracker')
    exposure_slider = cv2.getTrackbarPos('Exposure', 'Fly Tracker')
    dark_min_slider = cv2.getTrackbarPos('Dark_min', 'Fly Tracker')
    min_area = cv2.getTrackbarPos('Min_area', 'Fly Tracker')
    rect_w = cv2.getTrackbarPos('Rect Width', 'Fly Tracker')
    rect_h = cv2.getTrackbarPos('Rect Height', 'Fly Tracker')
    center_x = cv2.getTrackbarPos('Center X', 'Fly Tracker')
    center_y = cv2.getTrackbarPos('Center Y', 'Fly Tracker')

    # --- Exposure adjustment ---
    exposure_value = -float(exposure_slider)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

    # --- Adjust brightness/contrast ---
    frame = cv2.convertScaleAbs(frame, alpha=contrast / 50.0, beta=(brightness - 50) * 2)

    # --- Dark_min adjustment (larger = darker) ---
    dark_min = 255 - int(dark_min_slider * 2.55)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, dark_min, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)

    # --- Compute rectangle boundaries ---
    x1 = int(center_x - rect_w // 2)
    x2 = int(center_x + rect_w // 2)
    y1 = int(center_y - rect_h // 2)
    y2 = int(center_y + rect_h // 2)

    # --- Draw rectangular arena ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    mid_x = (x1 + x2) // 2
    cv2.line(frame, (mid_x, y1), (mid_x, y2), (255, 255, 255), 2)

    # --- Define compartments ---
    boxes = [
        (x1, y1, mid_x, y2),  # Left (#1)
        (mid_x, y1, x2, y2)   # Right (#2)
    ]

    movement_flags = []

    # --- Analyze each compartment ---
    for idx, (bx1, by1, bx2, by2) in enumerate(boxes):
        roi = mask[by1:by2, bx1:bx2]
        if roi.size == 0:
            movement_flags.append(0)
            continue

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            movement_flags.append(0)
            continue

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            movement_flags.append(0)
            continue

        cx = int(M["m10"] / M["m00"]) + bx1
        cy = int(M["m01"] / M["m00"]) + by1
        centroid_history[idx].append((cx, cy))

        # --- Draw marker ---
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # --- Movement detection every 1 sec ---
        if len(centroid_history[idx]) <= frames_per_sec:
            movement_flags.append(0)
        else:
            first = centroid_history[idx][0]
            current = centroid_history[idx][-1]
            dist = np.hypot(current[0] - first[0], current[1] - first[1])
            movement_flags.append(1 if dist > 5 else 0)

    # --- Log every 10 seconds ---
    now = datetime.now()
    if (now - last_log_time).total_seconds() >= 10:
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        results.append([timestamp] + movement_flags)
        last_log_time = now

    # --- Labels ---
    offset_y = 20
    cv2.putText(frame, f"#1:{movement_flags[0]}", (center_x - rect_w//4 - 20, center_y - offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"#2:{movement_flags[1]}", (center_x + rect_w//4 - 20, center_y - offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- STOP button ---
    cv2.rectangle(frame, (stop_button["x1"], stop_button["y1"]),
                  (stop_button["x2"], stop_button["y2"]), (0, 0, 255), -1)
    cv2.putText(frame, "STOP", (stop_button["x1"] + 10, stop_button["y1"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Display ---
    cv2.imshow("Fly Tracker", frame)

    # --- Autosave every 60 logs ---
    if len(results) % 60 == 0 and len(results) > 0:
        pd.DataFrame(results, columns=["Time", "Compartment_1", "Compartment_2"]).to_csv(output_file, index=False)

    # --- Exit ---
    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button["pressed"]:
        print("🛑 Tracking stopped.")
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
pd.DataFrame(results, columns=["Time", "Compartment_1", "Compartment_2"]).to_csv(output_file, index=False)
print(f"✅ Final results saved to {output_file}")
