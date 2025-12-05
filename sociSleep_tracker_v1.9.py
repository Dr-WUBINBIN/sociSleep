# About sociSleep_tracker.
# sociSleep_tracker_v1.9
# developed by Binbin Wu Ph.D.
# Ja Lab, UF Scripps Institute, University of Florida
# © 2025. All rights reserved.

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from collections import deque
import time
import sys

# === Config ===
OUTPUT_FILENAME = "fly_movement_data.csv"
cam_index = 1  # change if LifeCam is not 1
INIT_FRAMES = 5   # number of frames to average during initialization
MAX_MERGE_FACTOR = 0.85  # merged blob threshold = (areaB+areaC)*factor
movement_threshold = 3

# === State ===
fly_area_B = None
fly_area_C = None
merged_area_threshold = None

# Streaming CSV setup
output_file = os.path.join(os.getcwd(), OUTPUT_FILENAME)
csv_header_written = False

# open camera
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open LifeCam (index {}). Try changing cam_index.".format(cam_index))
    sys.exit(1)
print("✅ LifeCam opened via DirectShow (index {}).".format(cam_index))

# verify the actual dimensions the camera is using
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📐 Camera reports actual resolution: {actual_width}x{actual_height}")

# open camera settings dialog (native DirectShow property page).
def open_camera_settings_dialog_callback(state, userdata):
    try:
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
        time.sleep(0.5)
    except Exception as e:
        print("⚠️ Could not open property dialog via OpenCV:", e)
        return

# measure FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0
frames_per_sec = int(round(fps))
print(f"🎞️ Measured FPS: {fps:.2f} → using {frames_per_sec} for movement windows.")

# UI & Trackbars
cv2.namedWindow("Fly Tracker", cv2.WINDOW_NORMAL)
cv2.namedWindow('Arena Settings', cv2.WINDOW_AUTOSIZE)
def nothing(x): pass

# Arena geometry sliders (circular arenas)
cv2.createTrackbar('Left_CX', 'Arena Settings', 200, 640, nothing)
cv2.createTrackbar('Left_CY', 'Arena Settings', 240, 480, nothing)
cv2.createTrackbar('Left_R',  'Arena Settings', 150, 400, nothing)
cv2.createTrackbar('Right_CX','Arena Settings', 440, 640, nothing)
cv2.createTrackbar('Right_CY','Arena Settings', 240, 480, nothing)
cv2.createTrackbar('Right_R', 'Arena Settings', 150, 400, nothing)

# STOP/START and camera-setting button
stop_button = {"x1": 10, "y1": 10, "x2": 110, "y2": 50, "pressed": False}
settings_button = {"x1": 10, "y1": 60, "x2": 110, "y2": 100}
start_button = {"x1": 130, "y1": 10,"x2": 230, "y2": 50,"pressed": False}
tracking_started = False

mouse_state = {"clicks": [], "initialized": False}
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # STOP button
        if stop_button["x1"] <= x <= stop_button["x2"] and stop_button["y1"] <= y <= stop_button["y2"]:
            stop_button["pressed"] = True
            return
        # Camera-setting button
        if settings_button["x1"] <= x <= settings_button["x2"] and settings_button["y1"] <= y <= settings_button["y2"]:
            open_camera_settings_dialog_callback(None, None)
            return
        
        # initialization clicks
        if start_button["x1"] <= x <= start_button["x2"] and start_button["y1"] <= y <= start_button["y2"]:
            start_button["pressed"] = True
            print("START button pressed")
            return

cv2.setMouseCallback("Fly Tracker", mouse_callback)

# tracking variables
frame_counter = 0
max_hist = frames_per_sec * 2
fly_histories = {"A": deque(maxlen=max_hist), "B": deque(maxlen=max_hist), "C": deque(maxlen=max_hist)}
kalman_filters = {"A": None, "B": None, "C": None}
last_measurement_time = {"A": None, "B": None, "C": None}
last_log_time = datetime.now()
window = frames_per_sec
print("Press 'q' or click STOP to quit. Click START button to initialize identities(averaged over frames).")

# --- Kalman helpers ---
def create_kalman(dt=1.0):
    kf = cv2.KalmanFilter(4,2)
    kf.transitionMatrix = np.array([[1,0,dt,0],
                                    [0,1,0,dt],
                                    [0,0,1,0],
                                    [0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def update_kalman(kf, x, y):
    try:
        kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
    except Exception:
        pass

def nearest_assignment(preds, dets, max_dist=150):
    mapping = {label: None for label in preds.keys()}
    if len(dets) == 0:
        return mapping, list(range(len(dets)))
    costs = []
    for label, p in preds.items():
        if p is None:
            continue
        for i, d in enumerate(dets):
            dist = np.hypot(p[0]-d[0], p[1]-d[1])
            costs.append((dist, label, i))
    costs.sort(key=lambda x: x[0])
    used_labels, used_inds = set(), set()
    for dist, label, i in costs:
        if label in used_labels or i in used_inds:
            continue
        if dist <= max_dist:
            mapping[label] = dets[i]
            used_labels.add(label); used_inds.add(i)
    unmatched = [i for i in range(len(dets)) if i not in used_inds]
    return mapping, unmatched

def avg_position(list_pts):
    pts = [p for p in list_pts if p is not None]
    if not pts:
        return None
    return (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))

# Detection + filter
def filter_by_circle(mask, cx, cy, r):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if area < 4: continue
        M=cv2.moments(c)
        if M["m00"]==0: continue
        x=int(M["m10"]/M["m00"])
        y=int(M["m01"]/M["m00"])
        if (x-cx)**2 + (y-cy)**2 <= r*r:
            results.append((x,y,area))
    results.sort(key=lambda x:x[2], reverse=True)
    return results

# initialization
def initialize_tracking_circles(left_cx,left_cy,left_r,right_cx,right_cy,right_r, init_frames=INIT_FRAMES):
    """
    Gather detections over several frames and average them for robust init.
    Returns initA, initB, initC, areaA, areaB, areaC (None when missing).
    """
    print("Initializing: averaging detections for", init_frames, "frames...")
    a_pts, b_pts, c_pts = [], [], []
    a_areas, b_areas, c_areas = [], [], []
    frames_captured = 0

    # Capture several frames for stability
    for _ in range(init_frames):
        # read current frame from the live cap and preprocess
        ret, f = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        _, mask_local = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask_local = cv2.medianBlur(mask_local, 5)

        left_centroids = filter_by_circle(mask_local, left_cx,left_cy,left_r)
        right_centroids = filter_by_circle(mask_local, right_cx,right_cy,right_r)

        # choose largest for left (A)
        if left_centroids:
            cx, cy, area = left_centroids[0]
            a_pts.append((cx, cy))
            a_areas.append(area)

        # Right arena: choose top 2 largest areas
        if right_centroids:
            if len(right_centroids) >= 1:
                cx, cy, area = right_centroids[0] # Biggest
                b_pts.append((cx, cy))
                b_areas.append(area)
            if len(right_centroids) >= 2:
                cx, cy, area = right_centroids[1] # Second biggest
                c_pts.append((cx, cy))
                c_areas.append(area)
        frames_captured += 1
        time.sleep(0.02)  # small delay to let camera update

    if frames_captured == 0:
        print("⚠ Initialization frames not captured.")
        return None, None, None, None, None, None

    def mean_pt(pts):
        if not pts: return None
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (int(np.mean(xs)), int(np.mean(ys)))

    initA = mean_pt(a_pts)
    initB = mean_pt(b_pts)
    initC = mean_pt(c_pts)
    areaA = int(np.mean(a_areas)) if a_areas else None
    areaB = int(np.mean(b_areas)) if b_areas else None
    areaC = int(np.mean(c_areas)) if c_areas else None

    print("Initialization result: A:", initA, "areaA:", areaA, "B:", initB, "areaB:", areaB, "C:", initC, "areaC:", areaC)
    return initA, initB, initC, areaA, areaB, areaC

def init_kalman_filters(initA, initB, initC):
    """
    Create and seed Kalman filters for A, B, C when initial positions exist.
    If an init is None, leave the filter as None (will be created later when detection appears).
    """
    for label, init in zip(["A","B","C"], [initA, initB, initC]):
        if init is not None:
            kf = create_kalman(1.0/frames_per_sec)
            # statePost is 4x1: [x, y, vx, vy]
            kf.statePost = np.array([[np.float32(init[0])],
                                     [np.float32(init[1])],
                                     [np.float32(0.0)],
                                     [np.float32(0.0)]], dtype=np.float32)
            kalman_filters[label] = kf
            fly_histories[label].append(init)
            last_measurement_time[label] = datetime.now()
        else:
            kalman_filters[label] = None

def clamp_predict_in_circle(px, py, cx, cy, r):
    dx = px - cx
    dy = py - cy
    dist = (dx*dx + dy*dy) ** 0.5
    if dist > r:
        px = cx + dx/dist * r
        py = cy + dy/dist * r
    return int(px), int(py)

# === Main loop ===

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break
    frame_counter += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Read UI values ---
    left_cx = cv2.getTrackbarPos('Left_CX','Arena Settings')
    left_cy = cv2.getTrackbarPos('Left_CY','Arena Settings')
    left_r  = cv2.getTrackbarPos('Left_R','Arena Settings')
    right_cx= cv2.getTrackbarPos('Right_CX','Arena Settings')
    right_cy= cv2.getTrackbarPos('Right_CY','Arena Settings')
    right_r = cv2.getTrackbarPos('Right_R','Arena Settings')

    # --- Preprocess frame for detection ---
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 5)

    # --- Arena circles ---
    cv2.circle(frame, (left_cx,left_cy), left_r, (0,255,0), 2)
    cv2.circle(frame, (right_cx,right_cy), right_r, (0,255,0), 2)
    boxes = [
        ("left",  left_cx,  left_cy,  left_r),
        ("right", right_cx, right_cy, right_r)
    ]

    # --- Initialization ---
    if start_button["pressed"] and not tracking_started:
        print("Starting tracking initialization...")
        initA, initB, initC, areaA, areaB, areaC = initialize_tracking_circles(left_cx,left_cy,left_r, right_cx,right_cy,right_r)
        fly_area_B = areaB or areaA # Fallback if B was missing
        fly_area_C = areaC or areaA # Fallback if C was missing
        merged_area_threshold = (fly_area_B + fly_area_C) * MAX_MERGE_FACTOR
        print("merged_area_threshold set to", merged_area_threshold)
        
        init_kalman_filters(initA, initB, initC)
        tracking_started = True
        start_button["pressed"] = False

    # --- Detections in each circular arena ---
    detections_left = []
    detections_right = []
    # left
    left_centroids = filter_by_circle(mask, left_cx,left_cy,left_r)
    if left_centroids:
        cx, cy, area = left_centroids[0]
        detections_left.append((cx, cy, area))
    # right
    right_centroids = filter_by_circle(mask, right_cx, right_cy, right_r)
    # If merged_area_threshold is known and there is a single large blob, treat as merged
    if len(right_centroids) == 1 and merged_area_threshold is not None and right_centroids[0][2] >= merged_area_threshold:
        # merged blob --> do not add detection so assignment keeps predicting positions (or handle specially)
        is_merged = True
    else:
        is_merged = False
        # Right arena: Keep the top 2 largest blobs as detections (if they exist)
        if len(right_centroids) >= 1:
            cx, cy, area = right_centroids[0]
            detections_right.append((cx, cy, area))
        if len(right_centroids) >= 2:
            cx, cy, area = right_centroids[1]
            detections_right.append((cx, cy, area))

    # --- Predict from Kalman ---
    preds = {}
    for label in ["A","B","C"]:
        kf = kalman_filters.get(label)
        if kf is None:
            preds[label] = None
        else:
            # --- Timeout-based reset: disable Kalman if no measurement for > 4 sec ---
            if last_measurement_time[label] is not None:
                if (datetime.now() - last_measurement_time[label]).total_seconds() > 4:
                    # Drop the stale filter; wait for new measurement to reinitialize
                    kalman_filters[label] = None
                    preds[label] = None
                    continue
            try:
                p = kf.predict()
                px = float(p[0][0]) if isinstance(p[0], (list, np.ndarray)) else float(p[0])
                py = float(p[1][0]) if isinstance(p[1], (list, np.ndarray)) else float(p[1])
                # Clamp predicted coordinates per arena (A = left, B/C = right)
                if label == "A":
                    px, py = clamp_predict_in_circle(px, py, left_cx, left_cy, left_r)
                else:  # B or C
                    px, py = clamp_predict_in_circle(px, py, right_cx, right_cy, right_r)
                preds[label] = (px, py)
            except Exception:
                preds[label] = None

    # --- Assign left (A) ---
    if preds["A"] is not None:
        mapping_left, unmatched = nearest_assignment({"A": preds["A"]}, detections_left, max_dist=150)
        measA = mapping_left["A"]
        if measA is not None:
            update_kalman(kalman_filters["A"], measA[0], measA[1])
            fly_histories["A"].append(measA)
            last_measurement_time["A"] = datetime.now()
        else:
            if preds["A"] is not None and ((preds["A"][0]-left_cx)**2 + (preds["A"][1]-left_cy)**2 <= left_r**2):
                fly_histories["A"].append(preds["A"])
            else:
                fly_histories["A"].append(None)
    else:
        if len(detections_left)>=1 and kalman_filters["A"] is None:
            chosen = detections_left[0]
            kfA = create_kalman(1.0/frames_per_sec)
            kfA.statePost = np.array([[np.float32(chosen[0])],[np.float32(chosen[1])],[0],[0]], dtype=np.float32)
            kalman_filters["A"] = kfA
            fly_histories["A"].append((chosen[0], chosen[1]))
            last_measurement_time["A"] = datetime.now()

    # --- Assign right (B & C) ---
    preds_right = {}
    if preds["B"] is not None: preds_right["B"] = preds["B"]
    if preds["C"] is not None: preds_right["C"] = preds["C"]
    dets_points = [(d[0], d[1]) for d in detections_right]
    
    mapping_right, unmatched_r = nearest_assignment(preds_right, detections_right, max_dist=200)
    if "B" in mapping_right and mapping_right["B"] is not None:
        update_kalman(kalman_filters["B"], mapping_right["B"][0], mapping_right["B"][1])
        fly_histories["B"].append(mapping_right["B"])
        last_measurement_time["B"] = datetime.now()
    else:
        if preds["B"] is not None and ((preds["B"][0]-right_cx)**2 + (preds["B"][1]-right_cy)**2 <= right_r**2):
            fly_histories["B"].append(preds["B"])
        else:
            fly_histories["B"].append(None)
        if preds["B"] is None:
            if len(detections_right)>=1 and kalman_filters["B"] is None:
                chosen = detections_right[0]
                kfB = create_kalman(1.0/frames_per_sec)
                kfB.statePost = np.array([[np.float32(chosen[0])],[np.float32(chosen[1])],[0],[0]], dtype=np.float32)
                kalman_filters["B"] = kfB
                fly_histories["B"].append((chosen[0], chosen[1]))
                last_measurement_time["B"] = datetime.now()
                
    if "C" in mapping_right and mapping_right["C"] is not None:
        update_kalman(kalman_filters["C"], mapping_right["C"][0], mapping_right["C"][1])
        fly_histories["C"].append(mapping_right["C"])
        last_measurement_time["C"] = datetime.now()
    else:
        if preds["C"] is not None and ((preds["C"][0]-right_cx)**2 + (preds["C"][1]-right_cy)**2 <= right_r**2):
            fly_histories["C"].append(preds["C"])
        else:
            fly_histories["C"].append(None)
        if preds["C"] is None:
            if len(detections_right)==1 and kalman_filters["C"] is None and kalman_filters["B"] is not None:
                chosen = detections_right[0]
                kfC = create_kalman(1.0/frames_per_sec)
                kfC.statePost = np.array([[np.float32(chosen[0])],[np.float32(chosen[1])],[0],[0]], dtype=np.float32)
                kalman_filters["C"] = kfC
                fly_histories["C"].append((chosen[0], chosen[1]))
                last_measurement_time["C"] = datetime.now()

    # --- Visualization ---
    for (cx,cy,area) in detections_left:
        if not ((cx-left_cx)**2 + (cy-left_cy)**2 <= left_r**2):
            continue
        cv2.circle(frame, (cx,cy), 6, (0,0,0), 1)
    for (cx,cy,area) in detections_right:
        if not ((cx-right_cx)**2 + (cy-right_cy)**2 <= right_r**2):
            continue
        cv2.circle(frame, (cx,cy), 6, (0,0,0), 1)
    display_positions = {}
    for label in ["A","B","C"]:
        hist = fly_histories[label]
        if len(hist)>0 and hist[-1] is not None:
            pos = hist[-1]
            display_positions[label] = pos
            color = (0,0,255) if label=="A" else (255,0,0) if label=="B" else (0,255,0)
            cv2.drawMarker(frame, (int(pos[0]), int(pos[1])), color, cv2.MARKER_TILTED_CROSS, 20, 2)
            cv2.putText(frame, f"Fly{label} {int(pos[0])},{int(pos[1])}", (int(pos[0])+10, int(pos[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            display_positions[label] = (None,None)

    # --- Movement detection ---
    movement_flags = {}
    for label in ["A","B","C"]:
        hist = list(fly_histories[label])
        if len(hist)<2:
            movement_flags[label] = 0; continue
        recent = hist[-window:]
        past = hist[-2*window:-window] if len(hist)>=2*window else []
        recent_valid = [p for p in recent if p is not None]
        past_valid = [p for p in past if p is not None]
        if len(recent_valid) < window * 0.3 or len(past_valid) < window * 0.3:
            movement_flags[label] = 0; continue
        avg_recent = avg_position(recent_valid); avg_past = avg_position(past_valid)
        if avg_recent is None or avg_past is None:
            movement_flags[label] = 0; continue
        dist = np.hypot(avg_recent[0]-avg_past[0], avg_recent[1]-avg_past[1])
        movement_flags[label] = 1 if dist > movement_threshold else 0

    # --- data storing ---
    now = datetime.now()
    if (now - last_log_time).total_seconds() >= 1:
        # Ensure CSV header exists before first write
        if not csv_header_written:
            cols = [
                "Time",
                "A_x","A_y","A_movement",
                "B_x","B_y","B_movement",
                "C_x","C_y","C_movement"
            ]
            with open(output_file, "w") as f:
                f.write(",".join(cols) + "\n")
            csv_header_written = True

        row = [
            now.strftime("%Y-%m-%d %H:%M:%S"),
            display_positions["A"][0], display_positions["A"][1], movement_flags["A"],
            display_positions["B"][0], display_positions["B"][1], movement_flags["B"],
            display_positions["C"][0], display_positions["C"][1], movement_flags["C"]
        ]
            
        # Append row to CSV
        with open(output_file, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

        last_log_time = now

    # --- UI labels ---
    cv2.putText(frame, f"A_mov:{movement_flags.get('A',0)} B_mov:{movement_flags.get('B',0)} C_mov:{movement_flags.get('C',0)}",
                (left_cx + 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    # STOP button draw
    cv2.rectangle(frame, (stop_button["x1"], stop_button["y1"]), (stop_button["x2"], stop_button["y2"]), (0,0,255), -1)
    cv2.putText(frame, "Stop", (stop_button["x1"]+10, stop_button["y1"]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # camera-setting button draw
    cv2.rectangle(frame, (settings_button["x1"], settings_button["y1"]), (settings_button["x2"], settings_button["y2"]), (255, 100, 0), -1)
    cv2.putText(frame, "Camera", (settings_button["x1"] + 10, settings_button["y1"] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # START button draw
    cv2.rectangle(frame, (start_button["x1"], start_button["y1"]), (start_button["x2"], start_button["y2"]), (0, 255, 0), -1)
    cv2.putText(frame, "Start", (start_button["x1"] + 5, start_button["y1"] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Show
    cv2.imshow("Fly Tracker", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q') or stop_button["pressed"]:
        print("Stopping tracking...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Results saved to", output_file)
