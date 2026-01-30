
# sociSleep_trajectory_plotting_v2.0.
# designed for plotting output results from sociSleep_tracker_v2.0.
# developed by Binbin Wu Ph.D.
# Ja Lab, UF Scripps Institute, University of Florida
# © 2026. All rights reserved.

#Introduction
#Functionalities including:
# 1. plot fly trajectory (gray),
# 2. plot fly trajectory with speed display,
# 3. calculate high-speed trajectory length,
# 4. plot close-proximity trajectory (interaction-associated movement),
# 5. calculate close-proximity movement length and fraction.
# 6. calculate hight-speed close-proximity movement.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_fly_path(x, y, radius=80):

    # Convert to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    fig = plt.figure()
    ax = plt.subplot(3,3,5)

    # Plot trajectory
    ax.plot(x, y, color="black", linewidth=0.5, alpha=0.5)

    # Formatting
    ax.set_aspect("equal")
    ax.set_title("Fly trajectory", fontsize = 14)
    ax.set_xlabel("X (pixels)", fontsize = 14)
    ax.set_ylabel("Y (pixels)", fontsize = 14)

    plt.show()


def plot_fly_path_speed_colored_line(
    x, y,
    sampling_interval_sec=1,
    cmap="cividis",
    vmin=None,
    vmax=None,
    linewidth=0.5):

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        print("Not enough valid points to plot.")
        return

    # Compute speed (pixels/sec)
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) / sampling_interval_sec

    # Build line segments
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    # Color limits
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.percentile(speed, 95)

    #fig, ax = plt.subplots(figsize=(6, 6))
    fig = plt.figure()
    ax = plt.subplot(3,3,5)
    
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        linewidth=linewidth
    )
    lc.set_array(speed)
    ax.add_collection(lc)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X (pixels)", fontsize = 14)
    ax.set_ylabel("Y (pixels)", fontsize = 14)
    ax.set_title("Fly trajectory", fontsize = 14)

    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Speed (pixels / s)", fontsize = 14)

    plt.show()


def quantify_speed(x, y, sampling_interval_sec=1, high_speed_thresh=40, immobile = 3):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return None

    speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / sampling_interval_sec
    
    moving_speed = speed[speed > immobile]
    high_speed = speed[speed > high_speed_thresh]

    metrics = {
        "mean_speed": np.mean(moving_speed),
        "median_speed": np.median(moving_speed),
        "p90_speed": np.percentile(moving_speed, 90),
        "max_speed": np.max(moving_speed),
        "high_speed_total_time": len(high_speed),
        "high_speed_fraction": len(high_speed) / len(moving_speed),
    }

    return metrics



def plot_proximity_colored_path(
    xB, yB, xC, yC,
    distance_thresh=30,
    sampling_interval_sec=1,
    linewidth=0.5):
    xB = np.asarray(xB)
    yB = np.asarray(yB)
    xC = np.asarray(xC)
    yC = np.asarray(yC)

    # Remove NaNs synchronously
    valid = ~(np.isnan(xB) | np.isnan(yB) | np.isnan(xC) | np.isnan(yC))
    xB, yB, xC, yC = xB[valid], yB[valid], xC[valid], yC[valid]

    if len(xB) < 2:
        return

    # Inter-fly distance
    dist = np.sqrt((xB - xC)**2 + (yB - yC)**2)

    # Build segments for Fly B
    points = np.column_stack([xB, yB])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    # Color mask
    close_mask = dist[:-1] < distance_thresh

    colors = np.where(close_mask, "gold", "black")

    #fig, ax = plt.subplots(figsize=(6, 6))
    fig = plt.figure()
    ax = plt.subplot(3,3,5)
    
    lc = LineCollection(segments, colors=colors, linewidth=linewidth)
    ax.add_collection(lc)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Co-movement trajectory", fontsize = 14)
    ax.set_xlabel("X (pixels)", fontsize = 14)
    ax.set_ylabel("Y (pixels)", fontsize = 14)

    plt.show()

    return close_mask


def proximity_path_length(xB, yB, xC, yC, distance_thresh=30):
    xB = np.asarray(xB)
    yB = np.asarray(yB)
    xC = np.asarray(xC)
    yC = np.asarray(yC)

    # NaN removal
    valid = ~(np.isnan(xB) | np.isnan(yB) | np.isnan(xC) | np.isnan(yC))
    xB, yB, xC, yC = xB[valid], yB[valid], xC[valid], yC[valid]

    if len(xB) < 2:
        return None

    # Segment lengths for Fly B
    dx = np.diff(xB)
    dy = np.diff(yB)
    segment_length = np.sqrt(dx**2 + dy**2)

    # Inter-fly distance aligned to segments
    dist = np.sqrt((xB - xC)**2 + (yB - yC)**2)
    close_mask = dist[:-1] < distance_thresh

    red_length = np.sum(segment_length[close_mask])
    total_length = np.sum(segment_length)

    return {
        "proximity_distance": red_length,
        "fraction_proximity": red_length / total_length if total_length > 0 else 0
    }


# Calculate Fly B path length when Fly C is nearby AND Fly B is moving fast.
def proximity_highspeed_path_length(
    xB, yB, xC, yC,
    distance_thresh=30,
    high_speed_thresh=30,
    immobile=3,
    sampling_interval_sec=1):

    xB = np.asarray(xB)
    yB = np.asarray(yB)
    xC = np.asarray(xC)
    yC = np.asarray(yC)

    # Synchronous NaN removal
    valid = ~(np.isnan(xB) | np.isnan(yB) | np.isnan(xC) | np.isnan(yC))
    xB, yB, xC, yC = xB[valid], yB[valid], xC[valid], yC[valid]

    if len(xB) < 2:
        return None

    # Fly B segment lengths
    dx = np.diff(xB)
    dy = np.diff(yB)
    segment_length = np.sqrt(dx**2 + dy**2)

    # Fly B speed (pixels/sec)
    speedB = segment_length / sampling_interval_sec

    # Inter-fly distance (aligned to segments)
    dist = np.sqrt((xB - xC)**2 + (yB - yC)**2)
    proximity_mask = dist[:-1] < distance_thresh

    # Movement masks
    moving_mask = speedB > immobile
    high_speed_mask = speedB > high_speed_thresh

    # Combined condition
    combined_mask = proximity_mask & moving_mask & high_speed_mask

    proximity_highspeed_distance = np.sum(segment_length[combined_mask])
    total_movement_distance = np.sum(segment_length[moving_mask])

    return {
        "proximity_highspeed_distance": proximity_highspeed_distance,
        "fraction_of_movement": (
            proximity_highspeed_distance / total_movement_distance
            if total_movement_distance > 0 else 0
        )
    }






#-------------------example usage-----------------
df = pd.read_csv("/Users/binbin/Documents/social_sleep_tracker/results/wCS/9/raw data/Jan28.csv")

# Example: Fly A
# Daytime [0:43200]; Nighttime [43200:86400]
x = df["C_x"].iloc[43200:86400]
y = df["C_y"].iloc[43200:86400]

"""
#------fly trajectory---------
plot_fly_path(
    x, y,
    radius=80,         # arena radius in pixels 
)
"""

#------fly trajectory with speed---------

plot_fly_path_speed_colored_line(
    x, y,
    vmin=0,
    vmax=60
)
# define high speed threshold = 30 !!!!!!!!!!!!!!!!!!!!
speed_summary = quantify_speed(x, y, sampling_interval_sec=1, high_speed_thresh=30, immobile=3)
print('\n\nHigh-speed movement:')
print("\n",speed_summary)


#-------interaction-associated trajectory--------
df = pd.read_csv("/Users/binbin/Documents/social_sleep_tracker/results/wCS/9/raw data/Jan28.csv")
proximity_value = 30  # the distance threshold for determining interaction-associated movement.


# Example: Fly B, C
# Daytime [0:43200]; Nighttime [43200:86400]
xB = df["B_x"].iloc[43200:86400]
yB = df["B_y"].iloc[43200:86400]
xC = df["C_x"].iloc[43200:86400]
yC = df["C_y"].iloc[43200:86400]

mask = plot_proximity_colored_path(
    xB, yB, xC, yC,
    distance_thresh=proximity_value
)

metrics = proximity_path_length(xB, yB, xC, yC, distance_thresh=proximity_value)
print('\n\nCo-movement trajectory:')
print("\n",metrics)


ph_metrics = proximity_highspeed_path_length(
    xB, yB, xC, yC,
    distance_thresh=proximity_value,
    high_speed_thresh=30,
    immobile=3)
print("\n\nCo-movement + high-speed metrics:")
print("\n", ph_metrics)
