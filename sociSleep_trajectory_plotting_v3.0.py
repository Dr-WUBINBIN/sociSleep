
# sociSleep_trajectory_plotting_v3.0.
# designed for plotting output results from sociSleep_tracker_v2.0.
# developed by Binbin Wu Ph.D.
# Ja Lab, UF Scripps Institute, University of Florida
# © 2026. All rights reserved.

#Introduction
#Functionalities including:

# 1. plot fly trajectory with speed display,
# 2. calculate high-speed trajectory length,
# 3. plot close-proximity trajectory (interaction-associated movement),
# 4. calculate close-proximity movement length and fraction.
# 5. calculate hight-speed close-proximity movement.
# 6. calculate co-movement time (min) / 30 min
# 7. calculate movement time (min) / 30 min


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


#--------------Co-movement in social arena per 30min--------------------
def comovement_per_30min(
    x1, y1, x2, y2,
    distance_thresh=30,
    sampling_interval_sec=1,
    immobile=3,
    bin_minutes=30):

    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    # Synchronous NaN removal
    valid = ~(np.isnan(x1) | np.isnan(y1) | np.isnan(x2) | np.isnan(y2))
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    if len(x1) < 2:
        return None

    # Inter-fly distance
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # speeds
    speed1 = np.sqrt(np.diff(x1)**2 + np.diff(y1)**2) / sampling_interval_sec
    speed2 = np.sqrt(np.diff(x2)**2 + np.diff(y2)**2) / sampling_interval_sec
    
    # align lengths with speed arrays
    proximity_mask = dist[:-1] < distance_thresh
    
    moving1 = speed1 > immobile
    moving2 = speed2 > immobile
    
    # true co-movement
    comovement_mask = proximity_mask & moving1 & moving2

    # Number of samples per bin
    samples_per_bin = int(bin_minutes * 60 / sampling_interval_sec)

    results = []

    for start in range(0, len(comovement_mask), samples_per_bin):
        end = min(start + samples_per_bin, len(comovement_mask))

        bin_mask = comovement_mask[start:end]

        comovement_seconds = np.sum(bin_mask) * sampling_interval_sec
        #fraction_comovement = np.mean(bin_mask)

        results.append({
            "bin_start_min": start * sampling_interval_sec / 60,
            "bin_end_min": end * sampling_interval_sec / 60,
            "comovement_min": comovement_seconds / 60,
            #"fraction_comovement": fraction_comovement
        })

    return pd.DataFrame(results)



#--------------movement time--------------------
import re

def movement_time_per_30min(
    df,
    bin_minutes=30,
    sampling_interval_sec=1):

    # Find all movement columns
    movement_cols = [
        col for col in df.columns
        if col.endswith("_movement")
    ]

    if len(movement_cols) == 0:
        raise ValueError("No *_movement columns found.")

    samples_per_bin = int(
        bin_minutes * 60 / sampling_interval_sec
    )

    results = []

    for col in movement_cols:

        movement = df[col].fillna(0).astype(int).values

        for start in range(
                0,
                len(movement),
                samples_per_bin):

            end = min(
                start + samples_per_bin,
                len(movement)
            )

            bin_data = movement[start:end]

            movement_seconds = (
                np.sum(bin_data)
                * sampling_interval_sec
            )

            movement_fraction = np.mean(bin_data)

            results.append({
                "fly": col.replace("_movement", ""),
                "bin_start_min":
                    start * sampling_interval_sec / 60,
                "bin_end_min":
                    end * sampling_interval_sec / 60,
                "movement_min":
                    movement_seconds / 60})

    return pd.DataFrame(results)





#-----------------------------------------------------------------------
#-------------------example usage---------------------------------------
#-----------------------------------------------------------------------


#-----------------------------------------------------------------------
#delete triple-quoted docstrings when using specific function.
#-----------------------------------------------------------------------



"""
#-------------------------------------------------
#------fly trajectory with speed---------
#-------------------------------------------------
df = pd.read_csv('/Users/binbin/Documents/social_sleep_tracker/results/chronic_isolation/12/raw_data/Chronic_April9.csv')

# Example: Fly A
# only for columns like A_x, A_y, or B_x,B_y, or C_x, C_y
# Daytime [0:43200]; Nighttime [43200:86400]
x = df["C_x"].iloc[0:86400]
y = df["C_y"].iloc[0:86400]


plot_fly_path_speed_colored_line(
    x, y,
    vmin=0,
    vmax=60
)
# define high speed threshold = 30 !!!!!!!!!!!!!!!!!!!!
speed_summary = quantify_speed(x, y, sampling_interval_sec=1, high_speed_thresh=30, immobile=3)
print('\n\nHigh-speed movement:')
print("\n",speed_summary)

"""



"""
#------------------------------------------------
#-------interaction-associated trajectory--------
#------------------------------------------------

# Example: Fly B, C
# only for columns like B_x,B_y, C_x, C_y
df = pd.read_csv('/Users/binbin/Documents/social_sleep_tracker/results/wCS/3/raw data/Dec18.csv')
proximity_value = 30  # the distance threshold for determining interaction-associated movement.

# Daytime [0:43200]; Nighttime [43200:86400]
xB = df["B_x"].iloc[0:86400]
yB = df["B_y"].iloc[0:86400]
xC = df["C_x"].iloc[0:86400]
yC = df["C_y"].iloc[0:86400]

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
"""



#------------------------------------------------
#----------Co-movement per 30 min----------
#------------------------------------------------

# Example: results from sociSleep_double_tracker
# only for columns like  A1_x, A1_y ;A2_x, A2_y OR B1_x,B1_y; B2_x,B2_y
df = pd.read_csv('/Users/binbin/Downloads/SocialDrive_June21_oldPC.csv')
proximity_value = 30  # the distance threshold for determining interaction-associated movement.

# Daytime [0:43200]; Nighttime [43200:86400]

x1 = df["A1_x"].iloc[0:86400]
y1 = df["A1_y"].iloc[0:86400]
x2 = df["A2_x"].iloc[0:86400]
y2 = df["A2_y"].iloc[0:86400]

'''
x1 = df["B1_x"].iloc[0:86400]
y1 = df["B1_y"].iloc[0:86400]
x2 = df["B2_x"].iloc[0:86400]
y2 = df["B2_y"].iloc[0:86400]
'''


df_comovement = comovement_per_30min(
    x1, y1, x2, y2,
    distance_thresh=30,
    sampling_interval_sec=1,
    immobile=3,
    bin_minutes=30
)

print('\n\nCo-movement per 30 min: saved to csv file')
df_comovement.to_csv('social-deprived co-movement results.csv')


"""
#------------------------------------------------
#----------movement time per 30 min----------
#------------------------------------------------

df = pd.read_csv('/Users/binbin/Downloads/SocialDriveJune21_newPC.csv')

movement_df = movement_time_per_30min(df)

movement_df.to_csv('movement time.csv')
print('\n\nMovement time results saved to csv file')
"""

