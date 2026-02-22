
# sociSleep_analyzer_v2.0.
# designed for analyzing output results from sociSleep_tracker_v2.0.
# developed by Binbin Wu Ph.D.
# Ja Lab, UF Scripps Institute, University of Florida
# © 2026. All rights reserved.

#Introduction
#Functionalities including:
# 1. Sleep calculation,
# 2. Distance traveled,
# 3. Long-distance (>800 pixels) and short-distance classification.

import pandas as pd
import numpy as np

class FlySleepAnalyzer:
    def __init__(self, file_path, sampling_interval_sec=1):
        self.data = pd.read_csv(file_path)
        print("Columns in the file:", self.data.columns)  # Check loaded columns
        
        # Expect columns:
        # Time, A_x, A_y, A_movement, B_x, B_y, B_movement, C_x, C_y, C_movement
        self.columns = self.data.columns

        # Identify fly prefixes automatically (A, B, C, ...)
        fly_prefixes = sorted(set(col.split("_")[0] for col in self.columns if "_" in col))

        self.fly_prefixes = fly_prefixes

        # Movement (0/1) per fly
        movement_cols = [f"{p}_movement" for p in fly_prefixes]
        self.activities = self.data[movement_cols].fillna(0).values.T

        # X/Y coordinates per fly
        self.x_coords = self.data[[f"{p}_x" for p in fly_prefixes]].values.T
        self.y_coords = self.data[[f"{p}_y" for p in fly_prefixes]].values.T
        
        self.sampling_interval_sec = sampling_interval_sec
        
        # Number of samples that represent 30 minutes:
        # 30 minutes = 30 * 60 seconds. Divide by sampling interval (e.g., 1 s) -> 1800 samples.
        self.time_interval = int((30 * 60) / self.sampling_interval_sec)

    def calculate_sleep(self, inactivity_threshold=0, sleep_min=5):
        """
        Returns total sleep time per fly per 30-min interval (in minutes).
        inactivity_threshold: value <= this is considered inactive
        consecutive_periods: minimum consecutive inactive samples to count as sleep bout
        """
        consecutive_periods = int((sleep_min * 60) / self.sampling_interval_sec)
        flies, total_points = self.activities.shape
        
        # Calculate number of 30-minute intervals using sample-based time_interval
        num_intervals = total_points // self.time_interval
        
        # Initialize sleep time array in units of data points (will convert to minutes)
        total_sleep_points = np.zeros((flies, num_intervals), dtype=int)
        
        # Calculate sleep per 30 minutes interval using your original logic
        for fly in range(flies):
            consecutive_count = 0
            interval_index = 0
            bin_count = 0  # how many 30-min bins have been processed
            
            for i in range(total_points):
                if self.activities[fly, i] <= inactivity_threshold:
                    consecutive_count += 1
                else:
                    if consecutive_count >= consecutive_periods:
                        total_sleep_points[fly, interval_index] += consecutive_count
                    consecutive_count = 0
                
                # Check if one 30-minute block (in samples) has passed
                if (i + 1) % self.time_interval == 0:
                    # if a sleep bout spans to the end of the block, add it
                    if consecutive_count >= consecutive_periods:
                        total_sleep_points[fly, interval_index] += consecutive_count
                    consecutive_count = 0
                    interval_index += 1
                    bin_count += 1
                    
                    # ✅ stop the loop when we’ve filled all intact bins
                    if bin_count >= num_intervals:
                        break
            
        # Convert points -> minutes: (points * sampling_interval_sec) / 60
        total_sleep_minutes = (total_sleep_points * self.sampling_interval_sec) / 60.0
        
        return total_sleep_minutes  # shape: (flies, num_intervals)

    def calculate_distance(self):
        """
        Returns total traveled distance per fly per 30-min interval.
        Distance is calculated as sum of Euclidean frame-to-frame displacement.
        """
        flies, total_points = self.x_coords.shape
        num_intervals = total_points // self.time_interval

        total_distance = np.zeros((flies, num_intervals), dtype=float)

        for fly in range(flies):
            interval_index = 0

            for i in range(1, total_points):
                x0, y0 = self.x_coords[fly, i - 1], self.y_coords[fly, i - 1]
                x1, y1 = self.x_coords[fly, i], self.y_coords[fly, i]
                
                if not (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1)):
                    dx = x1 - x0
                    dy = y1 - y0
                    total_distance[fly, interval_index] += np.sqrt(dx * dx + dy * dy)

                # advance 30-min bin
                if i % self.time_interval == 0:
                    interval_index += 1
                    if interval_index >= num_intervals:
                        break

        return total_distance  # shape: (flies, num_intervals)


    def calculate_long_short_walking(self, long_threshold=800):
        """
        Detect continuous walking events and classify them as:
        - long-distance walking: cumulative distance >= 10*radius
        - short-distance walking: cumulative distance < 10*radius

        Returns:
            long_distances: total distance of long-walking events per fly per 30-min bin
            short_distances: total distance of short-walking events per fly per 30-min bin
            long_events: number of long-walking events per fly per 30-min bin
        """
        flies, total_points = self.x_coords.shape
        num_intervals = total_points // self.time_interval

        long_distances = np.zeros((flies, num_intervals), dtype=float)
        short_distances = np.zeros((flies, num_intervals), dtype=float)
        long_events = np.zeros((flies, num_intervals), dtype=int)

        for fly in range(flies):
            interval_index = 0
            current_event_dist = 0.0
            in_event = False

            for i in range(1, total_points):
                x0, y0 = self.x_coords[fly, i - 1], self.y_coords[fly, i - 1]
                x1, y1 = self.x_coords[fly, i], self.y_coords[fly, i]
                moving = self.activities[fly, i] == 1

                valid_step = (
                    moving and
                    not (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1))
                )

                if valid_step:
                    dx = x1 - x0
                    dy = y1 - y0
                    step_dist = np.sqrt(dx * dx + dy * dy)
                    current_event_dist += step_dist
                    in_event = True
                else:
                    # event ends
                    if in_event:
                        if current_event_dist >= long_threshold:
                            long_distances[fly, interval_index] += current_event_dist
                            long_events[fly, interval_index] += 1
                        else:
                            short_distances[fly, interval_index] += current_event_dist
                        current_event_dist = 0.0
                        in_event = False

                # advance 30-min bin
                if i % self.time_interval == 0:
                    # close event at bin boundary
                    if in_event:
                        if current_event_dist >= long_threshold:
                            long_distances[fly, interval_index] += current_event_dist
                            long_events[fly, interval_index] += 1
                        else:
                            short_distances[fly, interval_index] += current_event_dist
                        current_event_dist = 0.0
                        in_event = False

                    interval_index += 1
                    if interval_index >= num_intervals:
                        break

        return long_distances, short_distances, long_events


    def write_results_to_csv(self, output_file):
        total_sleep_minutes = self.calculate_sleep()
        total_distance = self.calculate_distance()
        long_dist, short_dist, long_events = self.calculate_long_short_walking()
        
        # Automatically generate column names from fly prefixes
        fly_names_sleep = [f"{p}_sleep" for p in self.fly_prefixes]
        fly_names_dist = [f"{p}_distance" for p in self.fly_prefixes]
        
        df_sleep = pd.DataFrame(total_sleep_minutes.T, columns=fly_names_sleep)
        df_dist = pd.DataFrame(total_distance.T, columns=fly_names_dist)

        df_long = pd.DataFrame(
            long_dist.T,
            columns=[f"{p}_long_distance" for p in self.fly_prefixes]
        )
        df_short = pd.DataFrame(
            short_dist.T,
            columns=[f"{p}_short_distance" for p in self.fly_prefixes]
        )
        df_events = pd.DataFrame(
            long_events.T,
            columns=[f"{p}_long_events" for p in self.fly_prefixes]
        )
        df = pd.concat([df_sleep, df_dist, df_long, df_short, df_events], axis=1)
        
        # add Time (min) column for each 30-min bin: 0, 30, 60, ...
        df.insert(0, "Time (min)", np.arange(0, df.shape[0] * 30, 30))
        
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")



# === Usage example ===
file_path = '/Users/binbin/Documents/social_sleep_tracker/results/wDahomey/7/raw data/Jan09.csv' # The file you want to analyze
output_file = 'sleep_distance_Jan09.csv'

analyzer = FlySleepAnalyzer(file_path, sampling_interval_sec=1)
analyzer.write_results_to_csv(output_file)
