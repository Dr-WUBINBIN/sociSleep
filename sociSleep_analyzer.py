
# About sleep_analysis.
# sleep_analysis_v1.0.
# designed for analyzing output result from social_sleep_tracker_v1.0.
# developed by Binbin Wu Ph.D.
# Ja Lab, UF Scripps Institute, University of Florida
# © 2025. All rights reserved.

import pandas as pd
import numpy as np

class FlySleepAnalyzer:
    def __init__(self, file_path, sampling_interval_sec=1):
        self.data = pd.read_csv(file_path)
        print("Columns in the file:", self.data.columns)  # Check loaded columns
        
        # Keep same shape expectation as your original code:
        # activities is a NumPy array shaped (flies x timepoints)
        self.activities = self.data.iloc[:, 1:].fillna(0).values.T
        
        self.sampling_interval_sec = sampling_interval_sec
        
        # Number of samples that represent 30 minutes:
        # 30 minutes = 30 * 60 seconds. Divide by sampling interval (e.g., 1 s) -> 1800 samples.
        self.time_interval = int((30 * 60) / self.sampling_interval_sec)

        # optional: store number of flies / timepoints
        self.num_flies, self.num_points = self.activities.shape

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

    def write_results_to_csv(self, output_file):
        total_sleep_minutes = self.calculate_sleep()
        
        # Build dataframe with Fly names if available
        num_flies_calculated = total_sleep_minutes.shape[0]
        if custom_fly_ids and len(custom_fly_ids) == num_flies_calculated:
            fly_names = custom_fly_ids
        # Try to recover fly names from original CSV columns
        else:
            try:
                fly_names = list(self.data.columns[1:1 + total_sleep_minutes.shape[0]])
            except Exception:
                fly_names = [f"Fly_{i+1}" for i in range(total_sleep_minutes.shape[0])]
        
        # Transpose to intervals as rows (optional — choose layout you prefer)
        df = pd.DataFrame(total_sleep_minutes.T, columns=fly_names)
        
        # add Time (min) column for each 30-min bin: 0, 30, 60, ...
        df.insert(0, "Time (min)", np.arange(0, df.shape[0] * 30, 30))
        
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

# === Usage example ===
file_path = '/Users/binbin/Downloads/wDahomey_night.csv'    # Fill the movement data file path
output_file = 'sleep_time.csv'
custom_fly_ids = ["FlyA_sleep", "FlyB_sleep", "FlyC_sleep"] # Adjust this list length to match your data

analyzer = FlySleepAnalyzer(file_path, sampling_interval_sec=1)
analyzer.write_results_to_csv(output_file)

