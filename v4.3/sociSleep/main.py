"""
sociSleep multi-camera fly tracker application entry point.

Automatically detects one or two USB cameras, launches per-camera trackers,
and displays a unified OpenCV dashboard.
"""

import sys

import cv2

from sociSleep.arena_config import ArenaSettings
from sociSleep.camera_detector import detect_usb_cameras
from sociSleep.dashboard import Dashboard
from sociSleep.design_selector import DesignSelector
from sociSleep.fly_tracker import FlyTracker


def main():
    """Detect cameras, start tracking, and run the dashboard event loop."""
    print("sociSleep Tracker — scanning for USB cameras...")
    camera_indices = detect_usb_cameras()

    if not camera_indices:
        print("❌ No USB cameras detected. Connect a camera and try again.")
        sys.exit(1)

    num_cameras = len(camera_indices)
    print(f"📷 Found {num_cameras} camera(s): indices {camera_indices}")

    design_selector = DesignSelector()
    arena_settings_list = [ArenaSettings(i + 1) for i in range(num_cameras)]
    trackers = [
        FlyTracker(
            camera_id=i + 1,
            camera_index=idx,
            arena_settings=arena_settings_list[i],
            design_selector=design_selector,
        )
        for i, idx in enumerate(camera_indices)
    ]

    dashboard = Dashboard(trackers)

    for tracker in trackers:
        tracker.start()

    print("Select plate design (slider or keys 1/2/3), then click START.")
    print("Press 'q' or click STOP to quit.")

    try:
        while not dashboard.should_stop:
            design_selector.refresh()
            dashboard.show()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                for tracker in trackers:
                    tracker.request_stop()
                break
            if not design_selector.is_locked:
                if key == ord("1"):
                    design_selector.set_by_index(0)
                elif key == ord("2"):
                    design_selector.set_by_index(1)
                elif key == ord("3"):
                    design_selector.set_by_index(2)
    finally:
        for tracker in trackers:
            tracker.request_stop()
            tracker.join(timeout=2.0)
        design_selector.save()
        design_selector.destroy()
        for arena in arena_settings_list:
            arena.save()
            arena.destroy()
        dashboard.destroy()
        for tracker in trackers:
            tracker.release()
        cv2.destroyAllWindows()

        for tracker in trackers:
            print(f"Results saved to {tracker.csv_logger.path}")

        print("Stopping tracking...")


if __name__ == "__main__":
    main()
