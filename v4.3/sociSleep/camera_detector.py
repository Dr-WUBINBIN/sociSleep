#Automatic USB camera detection.
import cv2

from pygrabber.dshow_graph import FilterGraph

ALLOWED_CAMERA_NAMES = [
    "LifeCam",  # matches similar LifeCam models
]

def detect_usb_cameras(max_cameras=2):
    """Return up to `max_cameras` DirectShow indices for supported USB cameras."""
    graph = FilterGraph()
    devices = graph.get_input_devices()

    print("\nSearching for supported USB cameras...")

    usb_indices = []

    for index, name in enumerate(devices):
        print(f"  [{index}] {name}")

        if any(keyword.lower() in name.lower() for keyword in ALLOWED_CAMERA_NAMES):
            usb_indices.append(index)
            print("      -> Selected")
        else:
            print("      -> Ignored")

    if len(usb_indices) == 0:
        raise RuntimeError(
            "No supported USB cameras found. Please connect a Microsoft LifeCam."
        )

    return usb_indices[:max_cameras]



def open_camera(camera_index):
    """
    Open a camera by index using DirectShow.

    Returns (VideoCapture, actual_width, actual_height, fps) or raises RuntimeError.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera (index {camera_index}). "
            "Ensure the device is connected and not in use."
        )
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, actual_width, actual_height, fps


# Detect USB cameras once when the program starts.
CAMERA_INDICES = detect_usb_cameras(max_cameras=2)
print(f"\nUsing camera indices: {CAMERA_INDICES}\n")

