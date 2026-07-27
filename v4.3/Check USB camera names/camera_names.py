# Please revise camera_detector.py -> ALLOWED_CAMERA_NAMES = [ "LifeCam"] 
# based on REAL USB cacmera names, only need key words.
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = graph.get_input_devices()
print("detected cameras:")
for i, name in enumerate(devices):
	print(f"{i}:{name}")
