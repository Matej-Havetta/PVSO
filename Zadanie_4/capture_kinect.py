import ctypes
import cv2
import numpy as np
import os
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

# Inicializuj Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                         PyKinectV2.FrameSourceTypes_Depth)

print("Spusti prehrávanie .xef v Kinect Studio 2.0 a stlač 's' pre uloženie snímky")

while True:
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_image = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
        color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)

    if kinect.has_new_depth_frame():
        depth_frame = kinect.get_last_depth_frame()
        depth_image = depth_frame.reshape((424, 512)).astype(np.uint16)
        depth_colored = cv2.convertScaleAbs(depth_image, alpha=0.03)

    # Zobrazenie
    cv2.imshow('Color Frame', color_bgr)
    cv2.imshow('Depth Frame', depth_colored)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("color.png", color_bgr)
        cv2.imwrite("depth.png", depth_image)
        print("Uložené: color.png a depth.png")
        break
    elif key == 27:  # ESC
        break

kinect.close()
cv2.destroyAllWindows()
