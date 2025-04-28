import ctypes
import _ctypes
import sys
import numpy as np
import cv2
import pygame
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

# Inicializácia Kinect runtime
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

# Čakanie na dáta
while True:
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        # Získaj color frame
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)
        color_frame = color_frame[:, :, :3]  # Vynechať alfa kanál (BGR)

        # Ulož farbu
        color_frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("captured_color.png", color_frame_rgb)

        # Získaj depth frame
        depth_frame = kinect.get_last_depth_frame()
        depth_frame = depth_frame.reshape((424, 512)).astype(np.uint16)

        # Ulož depth
        cv2.imwrite("captured_depth.png", depth_frame)

        print("Color and Depth frame captured and saved.")
        break

# Uvoľnenie Kinectu
kinect.close()
