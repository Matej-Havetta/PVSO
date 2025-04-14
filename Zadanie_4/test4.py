import numpy as np
import open3d as o3d
from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2

# Initialize Kinect runtime
kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
)

print("Waiting for frames...")

while True:
    if kinect.has_new_depth_frame() and kinect.has_new_color_frame():
        # Get depth and color frames
        depth_frame = kinect.get_last_depth_frame().reshape(
            (kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)
        ).astype(np.uint16)

        color_frame = kinect.get_last_color_frame().reshape(
            (kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)
        ).astype(np.uint8)

        break  # Capture one frame and stop

print("Frames captured, processing...")

# Map depth to camera space
depth_points = np.zeros((depth_frame.shape[0] * depth_frame.shape[1],), dtype=np.float32)
camera_points = kinect._mapper.MapDepthFrameToCameraSpace(depth_frame.flatten(), depth_points)

camera_points = np.ctypeslib.as_array(camera_points)
camera_points = camera_points.view(np.float32).reshape((-1, 3))

# Map depth to color space
color_points = np.zeros((depth_frame.shape[0] * depth_frame.shape[1] * 2,), dtype=np.float32)
mapped_color_points = kinect._mapper.MapDepthFrameToColorSpace(depth_frame.flatten(), color_points)
mapped_color_points = np.ctypeslib.as_array(mapped_color_points)
mapped_color_points = mapped_color_points.view(np.float32).reshape((-1, 2))

# Create a point cloud
points = []
colors = []

for i in range(len(camera_points)):
    x, y, z = camera_points[i]
    if np.isfinite(z) and z > 0:
        col_x, col_y = mapped_color_points[i]
        col_x = int(np.round(col_x))
        col_y = int(np.round(col_y))
        if 0 <= col_x < color_frame.shape[1] and 0 <= col_y < color_frame.shape[0]:
            color = color_frame[col_y, col_x, :3] / 255.0
            points.append([x, y, z])
            colors.append(color)

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

# Save to file
o3d.io.write_point_cloud("kinect_pointcloud.ply", pcd)
print("Point cloud saved as 'kinect_pointcloud.ply'")
