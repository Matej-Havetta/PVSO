import open3d as o3d
import numpy as np

# 1. Načítanie farebného a hĺbkového obrázku
color_raw = o3d.io.read_image("color.png")
depth_raw = o3d.io.read_image("depth.png")

# 2. Vytvorenie RGBD obrázku
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,
    depth_scale=1000.0,  # podľa Kinect V2: 1 jednotka = 1 mm
    depth_trunc=4.0,     # orezanie hĺbky za 4 m
    convert_rgb_to_intensity=False
)

# 3. Intrinzické parametre Kinect V2 (približné)
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=512, height=424, fx=365.456, fy=365.456, cx=254.878, cy=205.395)

# 4. Vytvorenie mračna bodov
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# 5. Pre správnu orientáciu
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

# 6. Zobrazenie
o3d.visualization.draw_geometries([pcd], window_name="Pôvodné mračno")

# Použijeme segmentáciu roviny pomocou RANSAC
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Detekovaná rovina: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Vytiahneme body, ktoré nie sú na rovine (očistené mračno)
pcd_clean = pcd.select_by_index(inliers, invert=True)

# Zobrazenie očisteného mračna
o3d.visualization.draw_geometries([pcd_clean], window_name="Očistené mračno (bez roviny)")
