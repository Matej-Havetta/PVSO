import open3d as o3d
import numpy as np
import cv2



# Načítanie farbného a hĺbkového obrazu
color_raw = cv2.imread("color2.png")  # BGR formát
depth_raw = cv2.imread("depth2.png", cv2.IMREAD_UNCHANGED)  # 16-bit hĺbkový obraz

# Preráskaluj farbný obraz na veľkosť hĺbkového
depth_height, depth_width = depth_raw.shape
color_resized = cv2.resize(color_raw, (depth_width, depth_height))  # (šírka, výška)

# Prevod na Open3D formát
color_o3d = o3d.geometry.Image(cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB))
depth_o3d = o3d.geometry.Image(depth_raw)

# Vytvorenie RGBD obrazu
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d, convert_rgb_to_intensity=False
)
# # 2. Vytvorenie RGBD obrázku
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color_raw, depth_raw,
#     depth_scale=1000.0,  # podľa Kinect V2: 1 jednotka = 1 mm
#     depth_trunc=4.0,     # orezanie hĺbky za 4 m
#     convert_rgb_to_intensity=False
# )

# 3. Intrinzické parametre Kinect V2 (približné)
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=512, height=424, fx=143.3, fy=97.5, cx=80.7, cy=67.9)

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
