import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# Načítanie mračna bodov
pcd = o3d.io.read_point_cloud("TLS_kitchen.ply")

# Zobrazenie mračna bodov
o3d.visualization.draw_geometries([pcd])

# Detekcia roviny pomocou RANSAC
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

# Extrakcia bodov, ktoré neležia na rovine
pcd_clean = pcd.select_by_index(inliers, invert=True)

# Zobrazenie vyčisteného mračna bodov
o3d.visualization.draw_geometries([pcd_clean])

# Konverzia mračna bodov na numpy pole
points = np.asarray(pcd_clean.points)

# Aplikácia DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=10)
labels = dbscan.fit_predict(points)

# Priradenie farieb podľa klastrov
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Šum (nepriradené body) budú čierne
pcd_clean.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Zobrazenie segmentovaného mračna bodov
o3d.visualization.draw_geometries([pcd_clean])

# Definovanie počtu klastrov
k = 2
kmeans = KMeans(n_clusters=k)
labels = kmeans.fit_predict(points)

# Priradenie farieb podľa klastrov
colors = plt.get_cmap("tab10")(labels / k)
pcd_clean.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Zobrazenie segmentovaného mračna bodov
o3d.visualization.draw_geometries([pcd_clean])
