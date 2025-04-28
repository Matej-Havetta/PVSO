# Zadanie 4 - Spracovanie mračna bodov
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt

# === 1. Nacitame vlastne aj cudzie mračno bodov ===
# Vlastné mračno z Kinectu (uložené ako PLY)
pcd_kinect = o3d.io.read_point_cloud("kinect_cloud.ply")

# Cudzie mračno bodov (väčší dataset z internetu)
pcd_external = o3d.io.read_point_cloud("external_dataset.ply")

# === 2. Zobrazíme mračná bodov ===
print("Zobrazenie Kinect mračna")
o3d.visualization.draw_geometries([pcd_kinect])

print("Zobrazenie veľkého datasetu")
o3d.visualization.draw_geometries([pcd_external])

# === 3. Očistenie mračna pomocou RANSAC ===
def remove_plane_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    print(f"RANSAC: Rovinná plocha nájdená: {plane_model}")
    # Extrahujeme iba body, ktoré neležia na rovine
    pcd_clean = pcd.select_by_index(inliers, invert=True)
    return pcd_clean

# Očistíme Kinect mračno
pcd_kinect_clean = remove_plane_ransac(pcd_kinect)

# Zobrazíme očistené mračno
print("Zobrazenie očisteného Kinect mračna")
o3d.visualization.draw_geometries([pcd_kinect_clean])

# === 4. Clustering - DBSCAN ===
def cluster_dbscan(pcd, eps=0.05, min_points=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(points)
    labels = clustering.labels_
    print(f"Počet nájdených klastrov (DBSCAN): {len(set(labels)) - (1 if -1 in labels else 0)}")
    return labels

labels_dbscan = cluster_dbscan(pcd_kinect_clean)

# === 5. Clustering - KMeans ===
def cluster_kmeans(pcd, n_clusters=5):
    points = np.asarray(pcd.points)
    kmeans = KMeans(n_clusters=n_clusters).fit(points)
    labels = kmeans.labels_
    print(f"Počet nájdených klastrov (KMeans): {n_clusters}")
    return labels

labels_kmeans = cluster_kmeans(pcd_kinect_clean)

# === 6. Vizualizácia klastrov ===
def visualize_clusters(pcd, labels, title="Clustering"):
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # noise
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    print(f"Zobrazenie klastrov: {title}")
    o3d.visualization.draw_geometries([pcd], window_name=title)

# Zobrazíme DBSCAN klastry
visualize_clusters(pcd_kinect_clean, labels_dbscan, title="DBSCAN Clustering")

# Zobrazíme KMeans klastry
# Pre korektnú vizualizáciu potrebujeme znova načítať pôvodné mračno (aby sa neprepisali farby)
pcd_kinect_clean_reloaded = o3d.io.read_point_cloud("kinect_cloud.ply")
pcd_kinect_clean_reloaded = remove_plane_ransac(pcd_kinect_clean_reloaded)

visualize_clusters(pcd_kinect_clean_reloaded, labels_kmeans, title="KMeans Clustering")
