import numpy as np
import open3d as o3d
import cv2


from sklearn.cluster import Birch
import matplotlib.pyplot as plt


# https://www.datatechnotes.com/2019/09/clustering-example-with-birch-method-in.html



################ ORIGINAL ###################
# whether to write in binary or text format
write_text = True

# load point cloud from file
file_path = '../../../Python projects/zad1/zadanie_4/TLS_kitchen.ply'

# read point cloud from file
pcd = o3d.io.read_point_cloud(file_path)
pcd.paint_uniform_color([0.6, 0.6, 0.6])
# visualize
o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)


############## BIRCH ##############################
segment_models = {}
segments = {}

max_plane_idx = 30
birch = Birch(threshold=0.05, n_clusters=1, branching_factor=50)

rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    segments[i] = rest.select_by_index(inliers)

    ############### BIRCH
    points = np.asarray(segments[i].points)
    birch.fit(points)
    labels = birch.predict(points)

    candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
    best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

    rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels==best_candidate)[0]))
    segments[i].paint_uniform_color(list(colors[:3]))

    print("pass", i, "/", max_plane_idx, "done.")


o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)],zoom=0.3199,front=[0.30159062875123849, 0.94077325609922868, 0.15488309545553303],lookat=[-3.9559999108314514, -0.055000066757202148, -0.27599999308586121],up=[-0.044411423633999815, -0.138726419067636, 0.98753122516983349])
cv2.waitKey(0)