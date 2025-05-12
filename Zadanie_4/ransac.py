
import open3d as o3d

import cv2

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


############### RANSAC ###################
segment_models={}
segments = {}

max_plane_idx=30

rest=pcd
for i in range(max_plane_idx):
    ##colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold=0.01,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    segments[i].paint_uniform_color(list([0.8, 0.8, 0.8]))
    rest = rest.select_by_index(inliers, invert=True)
    rest.paint_uniform_color([1, 0, 0])
    print("pass",i,"/",max_plane_idx,"done.")

o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
cv2.waitKey(0)