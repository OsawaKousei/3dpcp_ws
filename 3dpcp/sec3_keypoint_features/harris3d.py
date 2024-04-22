import sys

import numpy as np
import open3d as o3d  # type: ignore
from keypoints_to_spheres import keypoints_to_spheres  # type: ignore


def compute_harris_3dkeypoints(pcd, radious=0.01, max_nn=10, threshold=0.001):
    pcd.estimate_normals(
        serch_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radious, max_nn=max_nn
        )
    )
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    harris = np.zeros(len(np.array(pcd.points)))
    is_active = np.zeros(len(np.array(pcd.points)), dtype=bool)

    # compute harris coordinates
    for i in range(len(np.array(pcd.points))):
        [num_nn, inds, _] = pcd_tree.search_knn_vector_3d(
            pcd.points[i], max_nn
        )
        pcd_normals = pcd.select_by_index(inds)
        pcd_normals.points = pcd.normals.points
        [_, covar] = pcd_normals.compute_mean_and_covariance()
        harris[i] = np.linalg.det(covar) / np.trace(covar)
        if harris[i] > threshold:
            is_active[i] = True

    # NMS
    for i in range(len(np.array(pcd.points))):
        if is_active[i]:
            [num_nn, inds, _] = pcd_tree.search_knn_vector_3d(
                pcd.points[i], max_nn
            )
            inds.pop(harris[inds].argmax())
            is_active[inds] = False
    keypoints = pcd.select_by_index(np.where(is_active)[0])
    return keypoints


# main
file_name = sys.argv[1]
print("Lodding a point cloud from", file_name)
pcd = o3d.io.read_point_cloud(file_name)
print(pcd)

keypoints = compute_harris_3dkeypoints(pcd)
print(keypoints)

pcd.paint_uniform_color([0.5, 0.5, 0.5])
o3d.visualization.draw_geometries([keypoints_to_spheres(keypoints), pcd])
