import pyvista as pv
from temp_reader import temp_reader
import numpy as np
import os
import sys
import open3d as o3d

#points = pv.wrap(pv.Sphere().points)
#print (type(pv.Sphere().points))
#surf = points.reconstruct_surface()
#
#print (type(surf))
#pl = pv.Plotter(shape = (1,2))
#pl.add_mesh(points)
#pl.add_title("PC")
#pl.subplot(0, 1)
#pl.add_mesh(surf, color = True, show_edges = True)
#pl.add_title("Mesh")
#pl.show()


collision_mesh = temp_reader('../assets/urdf/ycb/035_power_drill/textured.obj')
points = collision_mesh.get_vertices()
#points = []
#for j in range(100):
#    for k in range(100):
#        points.append([j*0.01, k*0.01, 0])
#        points.append([j*0.01, 0, k*0.01])
#        points.append([j*0.01, k*0.01, 0.99])
#        points.append([j*0.01, 0.99, k*0.01])
#        points.append([0, j*0.01, k*0.01])
#        points.append([0.99, j*0.01, k*0.01])
#
#new_points = []
#for x, y, z in points:
#    if x**2 + y**2 + z**2 >= 0.5:
#        new_points.append([x,y,z])
#
#counter = 0
#for i in range(100):
#    for j in range(100):
#        for k in range(100):
#            if abs((i*0.01)**2 + (j*0.01)**2 + (k*0.01)**2 - 0.5) < 0.01:
#                new_points.append([i*0.01, j*0.01, k*0.01])
#                counter += 1
#print (counter)
#points = new_points
#data = np.load('env210_sequence15_scene.npy')
#print (data.dtype)
#neighbor = []
#for i in range(-1, 2):
#    for j in range(-1, 2):
#        for k in range(-1, 2):
#            if i != 0 or j != 0 or k != 0:
#                neighbor.append([i,j,k])
#print (len(neighbor))
#x_dim, y_dim, z_dim, _ = data.shape
#points = []
#for i in range(x_dim):
#    for j in range(y_dim):
#        for k in range(z_dim):
#            if data[i][j][k] == 0:
#                counter = 0
#                flag = False
#                for di, dj, dk in neighbor:
#                    new_i = i + di
#                    new_j = j + dj
#                    new_k = k + dk
#                    if 0 <= new_i < x_dim and \
#                       0 <= new_j < y_dim and \
#                       0 <= new_k < z_dim:
#                        counter += 1
#                        if data[new_i][new_j][new_k][0] != 0:
#                            flag = True
#                if flag or counter < 26:
#                    points.append([i*0.01,j*0.01,k*0.01])
#
#print (len(points))
#
#pcd = o3d.geometry.PointCloud()
#
#pcd.points = o3d.utility.Vector3dVector(points)
#
#pcd = pcd.uniform_down_sample(1)
#
#pcd.estimate_normals()
#
#o3d.visualization.draw_geometries([pcd])
#
#radii = [0.005, 0.01, 0.02, 0.04]
#
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
#
#
#o3d.visualization.draw_geometries([mesh])
#
#print (len(mesh.vertices), len(mesh.triangles))
#print(np.asarray(mesh.triangles))
#print("here")

print (len(points))
points = pv.PolyData(points)
#print(type(points))
surf = points.reconstruct_surface()
#
print (surf.faces)
print (len(np.asarray(surf.points)))

faces_arr = []
start = 0
while start < len(surf.faces):
    num = surf.faces[start]
    faces_arr.append(surf.faces[start+1:start + num + 1])
    start += (num+1)
print (np.asarray(faces_arr))
#pl = pv.Plotter(shape = (1,2))
#pl.add_mesh(points)
#pl.add_title("PC")
#pl.subplot(0, 1)
#pl.add_mesh(surf, color = True, show_edges = True)
#pl.add_title("Mesh")
#pl.show()

