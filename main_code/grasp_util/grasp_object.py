import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv

def cast_index_to_real(i, j, k, unit, offset):
    digit_location = np.array([i,j,k], dtype = np.float32) * unit
    center = np.array([0.005, 0.005, 0.005])
    return digit_location + offset + center

def cast_real_to_index(digit_location, unit, offset):
    center = np.array([0.005, 0.005, 0.005])
    shifted_location = (digit_location - center - offset)/unit
    return (round(shifted_location[0]), round(shifted_location[1]), round(shifted_location[2]))

def cast_pc_to_index(pc):
    center = np.array([0.0005, 0.0005, 0.0005])
    pc = (digit_location - center)/0.001

def cast_index_to_pc(index):
    center = np.array([0.0005, 0.0005, 0.0005])
    return index * 0.001 + center



class grasp_object:
    def __init__(self, z_min):
        self.seen_ = set()
        #self.completion_ = {}
        self.pcd_ = None
        self.center_ = None
        self.min_x_, self.min_y_, self.min_z_ = sys.maxsize, sys.maxsize, z_min
        self.max_x_, self.max_y_, self.max_z_ = -sys.maxsize, -sys.maxsize, -sys.maxsize
        self.cx_, self.cy_ = None, None
        self.dx_, self.dy_, self.dz_ = None, None, None

    def add_seen(self, seen):
        for element in seen:
            if element not in self.seen_:
                self.seen_.add(element)

        points = [list(x) for x in self.seen_]

        for x, y, z in points:
            self.min_x_ = min(self.min_x_, x)
            self.max_x_ = max(self.max_x_, x)
            self.min_y_ = min(self.min_y_, y)
            self.max_y_ = max(self.max_y_, y)
            self.min_z_ = min(self.min_z_, z)
            self.max_z_ = max(self.max_z_, z)


        self.dx_ = self.max_x_ - self.min_x_
        self.dy_ = self.max_y_ - self.min_y_
        self.dz_ = self.max_z_ - self.min_z_

        self.cx_ = self.min_x_ + self.dx_/2.0
        self.cy_ = self.min_y_ + self.dy_/2.0

        points = np.array(points)

        self.center_ = np.mean(points, axis = 0)

        self.pcd_ = o3d.geometry.PointCloud()
        self.pcd_.points = o3d.utility.Vector3dVector(points)



    #def add_completion(self, comp):
    #    self.completion_ = {}
    #    average_colors = np.array([list(x) for x in self.seen_.values()])
    #    average_colors = np.mean(average_colors, axis = 0)
    #    for element in comp:
    #        if tuple(element) not in self.seen_:
    #            if tuple(element) not in self.completion_:
    #                self.completion_[tuple(element)] = list(average_colors)

    def get_seen(self):
        return self.seen_

    #def get_completion(self):
    #    return self.completion_

    def is_part_of_current_pc(self, new_pc):
        dists = self.pcd_.compute_point_cloud_distance(new_pc)
        flag = False
        for dis in dists:
            if dis <= 1e-3:
                flag = True
                break
        return flag

    def get_distance_between_center(self, new_center):
        return np.sum((self.center_ - new_center)**2)

    def get_collision_mesh(self):
        cx, cy, cz = self.min_x_ + self.dx_/2.0, self.min_y_ + self.dy_/2.0, self.min_z_ + self.dz_/2.0
        vertices = np.array([[cx - self.dx_/2.0, cy - self.dy_/2.0, cz - self.dz_/2.0], 
														 [cx - self.dx_/2.0, cy + self.dy_/2.0, cz - self.dz_/2.0],
														 [cx + self.dx_/2.0, cy + self.dy_/2.0, cz - self.dz_/2.0],
														 [cx + self.dx_/2.0, cy - self.dy_/2.0, cz - self.dz_/2.0],
														 [cx - self.dx_/2.0, cy - self.dy_/2.0, cz + self.dz_/2.0],
														 [cx - self.dx_/2.0, cy + self.dy_/2.0, cz + self.dz_/2.0],
														 [cx + self.dx_/2.0, cy + self.dy_/2.0, cz + self.dz_/2.0],
														 [cx + self.dx_/2.0, cy - self.dy_/2.0, cz + self.dz_/2.0]])
        faces = np.array([[0, 2, 1], [0, 2, 3],
												  [4, 6, 5], [4, 6, 7],
												  [5, 2, 1], [5, 2, 6],
												  [7, 2, 3], [7, 2, 6],
												  [4, 3, 0], [4, 3, 7],
												  [4, 1, 0], [4, 1, 5]])
        return vertices, faces






        #pcd = pcd.uniform_down_sample(1)
        #pcd.estimate_normals()
        #radii = [0.005, 0.01]
        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        #return np.asarray(mesh.vertices), np.asarray(mesh.triangles)

        #points = pv.PolyData(points)
        #surf = points.reconstruct_surface()

        #mesh_vertices = np.asarray(surf.points)

        #mesh_faces = []

        #start = 0

        #pl = pv.Plotter(shape = (1,2))
        #pl.add_mesh(points)
        #pl.add_title("PC")
        #pl.subplot(0, 1)
        #pl.add_mesh(surf, color = True, show_edges = True)
        #pl.add_title("Mesh")
        #pl.show()

        #while start < len(surf.faces):
        #    num = surf.faces[start]
        #    mesh_faces.append(surf.faces[start+1 : start + 1 + num])
        #    start += (num + 1)
        #mesh_faces = np.asarray(mesh_faces)

        #return mesh_vertices, mesh_faces
        
