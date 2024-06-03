import os
import sys
import math
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import fcl
from scipy.spatial.transform import Rotation as R
from trac_ik_python.trac_ik import IK
import open3d as o3d
import cv2
import pdb

file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../util')
sys.path.append(util_dir)
sys.path.append('/home/j0k/coral/ompl-1.5.2/py-bindings')
import ompl.base as ob
import ompl.util as ou
import ompl.geometric as og

from stl_reader import stl_reader
from obj_reader import obj_reader
import time


def global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, - coord2 + offset2)

def rotation_concat(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]

    return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                    -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]





class robot_arm_configuration:
    
    #create voxel grid represetation for each link
    def __init__(self, file_path, robot_offset, point_cloud):
        with open('../assets/urdf/ur5e/ur5e_mimic_real_gripper_linear_motion.urdf') as f:
            urdf_str = f.read()
        self.ik_solver_ = IK('base_link', 'wrist_3_link', urdf_string = urdf_str)

        #handle gripper separately
        gripper_parts = ['robotiq_85_base_link', 'inner_knuckle', 'inner_finger', 
                                         'outer_knuckle', 'outer_finger']
        gripper_points = set()
        gripper_translation = [[0, 0, 0], 
                                                     [0.013, 0, 0.069],
                           [0.047, 0, 0.115],
                           [0.030, 0, 0.063],
                           [0.062, 0, 0.061],
                           [-0.013, 0, 0.069],
                           [-0.047, 0, 0.115],
                           [-0.031, 0, 0.063],
                           [-0.062, 0, 0.061]]
        gripper_rotation = [[0, 0, 0, 1],
                                                [0, 0, 0, 1],
                                                [0, 0, 0, 1],
                                                [0, 0, 0, 1],
                                                [0, 0, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 1, 0]]
        self.collision_models_ = {}
        index_addon = np.array([0, 0, 0])
        gripper_vertices = np.array([]).reshape(0, 3)
        gripper_faces = np.array([]).reshape(0, 3)
        for t in range(len(gripper_parts)):
            part_mesh = stl_reader(file_path + '../' + gripper_parts[t] + '_coarse.STL')
            mesh = pv.read(file_path + '../' + gripper_parts[t] + '_coarse.STL')

            min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
            max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
            for tx, ty, tz in part_mesh.get_vertices():
                min_x = min(min_x, tx)
                min_y = min(min_y, ty)
                min_z = min(min_z, tz)
                max_x = max(max_x, tx)
                max_y = max(max_y, ty)
                max_z = max(max_z, tz)

            bounding_points = []
            for tx in range(math.floor(min_x / 0.01), math.ceil(max_x / 0.01) + 1):
                for ty in range(math.floor(min_y / 0.01), math.ceil(max_y / 0.01) + 1):
                    for tz in range(math.floor(min_z / 0.01), math.ceil(max_z / 0.01) + 1):
                        bounding_points.append([tx * 0.01, ty * 0.01, tz * 0.01])

            bounding_points_poly = pv.PolyData(bounding_points)
    
            select = bounding_points_poly.select_enclosed_points(mesh)
            selected_points = select['SelectedPoints']
            temp_link_point_set = set()

            points_inside = []

            local_translation_1, local_rotation_1 = None, None
            local_translation_2, local_rotation_2 = None, None

            if t != 0:

                local_translation_1 = np.array(gripper_translation[t])
                local_translation_2 = np.array(gripper_translation[t + 4])
                local_rotation_1 = R.from_quat(gripper_rotation[t])
                local_rotation_2 = R.from_quat(gripper_rotation[t + 4])

                part_mesh2 = stl_reader(file_path + '../' + gripper_parts[t] + '_coarse.STL')
                part_mesh.transform(local_rotation_1, local_translation_1)
                part_mesh2.transform(local_rotation_2, local_translation_2)

                #left side
                left_part_vertices = part_mesh.get_vertices()
                left_part_faces = part_mesh.get_faces()
                left_part_faces += index_addon
                vertex_count, _ = left_part_vertices.shape
                face_count, _ = left_part_faces.shape
                    
                gripper_vertices = np.concatenate((gripper_vertices, left_part_vertices), axis = 0)
                gripper_faces = np.concatenate((gripper_faces, left_part_faces), axis = 0)

                index_addon += vertex_count

                #right side
                right_part_vertices = part_mesh2.get_vertices()
                right_part_faces = part_mesh2.get_faces()
                right_part_faces += index_addon
                vertex_count, _ = right_part_vertices.shape
                face_count, _ = right_part_faces.shape
                gripper_vertices = np.concatenate((gripper_vertices, right_part_vertices), axis = 0)
                gripper_faces = np.concatenate((gripper_faces, right_part_faces), axis = 0)

                index_addon += vertex_count
            else:
                part_vertices = part_mesh.get_vertices()
                part_faces = part_mesh.get_faces()
                part_faces += index_addon
                vertex_count, _ = part_vertices.shape
                face_count, _ = part_faces.shape

                gripper_vertices = np.concatenate((gripper_vertices, part_vertices), axis = 0)
                gripper_faces = np.concatenate((gripper_faces, part_faces), axis = 0)

                index_addon += vertex_count
                
            
            for i in range(len(bounding_points)):
                if selected_points[i]:
                    if t == 0:
                        gripper_points.add(tuple(bounding_points[i]))
                    else:
                        left_copy, right_copy = bounding_points[i], bounding_points[i]
                        left_copy = local_rotation_1.apply(left_copy)
                        left_copy += local_translation_1
                        right_copy = local_rotation_2.apply(right_copy)
                        right_copy += local_translation_2
                        gripper_points.add(tuple(left_copy.tolist()))
                        gripper_points.add(tuple(right_copy.tolist()))

        gri_min_x, gri_min_y, gri_min_z = sys.maxsize, sys.maxsize, sys.maxsize
        gri_max_x, gri_max_y, gri_max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize

        for x, y, z in gripper_points:
            gri_min_x = min(gri_min_x, x)
            gri_min_y = min(gri_min_y, y)
            gri_min_z = min(gri_min_z, z)
            gri_max_x = max(gri_max_x, x)
            gri_max_y = max(gri_max_y, y)
            gri_max_z = max(gri_max_z, z)
        
        self.link_points_ = {}
        self.bounding_points_ = {}

        #dummy_stl = part_mesh
        #dummy_stl.vertices_ = gripper_vertices
        #gripper_faces = gripper_faces.astype(int)
        #dummy_stl.faces_ = gripper_faces
        #print(gripper_vertices.shape, gripper_faces.shape)
        #dummy_stl.write_to_file('gripper.stl')

        self.link_names_ = ['base', 'shoulder', 'upperarm', 'forearm', 
                                                'wrist1', 'wrist2', 'wrist3', 'camera_and_frame']
        self.default_rotation_ = [R.from_euler('x', 90, degrees = True),
                                                            R.from_euler('xy', [90, 180], degrees = True),
                                                            R.from_euler('xy', [180, 180], degrees = True),
                                                            R.from_euler('z', -180, degrees = True),
                                                            R.from_euler('x', -180, degrees = True),
                                                            R.from_euler('x', 90, degrees = True),
                                                            R.from_euler('z', -90, degrees = True),
                                                            R.from_euler('z', 180, degrees = True)]
        self.default_translation_ = [[0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, -0.138, 0],
                                                                 [0, -0.007, 0],
                                                                 [0, 0.127, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0],
                                                                 [0, 0, 0]]
        #self.feasibility_map_ = self.load_maps('./grasp_util/static_feasibility_map.txt')
        #self.move_distance_map_ = self.load_maps('./grasp_util/static_move_distance_map.txt')
        #self.feasibility_map_ = self.load_maps('static_feasibility_map.txt')
        #self.move_distance_map_ = self.load_maps('static_move_distance_map.txt')

        self.offset_ = robot_offset


        link_points = {}
        for t in range(len(self.link_names_)):
            link = self.link_names_[t]
            link_mesh = stl_reader(file_path + link + '.stl')

            min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
            max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
            for tx, ty, tz in link_mesh.get_vertices():
                min_x = min(min_x, tx)
                min_y = min(min_y, ty)
                min_z = min(min_z, tz)
                max_x = max(max_x, tx)
                max_y = max(max_y, ty)
                max_z = max(max_z, tz)

            bounding_points = []
            for tx in range(math.floor(min_x / 0.01), math.ceil(max_x / 0.01) + 1):
                for ty in range(math.floor(min_y / 0.01), math.ceil(max_y / 0.01) + 1):
                    for tz in range(math.floor(min_z / 0.01), math.ceil(max_z / 0.01) + 1):
                        bounding_points.append([tx * 0.01, ty * 0.01, tz * 0.01])

            bounding_points_poly = pv.PolyData(bounding_points)
    
            mesh = pv.read(file_path + link + '.stl')
            select = bounding_points_poly.select_enclosed_points(mesh)
            selected_points = select['SelectedPoints']
            temp_link_point_set = set()

            points_inside = []

            for i in range(len(bounding_points)):
                if selected_points[i]:
                    temp_points = np.array(bounding_points[i])
                    temp_points = self.default_rotation_[t].apply(temp_points)
                    temp_points += self.default_translation_[t]
                    temp_points = temp_points.tolist()
                    temp_link_point_set.add(tuple(temp_points))
                    
            self.link_points_[link] = temp_link_point_set

            temp_rotation = self.default_rotation_[t]
            temp_translation = self.default_translation_[t]
            
            link_mesh.transform(temp_rotation, temp_translation)

            vertices, faces = link_mesh.get_vertices(), link_mesh.get_faces()

            self.collision_models_[link] = [vertices, faces.astype(int)]

            # add bounding box for each joint
            bounding_points_no_rotation = np.array([[min_x, min_y, min_z], 
                                                    [max_x, min_y, min_z],
                                                    [max_x, max_y, min_z],
                                                    [min_x, max_y, min_z],
                                                    [min_x, min_y, max_z],
                                                    [max_x, min_y, max_z],
                                                    [max_x, max_y, max_z],
                                                    [min_x, max_y, max_z]])
            self.bounding_points_[link] = temp_rotation.apply(bounding_points_no_rotation) + temp_translation


        self.link_points_['gripper'] = gripper_points

        # add bounding box for gripper
        self.bounding_points_['gripper'] = np.array([[gri_min_x, gri_min_y, gri_min_z], 
                                                     [gri_max_x, gri_min_y, gri_min_z],
                                                     [gri_max_x, gri_max_y, gri_min_z],
                                                     [gri_min_x, gri_max_y, gri_min_z],
                                                     [gri_min_x, gri_min_y, gri_max_z],
                                                     [gri_max_x, gri_min_y, gri_max_z],
                                                     [gri_max_x, gri_max_y, gri_max_z],
                                                     [gri_min_x, gri_max_y, gri_max_z]])

        self.link_names_.append('gripper')

        self.collision_models_['gripper'] = [gripper_vertices, gripper_faces.astype(int)]

        self.fcl_models_ = []

        for link in self.link_names_:
            m = fcl.BVHModel()
            vertices, faces = self.collision_models_[link]
            m.beginModel(len(vertices), len(faces))
            m.addSubModel(vertices, faces)
            m.endModel()
            self.fcl_models_.append(m)

        self.point_cloud = point_cloud

        if point_cloud is None:
            # adding banana
            self.obj_mesh = []
            collision_mesh = obj_reader('../assets/urdf/ycb/006_mustard_bottle/textured_vhacd.obj')
            collision_mesh.add_offset([0.60,0,0.139])#/////////////////////////////////////////////////////////////////////////////
            # collision_mesh.add_offset([0.4,0.3,0.139])
            # collision_mesh.add_offset([0.3,-0.2,0.139])
            # collision_mesh.add_offset([0.435,0.12,0.139])

            face = collision_mesh.get_faces()
            verts = collision_mesh.get_vertices()
            o = fcl.BVHModel()
            o.beginModel(len(verts), len(face))
            o.addSubModel(verts, face)
            o.endModel()
            self.obj_mesh.append([verts, face])


    def constrained_linear_motion_planner(self, distance):
        target_translation = [distance, 0, 0.2550]
        target_rotation = [0, 0, 0, 1]

        converted_target_translation = global_coord_converter(target_translation[0], 
                                                                                                                    target_translation[1],
                                                                                                                    target_translation[2],
                                                                                                                    self.offset_[0],
                                                                                                                    self.offset_[1],
                                                                                                                    self.offset_[2])

        converted_target_rotation = rotation_concat([-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2], target_rotation)

        dof_result = None
        while True:
            seed_state = [0.0]*self.ik_solver_.number_of_joints

            dof_result = self.ik_solver_.get_ik(seed_state,
                                                                 converted_target_translation[0],
                                                                 converted_target_translation[1],
                                                                 converted_target_translation[2],
                                                                 converted_target_rotation[0],
                                                                 converted_target_rotation[1],
                                                                 converted_target_rotation[2],
                                                                 converted_target_rotation[3])

            if dof_result: break
        
        dof_result = list(dof_result)
        #dof_result[0] += math.pi/4
        print([round(x, 4) for x in dof_result])

        self.check_collision_models(dof_result)

        #plane_normal = np.array([0.0, 0.0, 1.0])
        #col_plane = fcl.Plane(plane_normal, 0)
        #plane_obj = fcl.CollisionObect(col_plane, fcl.Transform())


    def calculate_transform_from_angles(self, angles):
        #link1 pose
        trans1, rot1 = [0, 0, 0], [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]
        
        #link2 pose
        rot2_initial = [-0, -math.sqrt(2)/2, math.sqrt(2)/2, -0]
        rot2_new = R.from_euler('z', angles[0]).as_quat().tolist()
        rot2_final = rotation_concat(rot2_new, rot2_initial)
        trans2, rot2 = [0, 0, 0.1625], rot2_final
        accu = rot2_new
        
        #link3 pose
        rot3_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
        rot3_vector = R.from_quat(accu).apply([0, 1, 0])
        rot3_final = rotation_concat(accu, rot3_initial)
        rot3_new = R.from_rotvec(angles[1]*rot3_vector).as_quat().tolist()
        rot3_final = rotation_concat(rot3_new, rot3_final)
        trans3, rot3 = [0, 0, 0.1625], rot3_final
        accu = rotation_concat(rot3_new, accu)
        
        #link4 pose
        rot4_initial = [math.sqrt(2)/2, -0, math.sqrt(2)/2, -0]
        rot4_vector = rot3_vector
        rot4_final = rot3_final
        rot4_offset = R.from_quat(rot3_final).apply([0, 0, 0.425])
        rot4_new = R.from_rotvec(angles[2]*rot4_vector).as_quat().tolist()
        rot4_final = rotation_concat(rot4_new, rot4_final)
        trans4, rot4 = trans3 + rot4_offset, rot4_final
        accu = rotation_concat(rot4_new, accu)
        
        #link5 pose
        rot5_offset = R.from_quat(rot4_final).apply([0, -0.1333, 0.3915])
        rot5_initial = [0, -0, 1, 0]
        rot5_vector = rot4_vector
        rot5_final = rotation_concat(accu, rot5_initial)
        rot5_new = R.from_rotvec(angles[3]*rot5_vector).as_quat().tolist()
        rot5_final = rotation_concat(rot5_new, rot5_final)
        trans5, rot5 = trans4 + rot5_offset, rot5_final
        accu = rotation_concat(rot5_new, accu)
        
        #link6 pose
        rot6_offset = R.from_quat(rot5_final).apply([0, 0, 0])
        rot6_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, -0]
        rot6_final = rotation_concat(accu, rot6_initial)
        rot6_vector = [0, 0, -1]
        rot6_vector = R.from_quat(accu).apply(rot6_vector)
        rot6_new = R.from_rotvec(angles[4]*rot6_vector).as_quat().tolist()
        rot6_final = rotation_concat(rot6_new, rot6_final)
        trans6, rot6 = trans5 + rot6_offset, rot6_final
        accu = rotation_concat(rot6_new, accu)
        
        #link7 pose
        rot7_offset = R.from_quat(rot6_final).apply([0, -0.0996, 0])
        rot7_initial = [math.sqrt(2)/2, math.sqrt(2)/2, 0, 0]
        rot7_final = rotation_concat(accu, rot7_initial)
        rot7_vector = [0, 1, 0]
        rot7_vector = R.from_quat(accu).apply(rot7_vector)
        rot7_new = R.from_rotvec(angles[5]*rot7_vector).as_quat().tolist()
        rot7_final = rotation_concat(rot7_new, rot7_final)
        trans7, rot7 = trans6 + rot7_offset, rot7_final
        accu = rotation_concat(rot7_new, accu)
        
        #camera pose
        rot9_offset = R.from_quat(rot7_final).apply([0.1, 0, 0.03])
        rot9_final = rot7
        trans9, rot9 = trans7 + rot9_offset, rot9_final

    #gripper
        rot8_offset = R.from_quat(rot7_final).apply([0.086, 0, 0])
        rot8_initial = [0, math.sqrt(2)/2, math.sqrt(2)/2, 0]
        rot8_final = rotation_concat(accu, rot8_initial)
        trans8, rot8 = trans7 + rot8_offset, rot8_final

        
        return [[trans1+self.offset_, rot1],
                [trans2+self.offset_, rot2],
                [trans3+self.offset_, rot3],
                [trans4+self.offset_, rot4],
                [trans5+self.offset_, rot5],
                [trans6+self.offset_, rot6],
                [trans7+self.offset_, rot7],
                        [trans9+self.offset_, rot9],
                        [trans8+self.offset_, rot8]]

    def apply_transform(self, pose):
        rotation = [x[1] for x in pose]
        translation = [x[0] for x in pose]
        res_points = []
        
        for i in range(len(rotation)):
            link_name = self.link_names_[i]
            
            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])

            temp_weights = 1.0/len(self.link_points_[link_name])
        
            for point in self.link_points_[link_name]:
                point = np.array(point)
                point = temp_rotation.apply(point)
                point += temp_translation
                res_points.append((list(point), temp_weights))
        
        return res_points

    def check_collision_models(self, angles, obj_collision_model = None, scene_info = None, show_obj_axes = False):
        plotter = pv.Plotter()

        #construct scene mesh
        if scene_info != None:
            self.construct_scene_meshs(scene_info, plotter)

        soft_colors = ['#ee4035','#f37736','#fdf498','#7bc043','#0392cf']


        color_index = 0
        if obj_collision_model != None:
            #construct obj mesh
            for obj_info in obj_collision_model:
                cx, cy, cz, dx, dy, dz = obj_info
                obj_mesh = pv.Cube((cx, cy, cz), dx, dy, dz)
                plotter.add_mesh(obj_mesh)
                color_index += 1

        #construct robot mesh
        _ = self.construct_robot_meshs(angles, plotter)

        # point cloud add
        if self.point_cloud is not None:
            pcd_mesh = pv.PolyData(self.point_cloud)
            plotter.add_mesh(pcd_mesh, color= "blue")
        else:
            # or add obj
            obj_verts, obj_face = self.obj_mesh[0]
            face_counts, _ = obj_face.shape
            obj_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), obj_face), axis = 1).astype(int)
            obj_mesh = pv.PolyData(np.array(obj_verts), np.array(obj_faces))
            plotter.add_mesh(obj_mesh, color= "#FFFF00")

        # Add the axes on object
        if show_obj_axes:
            axes = pv.Axes(
                show_actor=True,
                actor_scale=0.5,  # Adjust the scale as needed
            )
            axes.actor.SetPosition(obj_mesh.center)
            plotter.add_actor(axes.actor)
            banana = plotter.show_bounds(grid='front', location='outer', all_edges=True, color= "#000000")
            plotter.show_grid(color= "#000000")

        plotter.camera_position = 'yz'
        plotter.set_background('white')
        plotter.show()

    def construct_scene_meshs(self, scene_info, plotter):
        #construct scene mesh
        if scene_info != None:
            ex, ey, ez, eh = scene_info
            #base
            base = pv.Cube((ex/2.0 + 0.3, 0, ez/2.0), ex, ey, ez)
            left_cover = pv.Cube((ex/2.0 + 0.3, -ey/2.0 + 0.015, eh/2.0 + ez), ex, 0.03, eh)
            right_cover = pv.Cube((ex/2.0 + 0.3, ey/2.0 - 0.015, eh/2.0 + ez), ex, 0.03, eh)
            cover = pv.Cube((ex/2.0 + 0.3, 0, ez + eh + 0.015), ex, ey, 0.03)
            plotter.add_mesh(base)
            plotter.add_mesh(left_cover)
            plotter.add_mesh(right_cover)
            plotter.add_mesh(cover)

    def construct_robot_meshs(self, angles, plotter):
        transform_data = self.calculate_transform_from_angles(angles)
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]

        #construct robot mesh
        robot_mesh_list = []
        for i in range(len(rotation)):
            link_name = self.link_names_[i]

            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])

            temp_vertices, temp_faces = self.collision_models_[link_name]
            face_counts, _ =  temp_faces.shape

            new_vertices = temp_rotation.apply(temp_vertices) + temp_translation
            plot_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), temp_faces), axis = 1).astype(int)
            temp_mesh = pv.PolyData(np.array(new_vertices), np.array(plot_faces))

            robot_mesh_list.append(temp_mesh)
            plotter.add_mesh(temp_mesh, color = '#FF6961')
        _ = plotter.add_axes(line_width = 5)

        return robot_mesh_list
    
    def get_bounding_box(self, angles, plotter= None, update_mesh= None):
        transform_data = self.calculate_transform_from_angles(angles)
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]

        bbox_list = []
        mesh_list = []
        for i in range(len(rotation)):
            link_name = self.link_names_[i]

            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])

            verts_no_rotations = self.bounding_points_[link_name]
            verts = temp_rotation.apply(verts_no_rotations) + temp_translation
            tris = np.array([[0, 1, 2],
                             [0, 2, 3],
                             [4, 5, 6],
                             [4, 6, 7],
                             [0, 1, 5],
                             [0, 5, 4],
                             [3, 2, 6],
                             [3, 6, 7],
                             [1, 5, 6],
                             [1, 6, 2],
                             [0, 4, 7],
                             [0, 7, 3]])
            
            # new box
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform()
            bbox_list.append(fcl.CollisionObject(m, t))

            if plotter is not None:
                new_col = np.ones([tris.shape[0], 1], dtype = int) * 3
                new_tris = np.concatenate((new_col, tris), 1)
                temp_mesh = pv.PolyData(verts, new_tris)
                mesh_list.append(temp_mesh)
                plotter.add_mesh(temp_mesh, color = '#FF6961')

            elif update_mesh is not None:
                new_col = np.ones([tris.shape[0], 1], dtype = int) * 3
                new_tris = np.concatenate((new_col, tris), 1)
                temp_mesh = pv.PolyData(verts, new_tris)
                update_mesh[i].points = temp_mesh.points

        return bbox_list, mesh_list

    def get_swept_volume(self, path_list, test_name, grasp_idx, frame_rate= 60, scene_info= None, visualize= False):
        # get angles per frame
        pos_list = []
        for path_idx in range(1, len(path_list)):
            start_pos = np.array(path_list[path_idx - 1])
            end_pos = np.array(path_list[path_idx])
            delta = (end_pos - start_pos) / frame_rate

            for i in range(frame_rate + 1):
                pos_list.append(start_pos + (delta * i))


        # create fcl manager
        swept_manager = fcl.DynamicAABBTreeCollisionManager()
        swept_manager.setup()

        if visualize:
            plotter = pv.Plotter()
            print("test_data/swept_animations/" + test_name + "_grasp" + str(grasp_idx) + ".gif")
            plotter.open_gif("test_data/swept_animations/" + test_name + "_grasp" + str(grasp_idx) + ".gif")

            _, mesh_list = self.get_bounding_box(pos_list[0], plotter= plotter)
            pcd_mesh = pv.PolyData(self.point_cloud)
            plotter.add_mesh(pcd_mesh, color= "blue")
            if scene_info is not None:
                self.construct_scene_meshs(scene_info, plotter)
            plotter.camera_position = 'yz'
            plotter.set_background('white')
        else:
            plotter = None

        # add swept volume
        for angles in pos_list:
            bbox_list, _ = self.get_bounding_box(angles, update_mesh= mesh_list)
            if visualize:
                plotter.write_frame()


        swept_manager.registerObjects(bbox_list)
        swept_manager.setup()

        if visualize:
            plotter.camera_position = 'yz'
            plotter.close()

    def update_robot_meshs(self, angles, robot_mesh_list):
        transform_data = self.calculate_transform_from_angles(angles)
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]

        for i in range(len(rotation)):
            link_name = self.link_names_[i]

            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])

            temp_vertices, temp_faces = self.collision_models_[link_name]
            face_counts, _ =  temp_faces.shape

            new_vertices = temp_rotation.apply(temp_vertices) + temp_translation
            plot_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), temp_faces), axis = 1).astype(int)
            temp_mesh = pv.PolyData(np.array(new_vertices), np.array(plot_faces))
            robot_mesh_list[i].points = temp_mesh.points

    def path_animation(self, path_list, test_name, grasp_idx, scene_info= None, frame_rate= 50):
        # get angles per frame
        pos_list = []
        for path_idx in range(1, len(path_list)):
            start_pos = np.array(path_list[path_idx - 1])
            end_pos = np.array(path_list[path_idx])
            delta = (end_pos - start_pos) / frame_rate

            for i in range(frame_rate + 1):
                pos_list.append(start_pos + (delta * i))

        # init mesh
        plotter = pv.Plotter()
        robot_mesh_list = self.construct_robot_meshs(path_list[0], plotter)
        pcd_mesh = pv.PolyData(self.point_cloud)
        plotter.add_mesh(pcd_mesh, color= "blue")
        if scene_info is not None:
            self.construct_scene_meshs(scene_info, plotter)
            plotter.camera_position = 'yz'
        plotter.set_background('white')

        # animation start
        print("test_data/path_animations/" + test_name + "_grasp" + str(grasp_idx) + ".gif")
        plotter.open_gif("test_data/path_animations/" + test_name + "_grasp" + str(grasp_idx) + ".gif")
        num_frames = len(pos_list)
        plotter.write_frame()
        time.sleep(0.5)

        for frame_idx in range(num_frames):
            self.update_robot_meshs(pos_list[frame_idx], robot_mesh_list)
            plotter.add_mesh(pcd_mesh, color= "blue")
            plotter.write_frame()

        plotter.close() # problem with VTK-v9. It won't close the window.
        pv.close_all()


    def get_gripper_collision_model_at_pose(self, pose):
        rotation = np.array(pose[3:7])
        translation = np.array(pose[:3])
        r1 = R.from_quat(rotation)
        tf = fcl.Transform(r1.as_matrix(), translation)
        gripper_col = fcl.CollisionObject(self.fcl_models_[7], tf)
        return gripper_col

    def get_all_gripper_collision_models(self, start_pose, end_pose):
        distance = math.sqrt(sum([(x-y)**2 for x,y in zip(start_pose, end_pose)]))
        num_steps = round(distance/0.05)
        unit_step = [(y - x)/num_steps for x, y in zip(start_pose, end_pose)]
        start = start_pose[:]
        all_gripper_col = []
        for t in range(num_steps):
            start = [x + y for x, y in zip(start, unit_step)]
            all_gripper_col.append(self.get_gripper_collision_model_at_pose(start))
        return all_gripper_col

    def object_blocking_counter(self, dof_result, plane_model, static_env_models, flex_collision_models, ids):    
        pose_array = self.calculate_transform_from_angles(dof_result)
        ur5e_self_col = []
        #real_offset = np.array(state_tensor[0][:3])
        for t in range(8):
            rotation = np.array(pose_array[t][1])
            translation = np.array(pose_array[t][0])
            r1 = R.from_quat(rotation)
            tf = fcl.Transform(r1.as_matrix(), translation)
            ur5e_self_col.append(fcl.CollisionObject(self.fcl_models_[t], tf))


        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()

        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(ur5e_self_col)
        manager1.setup()

        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)

        for i in range(len(flex_collision_models)):
            temp_data = fcl.CollisionData(request = req)
            manager1.collide(flex_collision_models[i][0], temp_data, fcl.defaultCollisionCallback)
            if temp_data.result.is_collision and flex_collision_models[i][1] < ids:
                flex_collision_models[i][1] += ids


    def arm_collision_free(self, dof_result, plane_model, static_env_models, flex_collision_models):    
        pose_array = self.calculate_transform_from_angles(dof_result)
        ur5e_self_col = []
        #real_offset = np.array(state_tensor[0][:3])
        for t in range(8):
            rotation = np.array(pose_array[t][1])
            translation = np.array(pose_array[t][0])
            r1 = R.from_quat(rotation)
            tf = fcl.Transform(r1.as_matrix(), translation)
            ur5e_self_col.append(fcl.CollisionObject(self.fcl_models_[t], tf))


        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        self_collision_flag = False

        for t in range(8):
            if t != 0:
                if fcl.collide(ur5e_self_col[t], plane_model, request, result):
                    self_collision_flag = True
                    break
            col_with_other_part = False
            for q in range(8):
                if q < t-1 or q > t + 1:
                    if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
                        col_with_other_part = True
                        break
            if col_with_other_part:
                self_collision_flag = True
                break

        env_collision_flag = False
        manager1 = fcl.DynamicAABBTreeCollisionManager()
        manager1.registerObjects(ur5e_self_col)
        manager1.setup()

        manager2 = fcl.DynamicAABBTreeCollisionManager()
        manager2.registerObjects(static_env_models)
        manager2.setup()

        manager3 = fcl.DynamicAABBTreeCollisionManager()
        manager3.registerObjects([x for x in flex_collision_models])
        #manager3.registerObjects(flex_collision_models)
        manager3.setup()


        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
        rdata = fcl.CollisionData(request = req)
        manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)

        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
        rdata2 = fcl.CollisionData(request = req)
        manager1.collide(manager3, rdata2, fcl.defaultCollisionCallback)



        #for i in range(len(flex_collision_models)):
        #    temp_data = fcl.CollisionData(request = req)
        #    manager1.collide(flex_collision_models[i][0], temp_data, fcl.defaultCollisionCallback)
        #    if temp_data.result.is_collision and flex_collision_models[i][1] < ids:
        #        flex_collision_models[i][1] += ids

        if rdata.result.is_collision or rdata2.result.is_collision:
            env_collision_flag = True
        #print (self_collision_flag, env_collision_flag)

        return self_collision_flag == False and env_collision_flag == False



    
    def visualization(self, res_points):

            res_points_vis = pv.PolyData([x[0] for x in res_points])
            plotter = pv.Plotter()
            plotter.add_mesh(res_points_vis)
            plotter.show()

    def load_maps(self, file_name):
        kf_field = None
        with open(file_name, 'r') as f:
            data = f.readlines()
            x_dim, y_dim = [int(x) for x in data[0][:-1].split()]
            kf_field = []
            for line in data[1:]:
                temp_data = [float(x) for x in line[:-1].split()]
                kf_field.append(temp_data)
        #print (len(kf_field), len(kf_field[0]))
        kf_field = np.array(kf_field)
        return kf_field

    def shift_feasibility_map(self, scene):
        x_dim, y_dim = len(scene), len(scene[0])
        new_map = [[0]*y_dim for _ in range(x_dim)]
        for i in range(x_dim):
            index_x = round((i*0.01 - self.offset_[0])/0.01)
            index_x += 100
            for j in range(y_dim):
                index_y = round((j*0.01 - 0.6 - self.offset_[1])/0.01)
                index_y += 100
                if 0 <= index_x <= 200 and \
                    0 <= index_y <= 200:
                    new_map[i][j] = self.feasibility_map_[index_x][index_y]
        new_map = np.array(new_map)
        #plt.imshow(new_map, cmap='hot', interpolation='nearest')
        #plt.show()
        return new_map
        
    def shift_move_distance_map(self, scene, end_effector_offset):
        x_dim, y_dim = len(scene), len(scene[0])
        new_map = [[0]*y_dim for _ in range(x_dim)]
        for i in range(x_dim):
            index_x = round((i*0.01 - end_effector_offset[0])/0.01)
            index_x += 100
            for j in range(y_dim):
                index_y = round((j*0.01 - 0.6 - end_effector_offset[1])/0.01)
                index_y += 100
                if 0 <= index_x <= 200 and \
                    0 <= index_y <= 200:
                    new_map[i][j] = self.move_distance_map_[index_x][index_y]
        new_map = np.array(new_map)
        #plt.imshow(new_map, cmap='hot', interpolation='nearest')
        #plt.show()
        return new_map
    
    def grasp_verify(self, grasp_mat, cam_rot, cam_tran):
        # calc grasp_rot
        grasp_rot = grasp_mat[:3, :3]
        grasp_rot = R.from_quat(R.from_matrix(grasp_mat[:3,:3]).as_quat()) #convert 3x3 into rot
        axis_trans = np.array([
            [0, 0, 1],  # z' -> x
            [-1, 0, 0], # -x' -> y
            [0, -1, 0]  # -y' -> z
        ])
        rota = np.array(grasp_rot.as_matrix())
        grasp_rot = R.from_quat(R.from_matrix(np.dot(np.dot(axis_trans, rota), axis_trans.T)).as_quat())

        # get position of camera
        cam_rot_quat = R.from_quat(cam_rot)
    
        # target_rot_cam = R.from_matrix(grasp_mat[:3, :3])
        # target_rot_cam = np.array([target_rot_cam.as_quat()[0], target_rot_cam.as_quat()[1], target_rot_cam.as_quat()[2], target_rot_cam.as_quat()[3]])
        # grasp_rot = R.from_quat(quaternion_multiply([0.5, -0.5, 0.5, 0.5], target_rot_cam))
 
        # calc grasp_tran
        grasp_tran = grasp_mat[:3, 3]
        mat_rot = R.from_quat([-0.5, 0.5, -0.5, 0.5])
        grasp_tran = mat_rot.apply(grasp_tran)

        # grasp in global coord
        target_quat = (cam_rot_quat * grasp_rot).as_quat()
        target_pos = cam_tran + cam_rot_quat.apply(grasp_tran)

        # calc ik
        r_rot = R.from_quat([target_quat[0], target_quat[1], target_quat[2], target_quat[3]])
        cam_offset_vector = np.array([0.15, 0, 0])
        rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
        converted_coord = global_coord_converter(target_pos[0] - rot_cam_offset_vector[0],
                                                 target_pos[1] - rot_cam_offset_vector[1],
                                                 target_pos[2] - rot_cam_offset_vector[2], 
                                                 self.offset_[0], 
                                                 self.offset_[1],
                                                 self.offset_[2])
        converted_quat = rotation_concat([-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2], target_quat)
    
        seed_state = [0.0]*self.ik_solver_.number_of_joints
        dof_result = self.ik_solver_.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat[0], 
                                       converted_quat[1],
                                       converted_quat[2],
                                       converted_quat[3])
        
        return dof_result


def scene_registeration(points, scene):
    for po, weights in points:
        index1 = round(po[0]/0.01)
        index2 = round(po[1]/0.01)
        if index1 >= 0 and index1 <= 100 and \
             index2 >= -60 and index2 <= 60:
             scene[index1][index2 + 60] += weights

#this is called only once to generate the static txt file
def generate_move_distance_map():
    mv_field = [[0]*201 for _ in range(201)]
    for i in range(201):
        for j in range(201):
            loc_x = i * 0.01 - 1
            loc_y = j * 0.01 - 1
            distance = loc_x**2 + loc_y**2
            if distance <= 1:
                mv_field[i][j] = 1 - distance
    mv_field = np.array(mv_field)
    mv_field /= np.amax(mv_field)
    #print (np.amax(mv_field))
    plt.imshow(mv_field, cmap = 'hot', interpolation = 'nearest')
    plt.show()

    with open('static_move_distance_map.txt', 'w') as f:
        f.write('201 201\n')
        for i in range(201):
            line = ''
            for j in range(201):
                line += str(round(mv_field[i][j], 2))
                line += ' '
            line += '\n'
            f.write(line)
    f.close()
    return mv_field

#this is called only once to generate the static txt file
def generate_feasibility_map():
    input_file = '../arm_feasibility_field_backup.txt'
    offset_x, offset_y, offset_z = -10.0, 0, 0
    kf_field = [[0]*201 for _ in range(201)]
    distance_distri = [0]*120
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = [float(x) for x in line[:-1].split()]
            data[0] += offset_x
            data[1] += offset_y
            distance = math.sqrt(data[0]**2 + data[1]**2)
            index = round(distance/0.01)
            distance_distri[index] += 1
    for t in range(20):
        new_distance_distri = [0]*120
        for k in range(120):
            if distance_distri[k] != 0:
                window = distance_distri[max(k-2, 0): min(k+3, 120)]
                new_distance_distri[k] = sum(window)/len(window)
        distance_distri = new_distance_distri
    maxi = max(distance_distri)
    distance_distri = [x/maxi for x in distance_distri]
    plt.plot([x*0.01 for x in range(120)], distance_distri)
    plt.xlabel("distance towards base")
    plt.ylabel("likelihood")
    plt.grid()
    plt.show()
    kf_field = [[0]*201 for _ in range(201)]
    for i in range(201):
        for j in range(201):
            x_real, y_real = i*0.01-1, j*0.01-1
            index = round(math.sqrt(x_real**2 + y_real**2)/0.01)
            if 0 <= index < 120:
                kf_field[i][j] = distance_distri[index]
    kf_field = np.array(kf_field)
    kf_field /= np.amax(kf_field)
    with open('static_feasibility_map.txt', 'w') as f:
        f.write('201 201\n')
        for i in range(201):
            line = ''
            for j in range(201):
                line += str(round(kf_field[i][j], 2))
                line += ' '
            line += '\n'
            f.write(line)
    f.close()
    plt.imshow(kf_field, cmap='hot', interpolation='nearest')
    plt.show()


class ur5e_valid(ob.StateValidityChecker):
    def __init__(self, si, rac, plane_model, static_env_models, flex_collision_models, ids):
        super().__init__(si)
        self.rac_ = rac
        self.static_env_models_ = static_env_models
        self.plane_model_ = plane_model
        self.flex_collision_models_ = flex_collision_models
        self.ids_ = ids

    def isValid(self, dof_state):
        res = self.rac_.arm_collision_free(dof_state, self.plane_model_, self.static_env_models_, [])
        return res

class ur5e_valid_all(ob.StateValidityChecker):
    def __init__(self, si, rac, plane_model, static_env_models, flex_collision_models):
        super().__init__(si)
        self.rac_ = rac
        self.static_env_models_ = static_env_models
        self.plane_model_ = plane_model
        self.flex_collision_models_ = flex_collision_models

    def isValid(self, dof_state):
        res = self.rac_.arm_collision_free(dof_state, self.plane_model_, self.static_env_models_, self.flex_collision_models_)
        return res




class path_planner():
    def __init__(self, rac, plane_model, static_env_models):
        self.space_ = ob.RealVectorStateSpace(0)
        self.space_.addDimension(-0.48 - 1.26, -0.48 + 1.26)
        self.space_.addDimension(-1.19 - 0.98, -1.19 + 0.98)
        self.space_.addDimension(1.67 - 1.8, 1.67 + 1.8)
        self.space_.addDimension(-3.14, 3.14)
        self.space_.addDimension(-0.25 - 1.54, -0.25 + 1.54)
        self.space_.addDimension(-3.14, 3.14)

        self.si_ = ob.SpaceInformation(self.space_)

        self.rac_ = rac
        self.plane_model_ = plane_model
        self.static_env_models_ = static_env_models

    def plan_all(self, dof_start, dof_result, flex_collision_models):

        self.start_ = ob.State(self.space_)
        self.start_[0] = dof_start[0] 
        self.start_[1] = dof_start[1] 
        self.start_[2] = dof_start[2] 
        self.start_[3] = dof_start[3] 
        self.start_[4] = dof_start[4] 
        self.start_[5] = dof_start[5] 

        self.goal_ = ob.State(self.space_)
        self.goal_[0] = dof_result[0]
        self.goal_[1] = dof_result[1]
        self.goal_[2] = dof_result[2]
        self.goal_[3] = dof_result[3]
        self.goal_[4] = dof_result[4]
        self.goal_[5] = dof_result[5]

        validityChecker = ur5e_valid_all(self.si_, self.rac_, self.plane_model_, self.static_env_models_, flex_collision_models)
        self.si_.setStateValidityChecker(validityChecker)
        self.si_.setStateValidityCheckingResolution(0.0005)

        self.si_.setup()


        pdef = ob.ProblemDefinition(self.si_)
        pdef.setStartAndGoalStates(self.start_, self.goal_)
        shortestPathObjective = ob.PathLengthOptimizationObjective(self.si_)
        pdef.setOptimizationObjective(shortestPathObjective)

        optimizingPlanner = og.RRTConnect(self.si_)
        optimizingPlanner.setProblemDefinition(pdef)

        optimizingPlanner.setRange(1000000)
        optimizingPlanner.setup()

        
        temp_res = optimizingPlanner.solve(1)

        #print(temp_res.asString())
        if temp_res.asString() == 'Exact solution':
            path = pdef.getSolutionPath()

            path_simp = og.PathSimplifier(self.si_)

            res = path_simp.reduceVertices(path)

            path_list = []

            for t in range(path.getStateCount()):
                state = path.getState(t)
                path_list.append([state[0], state[1], state[2], state[3], state[4], state[5]])

            return path_list
        else:
            return None


    def plan(self, dof_start, dof_result, flex_collision_models, ids):

        self.start_ = ob.State(self.space_)
        self.start_[0] = dof_start[0] 
        self.start_[1] = dof_start[1] 
        self.start_[2] = dof_start[2] 
        self.start_[3] = dof_start[3] 
        self.start_[4] = dof_start[4] 
        self.start_[5] = dof_start[5] 

        self.goal_ = ob.State(self.space_)
        self.goal_[0] = dof_result[0]
        self.goal_[1] = dof_result[1]
        self.goal_[2] = dof_result[2]
        self.goal_[3] = dof_result[3]
        self.goal_[4] = dof_result[4]
        self.goal_[5] = dof_result[5]

        validityChecker = ur5e_valid(self.si_, self.rac_, self.plane_model_, self.static_env_models_, flex_collision_models, ids)
        self.si_.setStateValidityChecker(validityChecker)
        self.si_.setStateValidityCheckingResolution(0.001)

        self.si_.setup()


        pdef = ob.ProblemDefinition(self.si_)
        pdef.setStartAndGoalStates(self.start_, self.goal_)
        shortestPathObjective = ob.PathLengthOptimizationObjective(self.si_)
        pdef.setOptimizationObjective(shortestPathObjective)

        optimizingPlanner = og.RRTConnect(self.si_)
        optimizingPlanner.setProblemDefinition(pdef)

        optimizingPlanner.setRange(1000000)
        optimizingPlanner.setup()

        
        temp_res = optimizingPlanner.solve(30)

        #print(temp_res.asString())
        if temp_res.asString() == 'Exact solution':
            path = pdef.getSolutionPath()

            path_simp = og.PathSimplifier(self.si_)

            res = path_simp.reduceVertices(path)

            path_list = []

            for t in range(path.getStateCount()):
                state = path.getState(t)
                path_list.append([state[0], state[1], state[2], state[3], state[4], state[5]])
    

            for t in range(len(path_list)-1):
                source = path_list[t]
                target = path_list[t+1]
                steps = int(math.sqrt(sum([(x-y)**2 for x,y in zip(source, target)]))/0.005)
                delta_angle = np.array([(y - x)/steps for x, y in zip(source, target)])
                start =np.array(source)
                #print (steps)
                for k in range(steps+1):
                    source += delta_angle
                    self.rac_.object_blocking_counter(source, self.plane_model_, self.static_env_models_, flex_collision_models, ids)
                    #if flex_collision_models[1][1] >= ids:
                    #    print (source)

            return path
        else:
            return None

        
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0[3], quaternion0[0], quaternion0[1], quaternion0[2]
    w1, x1, y1, z1 = quaternion1[3], quaternion1[0], quaternion1[1], quaternion1[2]
    return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]

def write_to_pointcloud(color_image, depth_image, seg_image, cam_rotation, cam_translation):
        
        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_image)
        seg_raw = o3d.geometry.Image(seg_image)
       
        m, n = np.asarray(seg_raw).shape
        offset = np.array(cam_translation)
        rot = R.from_quat(cam_rotation)

        for i in range(m):
                for j in range(n):
                    if np.asarray(seg_raw)[i][j] != 1:
                        np.asarray(color_raw)[i][j][0] = 0
                        np.asarray(color_raw)[i][j][1] = 0
                        np.asarray(color_raw)[i][j][2] = 0
                        np.asarray(depth_raw)[i][j] = 0
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                  convert_rgb_to_intensity = False)
        param = o3d.camera.PinholeCameraIntrinsic(1280, 720, 910, 910, 640, 360)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image,
                        o3d.camera.PinholeCameraIntrinsic(
                        param))
        pcd_data = np.array(pcd.points, dtype = np.float32)
        
        pcd_data[:, [0,1,2]] = pcd_data[:, [2, 0, 1]]
        pcd_data[:, 1] *= -1
        pcd_data[:, 2] *= -1
        pcd_data = rot.apply(pcd_data)
        pcd_data += offset
        return pcd_data

def create_static_collision_model(scene_info):
    tx, ty, tz, th = scene_info
    print ("scene info: ", tx, ty, tz)

    #table
    col_table = fcl.Box(tx, ty, tz)
    trans_table = fcl.Transform(np.array([tx*0.5 + 0.3, 0.0, tz*0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)

    #left_cover
    col_left_cover = fcl.Box(tx, 0.03, th)
    trans_left_cover = fcl.Transform(np.array([tx*0.5 + 0.3, ty*0.5 - 0.015, tz + th*0.5]))
    left_cover_obj = fcl.CollisionObject(col_left_cover, trans_left_cover)

    #right_cover
    col_right_cover = fcl.Box(tx, 0.03, th)
    trans_right_cover = fcl.Transform(np.array([tx*0.5 + 0.3, - ty*0.5 + 0.015, tz + th*0.5]))
    right_cover_obj = fcl.CollisionObject(col_right_cover, trans_right_cover)

    #upper_cover
    col_upper_cover = fcl.Box(tx, ty, 0.03)
    trans_upper_cover = fcl.Transform(np.array([tx*0.5 + 0.3, 0.0, tz + th + 0.015]))
    upper_cover_obj = fcl.CollisionObject(col_upper_cover, trans_upper_cover)

    return [table_obj, left_cover_obj, right_cover_obj, upper_cover_obj]

def get_plan_path(rac, dof_result, scene_info):
    plane_normal = np.array([0,0,1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    if scene_info is not None:
        static_env_models = create_static_collision_model(scene_info)
    else:
        static_env_models = []

    pp = path_planner(rac, plane_obj, static_env_models)

    ur5e_start_dof = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    path = pp.plan_all(ur5e_start_dof, dof_result, static_env_models)

    return path

def grasp_path_check(file_path, test_data_root, grasp_root, test_name, scene_info= None, grasp_check= False):
    # get file name
    color_img_file_path = test_data_root + 'test_image/' + test_name + '.png'
    seg_img_file_path = test_data_root + 'test_seg_image/' + test_name + '.png'
    depth_img_file_path = test_data_root + 'test_depth_image/' + test_name + '.png'
    cam_file_path = test_data_root + 'test_npy/' + test_name + '.npy'
    grasp_file_path = grasp_root + 'predictions_' + test_name + '.npz'

    # read imgs
    color_img = cv2.imread(color_img_file_path, cv2.IMREAD_UNCHANGED)
    color_img = np.asarray(color_img)
    seg_img = cv2.imread(seg_img_file_path, cv2.IMREAD_UNCHANGED)
    seg_img = np.asarray(seg_img)
    depth_img = cv2.imread(depth_img_file_path, cv2.IMREAD_UNCHANGED)
    depth_img = np.asarray(depth_img)

    # get position of camera
    cam_datas = np.load(cam_file_path, allow_pickle=True)
    cam_rot = cam_datas.item()["cam_rot"]
    cam_tran = cam_datas.item()["cam_tran"]

    # get point cloud
    point_cloud = write_to_pointcloud(color_img, depth_img, seg_img, cam_rot, cam_tran)

    # configure env
    rac = robot_arm_configuration(file_path, np.array([0.0, 0, 0]), point_cloud)

    # reading grasp results
    grasp_datas = np.load(grasp_file_path, allow_pickle=True)
    grasp_score_idx = list(np.argsort(-grasp_datas["scores"].item()[1]))

    for i in grasp_score_idx:
        grasp_mat = grasp_datas["pred_grasps_cam"].item()[1][i]
        dof_result = rac.grasp_verify(grasp_mat, cam_rot, cam_tran)

        if dof_result is None: 
            print("skip imposible grasp")
            continue # skip imposible grasp

        if grasp_check:
            rac.check_collision_models(dof_result)

        # path planning & animation
        path_list = get_plan_path(rac, dof_result, scene_info)
        if path_list is not None:
            # rac.path_animation(path_list, test_name, grasp_idx=i, scene_info= scene_info, frame_rate= 30)
            rac.get_swept_volume(path_list, test_name, i, frame_rate=30, scene_info= scene_info, visualize=True)
            break
        # else:
            # rac.check_collision_models(dof_result)
            # input("No path generated. Grasp idx: ", i, ". Press enter to continue")


if __name__ == '__main__':
    # file_path = '../assets/urdf/ur5e/meshes/collision/'
    # rac = robot_arm_configuration(file_path, np.array([0.0, 0, 0]))
    # rac.get_bounding_box()
    # angles = [0, -math.pi/2, math.pi/2, -math.pi/2, 0, 0]

    # file paths
    file_path = '../assets/urdf/ur5e/meshes/collision/'
    test_data_root = 'test_data/'
    grasp_root = '../contact_graspnet/results/'
    test_name = 'pcd2'
    scene_info = [0.56, 0.86000001, 0.1, 0.5]
    # scene_info = None
    grasp_path_check(file_path, test_data_root, grasp_root, test_name, scene_info= scene_info, grasp_check= False)



    # plane_normal = np.array([0,0,1.0])
    # col_plane = fcl.Plane(plane_normal, 0)
    # plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    # pp = path_planner(rac, plane_obj, [])

    # ur5e_start_dof = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    # path = pp.plan_all(ur5e_start_dof, dof_result, [])
    # print(path)







    sys.exit(1)

    for t in range(20, 78):
        rac.constrained_linear_motion_planner(t*0.01)


    sys.exit(1)

    start_angle = [0, -math.pi/2, 0, -math.pi/2, 0, 0]

    cal_pose = rac.calculate_transform_from_angles(start_angle)

    rotation, translation = [], []

    for trans, rot in cal_pose: translation.append(trans), rotation.append(rot)

    rac.check_collision_models(start_angle)

    sys.exit(1)

    angles = [[0.118, 0.066, -1.133, -0.507, -1.524, 1.461],
                        [0.118, 0.064, -1.132, -0.502, -1.489, 1.445],
                        [0.118, 0.066, -1.134, -0.508, -1.537, 1.461],
                        [0.118, 0.076, -1.143, -0.557, -1.564, 1.438],
                        [0.118, 0.065, -1.133, -0.504, -1.527, 1.461]]

    dummy_scene = [[0]*121 for _ in range(101)]
    dummy_scene[30][60] = 5
    dummy_scene[30][40] = 2
    dummy_scene[30][80] = 1

    pl = path_planner(rac, [], [])

    shift_feasibility_map = rac.shift_feasibility_map(dummy_scene)
    shift_move_distance_map = rac.shift_move_distance_map(dummy_scene, [0.4, -0.1, 0])

    for an in angles:
        d_an = [(x - y)/1 for x,y in zip(an, start_angle)]
        temp_start = start_angle
        for t in range(1):
            temp_start = [temp_start[i] + d_an[i] for i in range(6)]
            #print (temp_start)
            #pl.plan(temp_start)
            cal_pose = rac.calculate_transform_from_angles(temp_start)

            rotation, translation = [], []
    
            for trans, rot in cal_pose:
                translation.append(trans)
                rotation.append(rot)

            #points = rac.apply_transform(rotation, translation)
            #scene_registeration(points, dummy_scene)
            #rac.visualization(points)
            #rac.check_collision_models(rotation, translation)

    #dummy_scene = np.array(dummy_scene)
    dummy_scene /= np.amax(dummy_scene)
    #dummy_scene = 1 - dummy_scene
    for i in range(101):
          for j in range(121):
              dummy_scene[i][j] += 0.5*(shift_feasibility_map[i][j]  +
                                      shift_move_distance_map[i][j])
    plt.imshow(dummy_scene, cmap='hot', interpolation='nearest')
    plt.show()


    dummy_scene /= np.amax(dummy_scene)

    conv_scene = [[0]*121 for _ in range(101)]
    for t in range(5, 101-5):
        for k in range(5, 121-5):
            temp_res = 0
            for tt in range(t-5, t+6):
                for kk in range(k-5, k+6):
                    temp_res += dummy_scene[tt][kk]
            conv_scene[t][k] = temp_res
    conv_scene = np.array(conv_scene)
    plt.imshow(conv_scene, cmap='hot', interpolation='nearest')
    plt.show()
                
