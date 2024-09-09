import os
import sys
import math
import numpy as np
import pyvista as pv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import fcl
from scipy.spatial.transform import Rotation as R
from trac_ik_python.trac_ik import IK
import open3d as o3d
import cv2

file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../util')
mcts_dir = os.path.join(file_dir, '../MCTS')
sys.path.append(util_dir)
sys.path.append(mcts_dir)
sys.path.append('/home/j0k/coral/ompl-1.5.2/py-bindings')
import ompl.base as ob
import ompl.util as ou
import ompl.geometric as og

from stl_reader import stl_reader
from obj_reader import obj_reader
import MCTS_algo_ICRA as mct
import MCTS_algo_ICRA_OG as mct_OG
import MCTS_algo_ICRA_OG2 as mct_OG2
import MCTS_algo_ICRA_base1 as mct_base1
import MCTS_algo_ICRA_base2 as mct_base2
from rearrangement_planning_util_ICRA import write_result
import time
import copy

import pdb

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
    def __init__(self, file_path, robot_offset, scene_info, point_cloud=None, target_mesh=None, obstacles_num=0, target_pos=None):
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
        self.target_mesh = target_mesh

        self.obj_mesh = []
        self.obj_pos_list = []
        self.obstacles_num = obstacles_num
        self.obj_color = ['#00fffb', '#ff00dd', '#bf00ff', '#ffae00', '#59ff00', '#FFFF00', '#cffc03', '#0335fc', '#ffa3f3', '#a3ffb9', '#b2a3ff', '#ffa3a3', '#4688f2']

        assert not (target_mesh is None and obstacles_num != 0), "Target mesh was not given to compare collision with other obstacles"

        # add obstacles
        collision_mesh_list = []
        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()


        for i in range(obstacles_num):
            obstacles_mesh = obj_reader('../assets/urdf/ycb/002_master_chef_can/textured_vhacd.obj')

            if i == 0:
                target_verts, target_tris = target_mesh
                target_m = fcl.BVHModel()
                target_m.beginModel(len(target_verts), len(target_tris))
                target_m.addSubModel(target_verts, target_tris)
                target_m.endModel()
                target_t = fcl.Transform()
                target_colision_mesh = fcl.CollisionObject(target_m, target_t)
                collision_mesh_list.append(target_colision_mesh)
                objs_manager.registerObjects(collision_mesh_list)
                objs_manager.setup()

            is_collision = True
            while is_collision:
                # random obj placing
                tx = np.random.uniform(0.41, scene_info[0] + 0.28)
                ty = np.random.uniform(-scene_info[1]/2 + 0.1, scene_info[1]/2 - 0.1)
                tz = scene_info[2] + 0.001

                # if i == 0:
                #     tx = 0.41
                #     ty = 0.05
                # if i == 1:
                #     tx = 0.41
                #     ty = 0.32
                # if i == 2:
                #     tx = 0.41
                #     ty = -0.15
                # if i == 3:
                #     tx = 0.65
                #     ty = 0.02
                # if i == 4:
                #     tx = 0.55
                #     ty = 0.17
                # if i == 5:
                #     tx = 0.51
                #     ty = 0.25

                #    cyan      pink       purple     orange    green       yellow
                #     0         1           2          3         4            5


                temp_tris = obstacles_mesh.get_faces()
                temp_verts = obstacles_mesh.get_vertices()

                # new obj
                temp_m = fcl.BVHModel()
                temp_m.beginModel(len(temp_verts), len(temp_tris))
                temp_m.addSubModel(temp_verts, temp_tris)
                temp_m.endModel()
                temp_t = fcl.Transform([tx, ty, tz])
                temp_collision_mesh = fcl.CollisionObject(temp_m, temp_t)

                # check collision
                req = fcl.CollisionRequest()
                rdata = fcl.CollisionData(request = req)
                objs_manager.collide(temp_collision_mesh, rdata, fcl.defaultCollisionCallback)
                is_collision = rdata.result.is_collision # update collision status

                if not is_collision:
                    dist = np.sqrt((tx - target_pos[0])**2 + (ty - target_pos[1])**2)
                    if dist <= 0.16:
                        is_collision = True
                        print("target contact recalc")
                        continue

                    for obj in self.obj_pos_list:
                        dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                        # print("idx:", i, "dist:", dist)
                        if dist <= 0.16:
                            is_collision = True
                            print("recalc")
                            continue

            collision_mesh_list.append(temp_collision_mesh)
            objs_manager.registerObjects(collision_mesh_list)
            objs_manager.setup()
            obstacles_mesh.add_offset([tx, ty, tz])
            temp_tris = obstacles_mesh.get_faces()
            temp_verts = obstacles_mesh.get_vertices()
            self.obj_pos_list.append([tx, ty])
            self.obj_mesh.append([temp_verts, temp_tris])


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
    
    def add_all_obj_meshs(self, plotter):
        if self.point_cloud is not None:
            pcd_mesh = pv.PolyData(self.point_cloud)
            plotter.add_mesh(pcd_mesh, color= "blue")

        if self.target_mesh is not None:
            # add matched target object from file
            target_verts, target_face = self.target_mesh
            face_counts, _ = target_face.shape
            target_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), target_face), axis = 1).astype(int)
            target_mesh = pv.PolyData(np.array(target_verts), np.array(target_faces))
            plotter.add_mesh(target_mesh, color= "red")
        
        if self.obstacles_num:
            # add choosen obstacles from config
            for i in range(len(self.obj_mesh)):
                obstac_verts, obstac_face = self.obj_mesh[i]
                face_counts, _ = obstac_face.shape
                obstac_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), obstac_face), axis = 1).astype(int)
                obstac_mesh = pv.PolyData(np.array(obstac_verts), np.array(obstac_faces))
                plotter.add_mesh(obstac_mesh, color= self.obj_color[i])

    def check_collision_models(self, angles, obj_collision_model=None, scene_info=None, show_obj_axes=False):
        plotter = pv.Plotter()
        plotter.camera_position = [5.0, 0.0, 0.5]
        plotter.set_background('white')
        plotter.camera.zoom(1.0)
        plotter.camera.focal_point = (0.0, 0.0, 0.5)

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

        # add all the objects
        self.add_all_obj_meshs(plotter)

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

        # plotter.camera_position = 'yz'
        # plotter.set_background('white')
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

    def construct_robot_meshs(self, angles, plotter, w_target=None):
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
            plot_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), temp_faces), axis = 1).astype(int)
            
            if link_name == "gripper" and w_target is not None:
                temp_vertices = w_target[0]
                plot_faces = w_target[1]
            
            new_vertices = temp_rotation.apply(temp_vertices) + temp_translation
            temp_mesh = pv.PolyData(np.array(new_vertices), np.array(plot_faces))
            robot_mesh_list.append(temp_mesh)
            plotter.add_mesh(temp_mesh, color = '#FF6961')
            
        _ = plotter.add_axes(line_width = 5)
        return robot_mesh_list
    
    def update_robot_meshs_swept(self, angles, robot_mesh_list=None, plotter=None, w_target=None):
        transform_data = self.calculate_transform_from_angles(angles)
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]

        bbox_list = []
        for i in range(len(rotation)):
            link_name = self.link_names_[i]

            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])

            temp_vertices, temp_faces = self.collision_models_[link_name]
            face_counts, _ =  temp_faces.shape
            plot_faces = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), temp_faces), axis = 1).astype(int)

            if link_name == "gripper" and w_target is not None:
                temp_vertices = w_target[0]
                plot_faces = w_target[1]

            new_vertices = temp_rotation.apply(temp_vertices) + temp_translation
            temp_mesh = pv.PolyData(np.array(new_vertices), np.array(plot_faces))

            # pdb.set_trace()
            if robot_mesh_list is not None:
                robot_mesh_list[i].points = temp_mesh.points
                plotter.add_mesh(temp_mesh, color = '#FF6961')

            # new box
            m = fcl.BVHModel()
            m.beginModel(len(new_vertices), len(plot_faces))
            m.addSubModel(new_vertices, plot_faces)
            m.endModel()
            t = fcl.Transform()
            bbox_list.append(fcl.CollisionObject(m, t))
        
        return bbox_list
    
    def update_bounding_box(self, angles, w_target, plotter= None, update_mesh= None):
        transform_data = self.calculate_transform_from_angles(angles)
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]

        bbox_list = []
        mesh_list = []
        verts_list = []
        for i in range(len(rotation)):
            link_name = self.link_names_[i]

            temp_translation = translation[i]
            temp_rotation = R.from_quat(rotation[i])

            verts_no_rotations = self.bounding_points_[link_name]

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
            
            if link_name == "gripper" and w_target is not None:
                verts_no_rotations = w_target[0]
                tris = w_target[1]
            
            verts = temp_rotation.apply(verts_no_rotations) + temp_translation
            verts_list.append(verts)
            
            # new box
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform()
            bbox_list.append(fcl.CollisionObject(m, t))

            # if plotter is not None:
            if plotter is not None:
                new_col = np.ones([tris.shape[0], 1], dtype = int) * 3
                new_tris = np.concatenate((new_col, tris), 1)
                temp_mesh = pv.PolyData(verts, new_tris)
                mesh_list.append(temp_mesh)
                plotter.add_mesh(temp_mesh, color = '#FF6961')
                if update_mesh is not None:
                    new_col = np.ones([tris.shape[0], 1], dtype = int) * 3
                    new_tris = np.concatenate((new_col, tris), 1)
                    temp_mesh = pv.PolyData(verts, new_tris)
                    update_mesh[i].points = temp_mesh.points

        return bbox_list, mesh_list, verts_list
    
    def modify_grasp_bbox(self, dof_result,target_mesh, visualize=False):
        # calculate inverse transform
        transform_data = self.calculate_transform_from_angles(dof_result)
        griper_trans = transform_data[-1][0]
        griper_rot = R.from_quat(transform_data[-1][1]).as_matrix()

        # apply inverse transform to target obj
        inv_rot = griper_rot.T
        inv_trans = -inv_rot @ griper_trans
        inv_rot = R.from_quat(R.from_matrix(inv_rot).as_quat())

        # get bounding of target object
        verts, _ = target_mesh
        min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
        max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
        for tx, ty, tz in verts:
            min_x = min(min_x, tx)
            min_y = min(min_y, ty)
            min_z = min(min_z, tz)
            max_x = max(max_x, tx)
            max_y = max(max_y, ty)
            max_z = max(max_z, tz)

        target_verts = np.array([[min_x, min_y, min_z], 
                                 [max_x, min_y, min_z],
                                 [max_x, max_y, min_z],
                                 [min_x, max_y, min_z],
                                 [min_x, min_y, max_z],
                                 [max_x, min_y, max_z],
                                 [max_x, max_y, max_z],
                                 [min_x, max_y, max_z]])
        target_verts = inv_rot.apply(target_verts) + inv_trans
        
        # merge target bbox and grasp bbox
        grasp_verts = self.bounding_points_['gripper']
        grasp_tris = np.array([[0, 1, 2],
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
        target_tris = grasp_tris + len(grasp_verts)
        merge_vert = np.concatenate((grasp_verts, target_verts))
        merge_tris = np.concatenate((grasp_tris, target_tris))
        
        if visualize:
            threes = np.ones((len(merge_tris), 1), dtype=int) * 3
            vis_tris = np.concatenate((threes, merge_tris), axis=1)
            plotter = pv.Plotter()
            temp_mesh = pv.PolyData(merge_vert, vis_tris)
            plotter.add_mesh(temp_mesh, color = '#FF6961')
            plotter.camera_position = 'yz'
            plotter.set_background('white')
            plotter.show()

        return [merge_vert, merge_tris]
    
    def modify_grasp_mesh(self, dof_result, target_mesh, visualize=False):
        # calculate inverse transform
        transform_data = self.calculate_transform_from_angles(dof_result)
        griper_trans = transform_data[-1][0]
        griper_rot = R.from_quat(transform_data[-1][1]).as_matrix()

        # apply inverse transform to target obj
        inv_rot = griper_rot.T
        inv_trans = -inv_rot @ griper_trans
        inv_rot = R.from_quat(R.from_matrix(inv_rot).as_quat())

        # target mesh
        target_verts, target_faces = target_mesh
        face_counts, _ = target_faces.shape
        target_verts = inv_rot.apply(target_verts) + inv_trans
        target_tris = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), target_faces), axis = 1).astype(int)

        # grasp mesh
        grasp_verts, grasp_faces = self.collision_models_["gripper"]
        face_counts, _ = grasp_faces.shape
        grasp_tris = np.concatenate((np.array([3]*face_counts).reshape(face_counts, 1), grasp_faces), axis = 1).astype(int)
        target_tris[:, 1:4] += len(grasp_verts)

        merge_vert = np.concatenate((grasp_verts, target_verts))
        merge_tris = np.concatenate((grasp_tris, target_tris))
        
        if visualize:
            plotter = pv.Plotter()
            temp_mesh = pv.PolyData(np.array(merge_vert), np.array(merge_tris))
            plotter.add_mesh(temp_mesh, color = '#FF6961')
            plotter.camera_position = 'yz'
            plotter.set_background('white')
            plotter.show()

        return [merge_vert, merge_tris]

    def get_swept_volume(self, path_list, test_name='', grasp_idx=None, w_target=None, frame_rate=60, scene_info=None, animation=False, static_vi=False, with_scene=False):
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
        swept_verts = []

        if animation:
            plotter = pv.Plotter()
            print("SAVING: test_data/swept_animations/" + test_name + "_grasp_w_obj" + str(grasp_idx) + ".gif")
            plotter.open_gif("test_data/swept_animations/" + test_name + "_grasp_w_obj" + str(grasp_idx) + ".gif")

            # add init mesh
            _, mesh_list,_ = self.update_bounding_box(pos_list[0], w_target, plotter= plotter)
            self.add_all_obj_meshs(plotter)

            if scene_info is not None and with_scene:
                self.construct_scene_meshs(scene_info, plotter)
            plotter.camera_position = 'yz'
            plotter.set_background('white')
        else:
            plotter = None
            mesh_list = None

        if static_vi:
            plotter2 = pv.Plotter()
            self.construct_scene_meshs(scene_info, plotter2)
            self.add_all_obj_meshs(plotter2)
            plotter2.set_background('white')
        else:
            plotter2 = None

        # add swept volume
        for angles in pos_list:
            bbox_list, _, verts_list = self.update_bounding_box(angles, w_target, update_mesh= mesh_list, plotter=plotter2)
            swept_verts += verts_list
            if animation:
                plotter.write_frame()
            swept_manager.registerObjects(bbox_list)
            swept_manager.setup()

        if static_vi:
            plotter2.camera_position = 'yz'
            plotter2.show()
        if animation:
            plotter.close()

        return swept_manager, swept_verts
    
    def get_swept_center(self, swept_verts, scene_info, max_height):
        # filter swepts outside of box
        x_min = 0.3
        x_max = 0.3 + scene_info[0]
        y_min = -scene_info[1] / 2
        y_max = scene_info[1] / 2
        z_min = scene_info[2]
        z_max = scene_info[2] + (max_height / 100) + 0.01
        # z_max = scene_info[2] + 0.15
        main_swept = []
        for pos in swept_verts:
            for verts in pos:
                if verts[0] <= x_min or verts[0] >= x_max:
                    continue
                if verts[1] <= y_min or verts[1] >= y_max:
                    continue
                if verts[2] <= z_min or verts[2] >= z_max:
                    continue
                main_swept.append(verts)

        # get bbox of swept
        min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
        max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
        for tx, ty, tz in main_swept:
            min_x = min(min_x, tx)
            min_y = min(min_y, ty)
            min_z = min(min_z, tz)
            max_x = max(max_x, tx)
            max_y = max(max_y, ty)
            max_z = max(max_z, tz)

        mid_x = (max_x + min_x) / 2
        mid_y = (max_y + min_y) / 2
        mid_z = (max_z + min_z) / 2

        # return [mid_x, mid_y, scene_info[2]], main_swept
        return [mid_x, mid_y, mid_z], main_swept
        
    def check_collision_w_swept(self, swept_manager1, swept_manager2):
        # creat obsiticles
        if not self.obstacles_num:
            print("!!no obstacles added!!")
            return
        
        collision_obj_list = []
        for obj_idx in range(len(self.obj_mesh)):
            # read collision mesh
            temp_verts, temp_tris = self.obj_mesh[obj_idx]
            temp_m = fcl.BVHModel()
            temp_m.beginModel(len(temp_verts), len(temp_tris))
            temp_m.addSubModel(temp_verts, temp_tris)
            temp_m.endModel()
            temp_t = fcl.Transform()

            # check collision
            req = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request = req)
            swept_manager1.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
            is_collision1 = rdata.result.is_collision
            swept_manager2.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
            is_collision2 = rdata.result.is_collision

            if is_collision1 or is_collision2:
                collision_obj_list.append(obj_idx)

        return collision_obj_list
            

    def get_swept_volume_wo_bbox(self, path_list, test_name, grasp_idx, w_target=None, frame_rate= 60, scene_info= None, visualize= False):
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
            plotter2 = pv.Plotter()
            print("SAVING: test_data/swept_animations/" + test_name + "_grasp_w_obj_bbox" + str(grasp_idx) + ".gif")
            plotter.open_gif("test_data/swept_animations/" + test_name + "_grasp_w_obj_bbox" + str(grasp_idx) + ".gif")

            # add init mesh
            mesh_list = self.construct_robot_meshs(pos_list[0], plotter, w_target=w_target)
            self.add_all_obj_meshs(plotter)
            self.add_all_obj_meshs(plotter2)

            if scene_info is not None:
                self.construct_scene_meshs(scene_info, plotter)
            plotter.camera_position = 'yz'
            plotter.set_background('white')
            plotter2.set_background('white')
        else:
            plotter = None
            plotter2 = None
            mesh_list = None

        # add swept volume
        for angles in pos_list:
            bbox_list = self.update_robot_meshs_swept(angles, mesh_list, plotter2, w_target)
            if visualize:
                plotter.write_frame()
            swept_manager.registerObjects(bbox_list)
            swept_manager.setup()

        if visualize:
            plotter2.camera_position = 'yz'
            plotter2.show()
            plotter.close()

        return swept_manager

    def update_robot_meshs(self, angles, robot_mesh_list):
        transform_data = self.calculate_transform_from_angles(angles)
        translation = [x[0] for x in transform_data]
        rotation = [x[1] for x in transform_data]

        bbox_list = []
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

    def path_animation(self, path_list, test_name, grasp_idx, scene_info=None, frame_rate=50):
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

        # add all the objects
        self.add_all_obj_meshs(plotter)


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
            # plotter.add_mesh(pcd_mesh, color= "blue")
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
        for t in range(9):
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
    
    def get_MCT_config(self, obj_pos_list, obj_mesh, target_pos, target_mesh):
        config= []
        radius_list = get_radius(obj_mesh)
        for i in range(len(obj_pos_list)):
            color = self.obj_color[i]
            config.append(obj_pos_list[i] + [radius_list[i]] + [color])

        target_radius = get_radius([target_mesh])
        target_pos_MCT = [target_pos[0], target_pos[1], target_radius[0], 'red']
        return config, target_pos_MCT
    
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
    
    def calc_grasp_pos(self, grasp_mat, cam_rot, cam_tran, offset):
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
 
        # calc grasp_tran
        grasp_tran = grasp_mat[:3, 3]
        mat_rot = R.from_quat([-0.5, 0.5, -0.5, 0.5])
        grasp_tran = mat_rot.apply(grasp_tran)

        # grasp in global coord
        target_quat = (cam_rot_quat * grasp_rot).as_quat()
        target_pos = cam_tran + cam_rot_quat.apply(grasp_tran) + offset

        return target_pos, target_quat
    
    def grasp_verify(self, target_pos, target_quat):
        # target_pos, target_quat = self.calc_grasp_pos(grasp_mat, cam_rot, cam_tran, offset)

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

    def plan_all(self, dof_start, dof_result, flex_collision_models, time_limit):

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

        
        temp_res = optimizingPlanner.solve(time_limit)

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
        
def grasp_generation():
    test_name = 'sugar_box_grasp'
    # test_name = 'banana_grasp'
    # test_name = 'mustard_bottle_grasp'

    grasp_file_path = '../contact_graspnet/results/' + test_name + '.npz'
    grasp_datas = np.load(grasp_file_path, allow_pickle=True)
    grasp_score_idx = list(np.argsort(-grasp_datas["scores"].item()[1]))

    cam_file_path = 'test_data/test_scenes/7.29.14.14/test_npy/0.npy'
    # cam_file_path = 'test_data/test_scenes/7.29.14.7/test_npy/1.npy'
    # cam_file_path = 'test_data/test_scenes/7.29.13.31/test_npy/0.npy'

    cam_datas = np.load(cam_file_path, allow_pickle=True)
    cam_rot = cam_datas.item()["cam_rot"]
    cam_tran = cam_datas.item()["cam_tran"]

    scene_info = [0.56, 0.86000001, 0.1, 0.5]
    rac = robot_arm_configuration('../assets/urdf/ur5e/meshes/collision/', np.array([0.0, 0, 0]), scene_info) # point_cloud=point_cloud

    generated_grasp = []

    for grasp_idx in range(len(grasp_score_idx)):
        grasp_mat = grasp_datas["pred_grasps_cam"].item()[1][grasp_idx]
        offset = [-0.5, 0, 0]
        target_pos, target_quat = rac.calc_grasp_pos(grasp_mat, cam_rot, cam_tran, offset)
        generated_grasp.append({"target_pos":target_pos, "target_quat":target_quat})

        # init2grasp_angels = rac.grasp_verify(grasp_mat, cam_rot, cam_tran, offset=[-0.5,0,0])

        # if init2grasp_angels is not None:
        #     rac.check_collision_models(init2grasp_angels)
    np.save("../assets/urdf/ycb/004_sugar_box/grasp_dict.npy", generated_grasp)
        
def get_matching_mesh(target_pcd, file_idxs, visualize=False):
    asset_root = '../assets/'
    object_common_prefix = "urdf/ycb/"
    object_asset_files = []
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        for idx, line in enumerate(f):
            if idx not in file_idxs:
                continue
            i = line.find('/')
            print(idx)
            object_asset_files.append(asset_root + object_common_prefix + line[:i] + '/textured_vhacd.obj')

    # estimate obj position
    points = np.array(target_pcd.points)
    pos = np.median(points[:,:2], axis=0)
    init_transform = np.identity(4)
    init_transform[:2, 3] = -pos
    init_transform[2,3] = -0.1

    min_dist = sys.maxsize
    obj_mesh = None
    obj_trans = None
    obj_name = None
    for asset_file in object_asset_files:
        mesh = o3d.io.read_triangle_mesh(asset_file)
        mesh.compute_vertex_normals()
        source_pcd = mesh.sample_points_uniformly(number_of_points=20000)
        dist, trans = pcd_matching(target_pcd, source_pcd, init_transform, visualize)
        # print(asset_file)
        # print(dist)
        if dist < min_dist:
            min_dist = dist
            obj_mesh = mesh
            obj_trans = trans
            obj_name = asset_file

    print("matched object: ", obj_name)
    
    # calc inverse transform 
    inv_rot = obj_trans[:3,:3].T
    inv_trans = -inv_rot @ obj_trans[:3, 3]
    inv_rot = R.from_matrix(inv_rot.copy())

    # make object model
    verts_no_rotations = np.asarray(obj_mesh.vertices)
    face = np.asarray(obj_mesh.triangles)
    verts = inv_rot.apply(verts_no_rotations) + inv_trans

    return [verts, face], inv_trans, min_dist, obj_name

def pcd_matching(target_pcd, source_pcd, init_transform, visualize):
    # threshold = 0.8
    # threshold = 0.6
    # threshold = 1.3
    threshold = 0.1

    # init_transform = np.identity(4)
    if visualize:
        draw_registration_result(target_pcd, source_pcd, init_transform)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        target_pcd, source_pcd, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=9999999))

    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     target_pcd, source_pcd, threshold, init_transform,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=9999999))

    target_pcd_trans = copy.deepcopy(target_pcd)
    target_pcd_trans = target_pcd_trans.transform(reg_p2p.transformation)

    # if visualize:
    #     draw_registration_result(target_pcd_trans, source_pcd, np.identity(4))

    # calc inverse transform 
    inv_rot = reg_p2p.transformation[:3,:3].T
    inv_trans = -inv_rot @ reg_p2p.transformation[:3, 3]
    inv_rot = R.from_matrix(inv_rot.copy())

    test_trans = np.zeros((4,4))
    # pdb.set_trace()
    test_trans[:3,:3] = inv_rot.as_matrix()
    test_trans[:3, 3] = inv_trans
    test_trans[3, 3] = 1.0

    source_pcd_trans = copy.deepcopy(source_pcd)
    source_pcd_trans = source_pcd_trans.transform(test_trans)

    if visualize:
        draw_registration_result(target_pcd, source_pcd_trans, np.identity(4))

    dist = target_pcd_trans.compute_point_cloud_distance(source_pcd)
    dist_sum = np.asarray(dist).sum()
    return dist_sum, reg_p2p.transformation

def draw_registration_result(target, source, transformation):
    target_temp = copy.deepcopy(target)
    source_temp = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0[3], quaternion0[0], quaternion0[1], quaternion0[2]
    w1, x1, y1, z1 = quaternion1[3], quaternion1[0], quaternion1[1], quaternion1[2]
    return [x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0]

def write_to_pointcloud(color_image, depth_image, seg_image, cam_rotation, cam_translation, visualization=False):
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
          o3d.camera.PinholeCameraIntrinsic(param)
          )

    pcd_data = np.array(pcd.points, dtype = np.float32)
    
    pcd_data[:, [0,1,2]] = pcd_data[:, [2, 0, 1]]
    pcd_data[:, 1] *= -1
    pcd_data[:, 2] *= -1
    pcd_data = rot.apply(pcd_data)
    pcd_data += offset

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data)

    if visualization:
        o3d.visualization.draw_geometries([pcd])

    return pcd_data, pcd

def create_static_collision_model(scene_info, pcd_mesh, obj_mesh):
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

    return_list = [table_obj, left_cover_obj, right_cover_obj, upper_cover_obj]

    # target object from pcd
    if pcd_mesh is not None:
        verts = np.asarray(pcd_mesh.vertices)
        tris = np.asarray(pcd_mesh.triangles)

        pcd_m = fcl.BVHModel()
        pcd_m.beginModel(len(verts), len(tris))
        pcd_m.addSubModel(verts, tris)
        pcd_m.endModel()
        pcd_t = fcl.Transform()
        pcd_obj = fcl.CollisionObject(pcd_m, pcd_t)
        return_list.append(pcd_obj)

    if obj_mesh is not None:
        # new obj
        verts, tris = obj_mesh
        obj_m = fcl.BVHModel()
        obj_m.beginModel(len(verts), len(tris))
        obj_m.addSubModel(verts, tris)
        obj_m.endModel()
        obj_t = fcl.Transform()
        source_obj = fcl.CollisionObject(obj_m, obj_t)
        return_list.append(source_obj)

    return return_list

def get_path2grasp(rac, dof_result, scene_info, pcd_mesh=None, target_mesh=None, time_limit=10):
    plane_normal = np.array([0,0,1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    if scene_info is not None:
        static_env_models = create_static_collision_model(scene_info, pcd_mesh, target_mesh)
    else:
        static_env_models = []

    pp = path_planner(rac, plane_obj, static_env_models)

    ur5e_start_dof = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    path = pp.plan_all(ur5e_start_dof, dof_result, static_env_models, time_limit)

    return path

def get_path2start(rac, dof_result, mod_grip, scene_info, time_limit=10):
    # add modified grasp to fcl
    vertices, faces = mod_grip
    mod_fcl_gripper = fcl.BVHModel()
    mod_fcl_gripper.beginModel(len(vertices), len(faces))
    mod_fcl_gripper.addSubModel(vertices, faces)
    mod_fcl_gripper.endModel()
    temp_fcl = rac.fcl_models_[8]
    rac.fcl_models_[8] = mod_fcl_gripper

    plane_normal = np.array([0,0,1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    if scene_info is not None:
        static_env_models = create_static_collision_model(scene_info, None, None)
    else:
        static_env_models = []

    pp = path_planner(rac, plane_obj, static_env_models)

    ur5e_start_dof = [0, -math.pi/2, 0, -math.pi/2, 0, 0]
    path = pp.plan_all(dof_result, ur5e_start_dof, static_env_models, time_limit)

    rac.fcl_models_[8] = temp_fcl

    return path

def get_radius(mesh_list):
    radius_list = []
    for mesh in mesh_list:
        verts, _ = mesh
        min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
        max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
        for tx, ty, tz in verts:
            min_x = min(min_x, tx)
            min_y = min(min_y, ty)
            min_z = min(min_z, tz)
            max_x = max(max_x, tx)
            max_y = max(max_y, ty)
            max_z = max(max_z, tz)
        
        radius = max(max_x - min_x, max_y - min_y)
        radius_list.append(radius/2)

    return radius_list

def get_rearrange_result(ML_MCTS_ins):
    goal_config = ML_MCTS_ins.track_level_steps_[-1][-1].goal_config_
    scale = ML_MCTS_ins.scale
    new_pos = []
    for i in range(len(goal_config)):
        new_pos.append([goal_config[i][1] * scale, -goal_config[i][0] * scale])

    new_obj_mesh = ML_MCTS_ins.track_level_steps_[-1][-1].obj_mesh

    return new_pos, new_obj_mesh

# -------------------------------------------------------------------------------------------------------------------------------------


def clustering(unknown_area, visualize=False):
    point_list = copy.deepcopy(unknown_area)
    cluster_list = []
    cluster_list_idx = []
    while True:
        idx = np.random.randint(len(point_list))
        point = point_list[idx]
        point_list = np.delete(point_list, idx, axis=0)
        cluster = [point.tolist()]

        og_idx = np.argwhere((unknown_area == np.array(point)).all(1))
        cluster_idx = [og_idx.tolist()[0][0]]

        for x, y in cluster:
            check1 = [x+1,y]
            if not check1 in cluster:
                idx1 = np.argwhere((point_list == np.array(check1)).all(1))
                if idx1.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check1)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check1)
                    point_list = np.delete(point_list, idx1, axis=0)

            check2 = [x-1,y]
            if not check2 in cluster:
                idx2 = np.argwhere((point_list == np.array(check2)).all(1))
                if idx2.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check2)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check2)
                    point_list = np.delete(point_list, idx2, axis=0)

            check3 = [x,y+1]
            if not check3 in cluster:
                idx3 = np.argwhere((point_list == np.array(check3)).all(1))
                if idx3.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check3)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check3)
                    point_list = np.delete(point_list, idx3, axis=0)

            check4 = [x,y-1]
            if not check4 in cluster:
                idx4 = np.argwhere((point_list == np.array(check4)).all(1))
                if idx4.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check4)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check4)
                    point_list = np.delete(point_list, idx4, axis=0)

            check5 = [x+1,y+1]
            if not check5 in cluster:
                idx5 = np.argwhere((point_list == np.array(check5)).all(1))
                if idx5.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check5)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check5)
                    point_list = np.delete(point_list, idx5, axis=0)

            check6 = [x+1,y-1]
            if not check6 in cluster:
                idx6 = np.argwhere((point_list == np.array(check6)).all(1))
                if idx6.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check6)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check6)
                    point_list = np.delete(point_list, idx6, axis=0)

            check7 = [x-1,y+1]
            if not check7 in cluster:
                idx7 = np.argwhere((point_list == np.array(check7)).all(1))
                if idx7.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check7)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check7)
                    point_list = np.delete(point_list, idx7, axis=0)

            check8 = [x-1,y-1]
            if not check8 in cluster:
                idx8 = np.argwhere((point_list == np.array(check8)).all(1))
                if idx8.size != 0:
                    og_idx = np.argwhere((unknown_area == np.array(check8)).all(1))
                    cluster_idx += og_idx.tolist()[0]
                    cluster.append(check8)
                    point_list = np.delete(point_list, idx8, axis=0)

        cluster_list.append(cluster)
        cluster_list_idx.append(cluster_idx)
        if point_list.size == 0:
            break

    cluster_list = sorted(cluster_list, key=len, reverse=True)
    cluster_list_idx = sorted(cluster_list_idx, key=len, reverse=True)

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        for i in range(len(cluster_list)):
            # pdb.set_trace()
            plt.scatter(np.array(cluster_list[i])[:,0], np.array(cluster_list[i])[:,1])

            # new_cluster = unknown_area[cluster_list_idx[i]]
            # plt.scatter(np.array(new_cluster)[:,0], np.array(new_cluster)[:,1], c='red')
            # plt.scatter(np.array(cluster_list[i])[:,0], np.array(cluster_list[i])[:,1], c='black')
        
        plt.show()

    return cluster_list, cluster_list_idx
        
    # for point in np.random.choice(unknown_area):

# def check_area_shape(cluster, min_val):
#     for degree in range(0, 180, 10):
#         theta = np.radians(degree)
#         cos, sin = np.cos(theta), np.sin(theta)
#         rot = np.array(((cos,-sin), (sin, cos)))

#         new_cluster = np.zeros((len(cluster),2))
#         for i, point in enumerate(cluster):
#             new_cluster[i] = np.dot(rot, point)

#         # get bbox of swept
#         min_x, min_y = sys.maxsize, sys.maxsize
#         max_x, max_y = -sys.maxsize, -sys.maxsize
#         for tx, ty in new_cluster:
#             min_x = min(min_x, tx)
#             min_y = min(min_y, ty)
#             max_x = max(max_x, tx)
#             max_y = max(max_y, ty)
        
#         if (max_x - min_x) < min_val or (max_y - min_y) < min_val:
#             return False
#     return True

def get_vaild_area(center_list, radius, visualize=False):
    # get possible outter points
    offset_list = [[0,0]]
    p_list = [[0,i] for i in range(1, radius+1)]
    n_list = [[0,-i] for i in range(1, radius+1)]
    for degree in range(0, 180, 5):
        theta = np.radians(degree)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array(((cos,-sin), (sin, cos)))
        for i in range(int(radius)):
            point1 = np.rint(np.dot(rot, p_list[i])).tolist()
            point2 = np.rint(np.dot(rot, n_list[i])).tolist()
            if point1 not in offset_list:
                offset_list.append(point1)
            if point2 not in offset_list:
                offset_list.append(point2)

    # plt.figure(figsize=(20,20))
    # plt.axis([-10,10,-10,10])
    # plt.scatter(np.array(offset_list)[:,0], np.array(offset_list)[:,1])
    # plt.show()

    valid_area = []
    for center in center_list:
        for offset in offset_list:
            check_point = [center[0] + offset[0], center[1] + offset[1]]
            if check_point not in valid_area:
                valid_area += check_point

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        plt.scatter(np.array(center_list)[:,0], np.array(center_list)[:,1], color='red')
        plt.scatter(np.array(valid_area)[:, 0], np.array(valid_area)[:, 1], color='green')
        plt.show()

    return valid_area

    
def check_obj_fit(cluster, radius, visualize=False):
    # get possible outter points
    offset_list = []
    p_list = [[0,i] for i in range(1, radius+1)]
    n_list = [[0,-i] for i in range(1, radius+1)]
    for degree in range(0, 180, 1):
        theta = np.radians(degree)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array(((cos,-sin), (sin, cos)))
        for i in range(radius):
            point1 = np.rint(np.dot(rot, p_list[i])).tolist()
            point2 = np.rint(np.dot(rot, n_list[i])).tolist()
            if point1 not in offset_list:
                offset_list.append(point1)
            if point2 not in offset_list:
                offset_list.append(point2)

    valid_center = []
    valid_area = []
    for center in cluster:
        is_false = False
        point = []
        for offset in offset_list:
            check_point = [center[0] + offset[0], center[1] + offset[1]]

            if check_point not in cluster:
                is_false = True
                break
            point.append(check_point)
        if is_false:
            continue

        valid_center.append(center)
        valid_area.append(point)

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        plt.scatter(np.array(cluster)[:,0], np.array(cluster)[:,1], color='black')
        if valid_center:
            # plt.scatter(np.array(valid_area)[:, 0], np.array(valid_area)[:, 1], color='green')
            for vaild_points in valid_area:
                plt.scatter(np.array(vaild_points)[:, 0], np.array(vaild_points)[:, 1], color='green')
            plt.scatter(np.array(valid_center)[:, 0], np.array(valid_center)[:, 1], color='red')
        plt.show()

    return valid_area, valid_center

def cal_cam_angle_for_area(valid_points, curr_config, scene_info, visualize=False):
    left_point =  int(-scene_info[1]/2 * 100)
    right_point = int( scene_info[1]/2 * 100)

    line_list = []
    cluster_angles = {'foc':[], 'loc':[]}
    center = np.median(valid_points, axis=0)

    for point in np.arange(left_point, right_point + 1, int(scene_info[1] * 100) / 30):
        vec = ([point, 25] - center)

        is_collision = False
        for obj in curr_config:
            obj_pos = [-obj[1] * 100, obj[0] * 100]
            check_range = np.dot(obj_pos - np.array([point, 25]), -vec/np.linalg.norm(vec))
            if abs(check_range) > np.linalg.norm(vec):
                continue

            obj_vec = obj_pos - center
            radius = obj[2] * 100
            dist = abs((obj_vec[0] * vec[1] - obj_vec[1] * vec[0]) / np.linalg.norm(vec))
            if dist <= radius + 2:
                is_collision = True
                break

        if not is_collision:
            line_list.append(point)

    cluster_angles["foc"].append(np.array([center[1], -center[0]]) / 100)
    for point in line_list:
        cluster_angles["loc"].append(np.array([25, -point]) / 100)
        
    if visualize:
        plt.figure(figsize=(20,20))
        x = scene_info[1] / 2 * 100
        y = scene_info[0] * 100 + 30
        plt.axis([-x, x, 0, y])
        plt.scatter(np.array(valid_points)[:,0], np.array(valid_points)[:,1], color='orange')

        for obj in curr_config:
            obj_pos = [-obj[1] * 100, obj[0] * 100]
            radius = radius = obj[2] * 100
            temp_circle = mpatches.Circle((obj_pos), radius, color = obj[3])
            plt.gca().add_patch(temp_circle)

        foc = cluster_angles['foc'][0] * 100
        for point in cluster_angles['loc']:
            loc = point * 100
            plt.plot([-foc[1], -loc[1]], [foc[0], loc[0]], marker='o', color='black')

        plt.show()

    if len(cluster_angles["loc"]) == 0:
        return None

    return cluster_angles

def delete_obj_spots(curr_config, target_pos, unknown_area, visualize=False):
    pos_list = curr_config + [target_pos]
    offset_list = [[0,0]]
    p_list = [[0,i] for i in range(1, 6)]
    n_list = [[0,-i] for i in range(1, 6)]
    for degree in range(0, 180, 1):
        theta = np.radians(degree)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array(((cos,-sin), (sin, cos)))
        for i in range(5):
            point1 = np.rint(np.dot(rot, p_list[i])).tolist()
            point2 = np.rint(np.dot(rot, n_list[i])).tolist()
            if point1 not in offset_list:
                offset_list.append(point1)
            if point2 not in offset_list:
                offset_list.append(point2)

    new_area = copy.deepcopy(unknown_area)
    for obj in pos_list:
        obj_pos = np.array([-obj[1], obj[0]]) * 100
        radius = int(obj[2] * 100)
        ratio = radius / 5
        obj_area = np.rint(np.array(offset_list) * ratio + obj_pos)
        # plt.figure(figsize=(20,20))
        # plt.axis([-43,43,0,86])
        # plt.scatter(obj_area[:,0], obj_area[:,1], color='blue')
        # plt.show()

        for point in obj_area:
            idx = np.argwhere((new_area == point).all(1))
            if idx.size != 0:
                    new_area = np.delete(new_area, idx, axis=0)

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        plt.scatter(new_area[:,0], new_area[:,1], color='black')
        plt.show()

    return new_area

def process_unknown_area(unknown_area, curr_config, target_pos, min_radius, valid_center_num = 5, visualize=False):
    if len(unknown_area) <= valid_center_num:
        return [], []
    
    unknown_area = delete_obj_spots(curr_config, target_pos, unknown_area, visualize=False)
    cluster_list, _ = clustering(unknown_area)

    # get potential_centers
    potential_center_cluster = []
    valid_area_cluster = []
    for cluster in cluster_list:
        centered_valid_area, centers = check_obj_fit(cluster, int(min_radius), visualize=False)
        if len(centers) == 0:
            continue

        clustered_centers, clustered_center_idx = clustering(np.array(centers), visualize=False)
        
        # get valid_area
        valid_points = []
        for cluster_idx in range(len(clustered_centers)):
            if len(clustered_centers[cluster_idx]) < valid_center_num:
                continue

            center_idx = sorted(set(clustered_center_idx[cluster_idx]))
            centers = clustered_centers[cluster_idx]
            valid_points = np.array(centered_valid_area)[center_idx]
            valid_points = valid_points.reshape(-1,2)

            potential_center_cluster.append(centers)
            valid_area_cluster.append(valid_points.tolist())
            
            # plt.figure(figsize=(20,20))
            # plt.axis([-43,43,0,86])
            # plt.scatter(np.array(valid_area)[:,0], np.array(valid_area)[:,1], color='green')
            # plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], color='red')
            # plt.show()

    if visualize:
        potential_centers = np.array(sum(potential_center_cluster, []))
        valid_area = np.array(sum(valid_area_cluster, []))
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        plt.scatter(np.array(unknown_area)[:,0], np.array(unknown_area)[:,1], color='black')
        plt.scatter(np.array(valid_area)[:,0], np.array(valid_area)[:,1], color='green')
        plt.scatter(np.array(potential_centers)[:,0], np.array(potential_centers)[:,1], color='red')
        plt.show()

    return unknown_area, potential_center_cluster, valid_area_cluster

def transfor2global(curr_config):
    new_config = []
    for obj in curr_config:
        new_config.append([obj[1] / 100, -obj[0] / 100, obj[2] / 100, obj[3]])

    return new_config


# ------------------------------------------------------------------------------------------------------------------------------------------

def grasp_path_check(file_path, test_data_root, grasp_root, test_name, scene_info= None, grasp_check= False, obstacles_num=None):
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

    # get point cloud and mesh
    point_cloud, pcd = write_to_pointcloud(color_img, depth_img, seg_img, cam_rot, cam_tran)

    # pcd to mesh
    pcd.estimate_normals()
    radii = [0.002]
    pcd_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    # o3d.visualization.draw_geometries([pcd_mesh])

    # find matching object mesh file
    downpcd = pcd.voxel_down_sample(voxel_size=0.005) # downsampe pcd
    target_mesh, target_pos, _, _ = get_matching_mesh(downpcd, [3, 5] ,visualize=False)
    
    for vert in target_mesh[0]:
        vert[0] += 0.2
        vert[1] += 0.1

    target_pos[0] += 0.2
    target_pos[1] += 0.1
    
    grasp_datas = np.load(grasp_file_path, allow_pickle=True)
    grasp_score_idx = list(np.argsort(-grasp_datas["scores"].item()[1]))
    rac = robot_arm_configuration(file_path, np.array([0.0, 0, 0]), scene_info, target_mesh=target_mesh, obstacles_num=obstacles_num, target_pos=target_pos) # point_cloud=point_cloud

    rac.check_collision_models([0,-math.pi/2,0,-math.pi/2,0,0], scene_info=scene_info)
    return
    num_saved = 0
    total_time = 0
    for i in grasp_score_idx:
        print("idx", i)
        # configure env

        # reading grasp results
        grasp_mat = grasp_datas["pred_grasps_cam"].item()[1][i]
        target_pos, target_quat = rac.calc_grasp_pos(grasp_mat, cam_rot, cam_tran,offset=[0, 0, 0.0])
        init2grasp_angels = rac.grasp_verify(target_pos, target_quat)
        target_pos, target_quat = rac.calc_grasp_pos(grasp_mat, cam_rot, cam_tran, offset=[0, 0, 0.01])
        grasp2init_angels = rac.grasp_verify(target_pos, target_quat)

        if init2grasp_angels is None or grasp2init_angels is None:
            print("skip imposible grasp")
            continue # skip imposible grasp

        if grasp_check:
            rac.check_collision_models(init2grasp_angels, scene_info=scene_info)
            continue

        # get object in hand mesh
        mod_bbox = rac.modify_grasp_bbox(init2grasp_angels, target_mesh, visualize=False)
        mod_mesh = rac.modify_grasp_mesh(init2grasp_angels, target_mesh, visualize=False)

        init2grasp_path = None
        grasp2init_path = None
        swept_volume1 = None
        swept_volume2 = None
        swept_verts1 = None
        swept_verts2 = None
        collision_length = sys.maxsize
        start_time = time.time()
        for _ in range(2):
            # path planning & animation
            init2grasp_path_temp = get_path2grasp(rac, init2grasp_angels, scene_info, target_mesh=target_mesh) # pcd_mesh=pcd_mesh
            grasp2init_path_temp = get_path2start(rac, grasp2init_angels, mod_mesh, scene_info)

            if init2grasp_path_temp is None or grasp2init_path_temp is None:
                print("No path generated")
                continue

            # rac.path_animation(init2grasp_path, test_name, grasp_idx=i, scene_info=scene_info, frame_rate=30)
            # calculate swept volume with bounding box
            swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(init2grasp_path_temp, test_name, i, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
            swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(grasp2init_path_temp, test_name, i, w_target=mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)

            # check collision
            collision_obj_list = rac.check_collision_w_swept(swept_volume1_temp, swept_volume2_temp)
            num_collision_temp = len(collision_obj_list)
            if collision_length > num_collision_temp:
                collision_length = num_collision_temp
                init2grasp_path = init2grasp_path_temp
                grasp2init_path = grasp2init_path_temp
                swept_volume1 = swept_volume1_temp
                swept_volume2 = swept_volume2_temp
                swept_verts1 = swept_verts1_temp
                swept_verts2 = swept_verts2_temp

        if init2grasp_path is None or grasp2init_path is None:
            print("No path generated")
            continue


        # calculate swept volume with mesh
        # swept_volume1 = rac.get_swept_volume_wo_bbox(init2grasp_path, test_name, i, w_target=mod_mesh, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
        # swept_volume2 = rac.get_swept_volume_wo_bbox(grasp2init_path, test_name, i, w_target=mod_mesh, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)

        # check collision
        # collision_obj_list = rac.check_collision_w_swept(swept_volume1, swept_volume2)

        # if grasp_check:
        #     rac.check_collision_models(init2grasp_angels, scene_info=scene_info)
            # continue

        # # get rearrange planning
        # curr_config, target_pos_MCT = rac.get_MCT_config(rac.obj_pos_list, rac.obj_mesh, target_pos, target_mesh)
        # print(curr_config)
        # ML_MCTS_ins = mct.multi_level_MCTS_algo(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info, swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh, target_pos=target_pos_MCT)
        # ML_MCTS_ins.animate_whole_sequence()

        # # update values
        # rac.obj_pos_list, rac.obj_mesh = get_rearrange_result(ML_MCTS_ins)

        # # check collision
        # collision_obj_list = rac.check_collision_w_swept(swept_volume1, swept_volume2)        

        end_time = time.time()
        temp_time = end_time - start_time

        save_info = {"idx" : i,
                     "init2grasp_path" : init2grasp_path,
                     "grasp2init_path" : grasp2init_path,
                     "obj_pos_list" : rac.obj_pos_list,
                     "obj_mesh" : rac.obj_mesh,
                     "scene_info" : scene_info,
                     "w_target" : mod_bbox,
                     "test_name" : test_name,
                     "target_mesh" : target_mesh,
                     "obstacles_num" : rac.obstacles_num,
                     "target_pos" : target_pos}
        
        comp = np.array([save_info, temp_time])
        name = "test_data/MCTS_input/" + test_name + "_grasp_" + str(i) + "_obj_num_" + str(rac.obstacles_num) + "_vid"
        np.save(name, comp)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!saved:", name, temp_time, "sec")
        num_saved += 1
        total_time += temp_time
        # return

        if num_saved == 5:
            break

    return
    # return mesh_time_list, bbox_time_list

def check_MCTS(MCTS_root, MCTS_name, file_path=None):
    MCTS_path = MCTS_root + MCTS_name
    data = np.load(MCTS_path, allow_pickle=True)
    init2grasp_path = data[0]["init2grasp_path"]
    grasp2init_path = data[0]["grasp2init_path"]
    obj_pos_list = data[0]["obj_pos_list"]
    scene_info = data[0]["scene_info"]
    test_name = data[0]["test_name"]
    obj_mesh = data[0]["obj_mesh"]
    w_target = data[0]["w_target"]
    idx = data[0]["idx"]
    obj_pos_list = data[0]["obj_pos_list"]
    target_mesh = data[0]["target_mesh"]
    obstacles_num = data[0]["obstacles_num"]
    target_pos = data[0]["target_pos"]

    unknown_area = data[0]["unknown_area"]
    valid_area = data[0]["valid_area"]
    potential_centers = data[0]["potential_centers"]

    # obstacles_num = 3
    # scene_info = [0.70, 1.1000001, 0.1, 0.5]

    # rac = robot_arm_configuration(file_path, np.array([0.0, 0, 0]), scene_info, target_mesh=target_mesh, obstacles_num=obstacles_num, target_pos=target_pos) # point_cloud=point_cloud
    rac = robot_arm_configuration('../assets/urdf/ur5e/meshes/collision/', np.array([0.0, 0, 0]), scene_info) # point_cloud=point_cloud
    rac.target_mesh = target_mesh
    rac.obstacles_num = obstacles_num
    rac.obj_mesh = obj_mesh
    rac.obj_pos_list = obj_pos_list


    # ['#00fffb', '#ff00dd', '#bf00ff', '#ffae00', '#59ff00', '#FFFF00']
    #    cyan      pink       purple     orange    green       yellow
    #     0         1           2          3         4            5

    # calculate swept volume with bounding box
    swept_volume1, swept_verts1 = rac.get_swept_volume(init2grasp_path, test_name, idx, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False, with_scene=True)
    swept_volume2, swept_verts2 = rac.get_swept_volume(grasp2init_path, test_name, idx, w_target=w_target, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False, with_scene=True)
    # swept_center, swept_verts = rac.get_swept_center(swept_verts1+swept_verts2, scene_info)


    # Replay -------------------------------------------------------------------------------------------------------------------
    # print("REPLAY START")
    # data = np.load("test_data/MCTS_result/"+MCTS_name, allow_pickle=True)
    # mcts = data[0]
    # mcts.swept_manager1 = swept_volume1
    # mcts.swept_manager2 = swept_volume2

    # for tree in mcts.track_level_steps_:
    #     for root in tree:
    #         mcts.insert_swept(root)
    #         mcts.insert_radius(root)


    # mcts.global_optimization()

    # pdb.set_trace()
    # mcts.animate_whole_sequence()

    # rac.obj_pos_list, rac.obj_mesh = get_rearrange_result(mcts)
    # collision_obj_list = rac.check_collision_w_swept(swept_volume1, swept_volume2)

    # assert not collision_obj_list, "Rearranging failed"
    # return
    # ---------------------------------------------------------------------------------------------------------------------------

    # check collision
    collision_obj_list = rac.check_collision_w_swept(swept_volume1, swept_volume2)
    print("objects in collision: ", collision_obj_list)

    # get rearrange planning
    curr_config, target_pos_MCT = rac.get_MCT_config(rac.obj_pos_list, rac.obj_mesh, target_pos, target_mesh)

    unknown_area = delete_obj_spots(curr_config, target_pos_MCT, unknown_area, visualize=False)
    unknown_area, potential_center_cluster, valid_area_cluster = process_unknown_area(unknown_area, curr_config, target_pos_MCT, min_radius= 3, valid_center_num=5, visualize=False)
    potential_centers = np.array(sum(potential_center_cluster, []))
    valid_area = np.array(sum(valid_area_cluster, []))

    # --------------------------------------------------------------------------------------------------------
    # ML_MCTS_ins = mct.multi_level_MCTS_algo(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info,
    #                                         swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh,
    #                                         target_pos=target_pos_MCT)
    # ML_MCTS_ins.init_MCTS()


    # obj_idx, collision_points = ML_MCTS_ins.unknown_tunnel_check()
    # check_cluster = None
    # for cluster in valid_area_cluster:
    #     exist = np.any(np.isin(cluster, collision_points).all(1))
    #     if exist:
    #         check_cluster = cluster
    #         break
    
    # cluster_angles = cal_cam_angle_for_area(check_cluster, curr_config + [target_pos_MCT], scene_info, visualize=True)
    # pdb.set_trace()
    
    # cluster_angles = cal_cam_angle_for_area(centers, curr_config + [target_pos_MCT], scene_info, visualize=True)
    # end_point = cluster_angles['loc'][0]
    # focus_point = cluster_angles['foc'][0]
    # pdb.set_trace()
    # camera_loc, camera_focus, dof_result = ur5.cam_loc_selection_for_clusters(sim, envs[-1], test_cam, scene, focus_point, end_point, scene_info)

    print(curr_config)


    ML_MCTS_ins = mct.multi_level_MCTS_algo(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info,
                                            swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh,
                                            target_pos=copy.deepcopy(target_pos_MCT), unknown_area=unknown_area, valid_area=valid_area,
                                            potential_centers=potential_centers)
    # ML_MCTS_ins_OG = mct_OG.multi_level_MCTS_algo_OG(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info,
    #                                         swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh,
    #                                         target_pos=copy.deepcopy(target_pos_MCT), unknown_area=unknown_area, valid_area=valid_area,
    #                                         potential_centers=potential_centers)
    # ML_MCTS_ins_OG2 = mct_OG2.multi_level_MCTS_algo_OG2(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info,
    #                                         swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh,
    #                                         target_pos=copy.deepcopy(target_pos_MCT), unknown_area=unknown_area, valid_area=valid_area,
    #                                         potential_centers=potential_centers)
    # ML_MCTS_ins_base1 = mct_base1.multi_level_MCTS_algo_base1(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info,
    #                                         swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh,
    #                                         target_pos=copy.deepcopy(target_pos_MCT), unknown_area=unknown_area, valid_area=valid_area,
    #                                         potential_centers=potential_centers)
    # ML_MCTS_ins_base2 = mct_base2.multi_level_MCTS_algo_base2(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info,
    #                                         swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh,
    #                                         target_pos=copy.deepcopy(target_pos_MCT), unknown_area=unknown_area, valid_area=valid_area,
    #                                         potential_centers=potential_centers)
    
    
    # swept_volume1, _ = rac.get_swept_volume(init2grasp_path, test_name, idx, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
    # ML_MCTS_ins_base1.init_MCTS()
    # ML_MCTS_ins_base1.MCTS_ins.MCTS_tree_.tunnel_and_normal_visualizer(unknown_show=True)

    # obj_idx, collision_points = ML_MCTS_ins_base1.unknown_tunnel_check()

    # ML_MCTS_ins.init_MCTS()
    # is_success, child_node_list = ML_MCTS_ins.run_mcts(2)

    # ML_MCTS_ins_OG.init_MCTS()
    # is_success, child_node_list = ML_MCTS_ins_OG.run_mcts(2)

    # ML_MCTS_ins_base1.init_MCTS()
    # is_success, child_node_list = ML_MCTS_ins_base1.run_mcts(2)
    # ML_MCTS_ins_base1.animate_whole_sequence()


    ML_MCTS_ins.init_MCTS()
    ML_MCTS_ins.MCTS_ins.MCTS_tree_.tunnel_and_normal_visualizer(unknown_show=True)
    is_plan_success, child_node_list = ML_MCTS_ins.run_mcts(30)

    if not is_plan_success:
        max_reward = -sys.maxsize
        min_num_collision = sys.maxsize
        max_node = None
        for child in child_node_list:
            if len(child.check_collision_w_swept()) < min_num_collision and len(child.check_collision_w_swept()) != 0:
                max_node = child
                max_reward = child.reward_
            elif len(child.check_collision_w_swept()) == min_num_collision:
                if child.reward_ > max_reward:
                    max_reward = child.reward_
                    max_node = child

        collision_check_obj = []
        swept_check_obj = []
        swept_obj = max_node.check_collision_w_swept()
        for obj_idx in swept_obj:
            tunnel = max_node.get_tunnel(max_node.robot_, max_node.curr_config_[obj_idx][:2])
            tunnel_collision_obj = max_node.collision_tunnel_object(tunnel)
            tunnel_collision_obj.remove(obj_idx)

            if tunnel_collision_obj:
                print(tunnel_collision_obj)
                collision_check_obj += tunnel_collision_obj
            else:
                swept_check_obj.append(obj_idx)

        check_obj = swept_check_obj + sorted(set(collision_check_obj))
        # max_node.tunnel_and_normal_visualizer()

        max_region_dict = {}
        for cluster_idx in range(len(valid_area_cluster)):
            if len(valid_area_cluster[cluster_idx]) < 5:
                continue
            temp_valid_area = copy.deepcopy(valid_area_cluster)
            temp_valid_area.pop(cluster_idx)
            temp_valid_area = np.array(sum(temp_valid_area, []))

            total_new_region = 0
            for obj_idx in check_obj:
                total_new_region += max_node.region_counting(obj_idx, temp_valid_area)
                max_region_dict[cluster_idx] = total_new_region

        region_list = [*dict(sorted(max_region_dict.items(), key=lambda item: item[1], reverse=True))]
        mcts_out_angle = None
        for idx in region_list:
            mcts_out_angle = cal_cam_angle_for_area(valid_area_cluster[idx], ML_MCTS_ins.curr_config_ + [target_pos_MCT], scene_info, visualize=False)
            if mcts_out_angle is not None:
                break

        if mcts_out_angle is None:
            print("------ RUN FAILED WITH UNOBSERVABLE AREA ------")

        mcts_selected_cluster = valid_area_cluster[idx]
        run_mcts = False


    # if is_plan_success:
        # num_collision_obj_ = len(ML_MCTS_ins.track_level_steps_[0][0].check_collision_w_swept())
        # ML_MCTS_ins.global_optimization()
        # res_plan = ML_MCTS_ins.save_planning_results()
        # write_result(new_folder, 'test_results/complete_sensing/MCTS*', view, len(curr_config), ML_MCTS_ins.time_consumption_, ML_MCTS_ins.total_steps_, ML_MCTS_ins.calculate_total_length_travelled(), ML_MCTS_ins.calculate_total_length_displacement(), num_collision_obj_, 0, 0, 0, 0, res_plan, extra_name="_w_num_obj")

    # else:
    #     write_result(new_folder, 'test_results/complete_sensing/MCTS*', view, len(curr_config), ML_MCTS_ins.time_consumption_, "Failed", "Failed", "Failed", num_collision_obj_, 0, 0, 0, 0, None, "_w_num_obj")


    return is_plan_success

def test_plz(root):
    for folder_name in range(38, 0, -1):
        print('----------------------' + str(folder_name) + '----------------------')
        test_name = "/test_results/complete_sensing/MCTS*/"
        data_folder = root + str(folder_name) + test_name

        file_list = []
        for file_name in os.listdir(data_folder):
            if file_name.endswith(".npy") and 'cam_dofs' not in file_name and 'result' not in file_name and 'temp' in file_name:
                file_list.append(file_name)

        file_list.sort()
        for file in file_list:
            result = check_MCTS(data_folder, file)
            if result:
                break

def scene_height_check(scene_info, target_idx):
    GT_TARGET_POS = [np.random.uniform(0.20 + scene_info[0]/2, scene_info[0]),
                         np.random.uniform(-scene_info[1]/2 + 0.1, scene_info[1]/2 - 0.2),
                         scene_info[3] + 0.08]
    # GT_TARGET_POS = [0.8,0.5,1.0]
    # init setup
    file_path = '../assets/urdf/ur5e/meshes/collision/'
    rac = robot_arm_configuration('../assets/urdf/ur5e/meshes/collision/', np.array([0.0, 0, 0]), scene_info) # point_cloud=point_cloud

    target_obj_pos = copy.deepcopy(GT_TARGET_POS)

    asset_root = "/home/j0k/Project/Imsa/assets/"
    object_common_prefix = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        for idx, line in enumerate(f):
            if idx != target_idx:
                continue
            i = line.find('/')
            print(idx)
            obj_name = asset_root + object_common_prefix + line[:i] + '/textured_vhacd.obj'

    target_obj_mesh = obj_reader(obj_name)
    target_obj_mesh.add_offset([target_obj_pos[0], target_obj_pos[1], 0.1])
    target_obj_mesh = [target_obj_mesh.get_vertices(), target_obj_mesh.get_faces()]

    # read grasp data
    grasp_file = "/".join(obj_name.split('/')[:-1]) + "/grasp_dict.npy"
    grasp_data = np.load(grasp_file, allow_pickle=True)
    # generate swept volume
    num_grasp = 0
    swept_size = sys.maxsize
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(np.arange(len(grasp_list)))
    rac.target_mesh = target_obj_mesh
    get_out = False
    while num_grasp == 0:
        skip_grasp = 0
        for grasp_idx in grasp_list[:20]:
            target_grasp_pos = grasp_data[grasp_idx]['target_pos']
            target_grasp_quat = grasp_data[grasp_idx]['target_quat']
            target_grasp_pos[:2] = target_grasp_pos[:2] + target_obj_pos[:2]
            init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
            grasp2init_angels_temp = rac.grasp_verify(target_grasp_pos + [0,0,0.01], target_grasp_quat)
            if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
                skip_grasp += 1
                assert skip_grasp < len(grasp_list), "imposible grasp"
                print("skip imposible grasp")
                continue
            # rac.check_collision_models(grasp2init_angels_temp, scene_info=scene_info)

            # skip = input("skip?")
            # if not skip:
            #     print("skiping")
            #     continue

            init2grasp_path_temp = get_path2grasp(rac, init2grasp_angels_temp, scene_info, target_mesh=target_obj_mesh, time_limit=30)
            if init2grasp_path_temp is None:
                print("No path generated\n")
                continue

            temp_mod_bbox = rac.modify_grasp_bbox(init2grasp_angels_temp, target_obj_mesh, visualize=False)
            grasp2init_path_temp = get_path2start(rac, grasp2init_angels_temp, temp_mod_bbox, scene_info, time_limit=30)
            if grasp2init_path_temp is None:
                print("No path generated\n")
                continue
            num_grasp += 1
            print("\n----------grasp---------------\n")

        #     # compare swept volumes
        #     swept_center_temp, swept_verts_temp = rac.get_swept_center(swept_verts1_temp+swept_verts2_temp, scene_info, MAX_HEIGHT)
        #     temp_swept_size = get_swept_volume_size(swept_verts_temp)
        #     if temp_swept_size < swept_size:
        #         swept_size = temp_swept_size
        #         MAIN_swept_volume1 = swept_volume1_temp
        #         MAIN_swept_volume2 = swept_volume2_temp
        #         init2grasp_path = init2grasp_path_temp
        #         grasp2init_path = grasp2init_path_temp
        #         swept_center = swept_center_temp
        #         swept_verts = swept_verts_temp
        #         W_TARGET = temp_mod_bbox
            if num_grasp == 5:
                break
        if num_grasp > 0:
            break
        print(num_grasp)
        return num_grasp


if __name__ == '__main__':
    root = "test_data/collected_data/"
    # test_plz(root)
    # sys.exit(1)
    

    # grasp_generation()
    # file_path = '../assets/urdf/ur5e/meshes/collision/'
    # rac = robot_arm_configuration(file_path, np.array([0.0, 0, 0]))
    # rac.update_bounding_box()
    # angles = [0, -math.pi/2, math.pi/2, -math.pi/2, 0, 0]
    # object_matching(None)
    # sys.exit(1)

    # # add matched target object from file
    # asset_file = "../assets/urdf/ycb/001_chips_can/textured_vhacd.obj"
    # mesh = o3d.io.read_triangle_mesh(asset_file)
    # mesh.translate((0.0035, -0.0134, -0.2435))
    # mesh.compute_vertex_normals()
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    # size=0.2, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([mesh+mesh_frame])
    # sys.exit(1)

    # file paths
    file_path = '../assets/urdf/ur5e/meshes/collision/'
    test_data_root = 'test_data/'
    grasp_root = '../contact_graspnet/results/'
    test_name = 'pcd1' # banana
    # test_name = 'pcd2' # mustard
    # test_name = 'pcd3' # master chef can
    # scene_info = [0.56, 0.86000001, 0.1, 0.5]
    
    scene_info = [0.85,  1.2, 0.1, 0.6]
    # scene_info = [0.65,  1.16, 0.1, 0.3]
    # scene_info = [0.70,  1.06, 0.1, 0.4]
    # scene_info = [0.66,  0.9, 0.1, 0.47]


    # scene_info = [0.65,  0.9, 0.1, 0.42]
    # scene_info = [0.76,  1.16, 0.1, 0.5]
    # for i in range(10):

    scene_info = [0.80,  1.2, 0.1, 0.6]
    scene_info = [0.65,  0.9, 0.1, 0.42]
    # num = scene_height_check(scene_info, 5)
    # pdb.set_trace()
    grasp_path_check(file_path, test_data_root, grasp_root, test_name, scene_info=scene_info, obstacles_num=7, grasp_check=False)
    sys.exit(1)

    # test_name = '0'
    # # get file name
    # color_img_file_path = test_data_root + 'test_image/' + test_name + '.png'
    # seg_img_file_path = test_data_root + 'test_seg_image/' + test_name + '.png'
    # depth_img_file_path = test_data_root + 'test_depth_image/' + test_name + '.png'
    # cam_file_path = test_data_root + 'test_npy/' + test_name + '.npy'
    # grasp_file_path = grasp_root + 'predictions_' + test_name + '.npz'

    data_root = "test_data/test_scenes/"
    # data_root = "test_data/test_multi/"
    # scene_name = "7.17.17.18/"
    # scene_name = "7.17.18.43/"
    # scene_name = "7.17.18.53/"
    # scene_name = "7.17.19.19/"
    # scene_name = "7.17.19.33/"

    # scene_name = "7.18.10.35/"
    # scene_name = "7.18.10.43/"
    # scene_name = "7.18.10.47/" # good case
    # scene_name = "7.18.10.53/"
    # scene_name = "7.18.10.56/"

    # After pipline
    scene_name = "7.31.16.0_unsolve/"
    mcts_name = "temp_scene2.npy"

    # scene_name = "7.31.16.57_unsolve/"
    # mcts_name = "temp_scene3.npy"

    # scene_name = "8.1.21.13/"
    # mcts_name = "temp_scene2_success.npy"

    # scene_name = "8.1.22.50_failed/"
    # mcts_name = "temp_scene2_failed.npy"

    # scene_name = "8.2.16.24_failed/"
    # mcts_name = "temp_scene2_success.npy"
    # mcts_name = "groud_truth_scene2_success.npy"

    scene_name = "8.1.22.50_failed/" # unsolve
    mcts_name = "temp_scene2_failed.npy"

    scene_name = "8.7.18.14/"
    mcts_name = "temp_scene2_success.npy"

    # --------------------------------------------------------

    scene_name = "8.6.18.25/"
    mcts_name = "temp_scene3_success.npy"

    scene_name = "8.6.18.15/"
    mcts_name = "temp_scene3_success.npy"

    # scene_name = "7.31.16.39_success/"
    # mcts_name = "groud_truth_scene4.npy"

    # scene_name = "8.7.18.14/"
    # mcts_name = "temp_scene2_success.npy"

    scene_name = "8.7.18.4/"
    mcts_name = "temp_scene2_success.npy"

    # scene_name = "8.7.20.15/"
    # mcts_name = "temp_scene2None.npy"

    # scene_name = "8.7.20.28/"
    # mcts_name = "temp_scene2_success.npy"

    scene_name = "8.7.20.39/"
    mcts_name = "temp_scene2None.npy"

    # scene_name = "8.7.22.6/"
    # mcts_name = "temp_scene1_success.npy"

    # scene_name = "8.7.22.34/"
    # mcts_name = "temp_scene2_success.npy"

# -----------------------------------------------------------

    # scene_name = "8.12.11.20/"
    # mcts_name = "mcts*_temp_scene2_success.npy"

    # data_root = "test_data/collected_data/"
    # scene_name = "5/test_results/complete_sensing/BASE2/"
    # mcts_name = "temp_scene3_failed.npy"

    # data_root = "test_data/test_active_sensing/"
    # scene_name = "8.26.16.57/test_results/complete_sensing/BASE1/"
    # mcts_name = "temp_scene3_success.npy"

    # data_root = "test_data/test_active_sensing/"
    # scene_name = "8.28.21.26/test_results/complete_sensing/MCTS*/"
    # mcts_name = "temp_scene1_failed.npy"

    # data_root = "test_data/test_active_sensing/"
    # scene_name = "8.29.0.0/test_results/complete_sensing/MCTS*/"
    # mcts_name = "temp_scene5_failed.npy"

    # data_root = "test_data/test_active_sensing/"
    # scene_name = "8.29.11.37/test_results/complete_sensing/BASE2/"
    # mcts_name = "temp_scene9_failed.npy"

    # data_root = "test_data/test_active_sensing/"
    # scene_name = "8.29.16.37/test_results/complete_sensing/MCTS*/"
    # mcts_name = "temp_scene2_failed.npy"

    # data_root = "test_data/test_active_sensing/"
    # scene_name = "8/test_results/complete_sensing/MCTS*/"
    # mcts_name = "temp_scene4_success.npy"

    data_root = "test_data/temp/eval2/"
    scene_name = "35/test_results/complete_sensing/MCTS*/"
    mcts_name = "temp_scene3_failed.npy"

    # data_root = "test_data/collected_data/"
    # scene_name = "8/test_results/complete_sensing/BASE2/"
    # mcts_name = "temp_scene2_success.npy"

    mcts_root = data_root + scene_name
    # mcts_name = "groud_truth_scene.npy"
    check_MCTS(mcts_root, mcts_name, file_path) 

    scene_name = "8.7.18.40_!!/"
    mcts_name = "temp_scene1None.npy" # unsolve


    scene_info = [0.56, 0.86000001, 0.1, 0.5]
    # scene_info = [0.80, 1.0000001, 0.1, 0.5]
    mcts_root = "test_data/MCTS_input/"
    # mcts_name = "pcd2_grasp_93_obj_num_7_vid.npy"
    # check_MCTS(mcts_root, mcts_name, file_path)

    # mcts_name = "pcd2_grasp_4_obj_num6.npy"
    # mcts_name = "ur5_sim_3obj.npy"
    # mcts_name = "ur5_sim_3obj2.npy"
    # mcts_name = "ur5_sim_3obj2_temp.npy"

    # mcts_name = "pcd2_grasp_40_obj_num_6_minS.npy" # easy
    # mcts_name = "pcd2_grasp_119_obj_num_6_minS.npy" # super easy
    # mcts_name = "pcd2_grasp_95_obj_num_6_minS.npy" # unsol
    # mcts_name = "pcd2_grasp_47_obj_num_6_minS.npy" # easy
    # mcts_name = "pcd2_grasp_144_obj_num_6_minS.npy" # hard
    # mcts_name = "pcd2_grasp_133_obj_num_6_minS.npy" # easy
    # mcts_name = "pcd2_grasp_51_obj_num_6_minS.npy" # easy
    # mcts_name = "pcd2_grasp_65_obj_num_6_minS.npy" # use swept 33sec
    # mcts_name = "pcd2_grasp_110_obj_num_6_minS.npy" # easy
    # mcts_name = "pcd2_grasp_115_obj_num_6_minS.npy" # no collision

    # mcts_name = "pcd2_grasp_50_obj_num_7_bigS.npy" # no collision
    # mcts_name = "pcd2_grasp_169_obj_num_7_bigS.npy" # no collision
    # mcts_name = "pcd2_grasp_54_obj_num_7_bigS.npy" # no collision
    # mcts_name = "pcd2_grasp_105_obj_num_7_bigS.npy"
    # mcts_name = "pcd2_grasp_115_obj_num_7_bigS.npy" # no collision

    # mcts_name = "pcd2_grasp_101_obj_num_9_bigS.npy" # no collision
    mcts_name = "pcd2_grasp_50_obj_num_9_bigS.npy" # not solv
    # mcts_name = "pcd2_grasp_169_obj_num_9_bigS.npy"
    # mcts_name = "pcd2_grasp_54_obj_num_9_bigS.npy"
    # mcts_name = "pcd2_grasp_115_obj_num_9_bigS.npy"
    
    # mcts_name = "pcd2_grasp_65_obj_num_10_bigS.npy"
    # mcts_name = "pcd2_grasp_101_obj_num_10_bigS.npy" #depend
    # mcts_name = "pcd2_grasp_169_obj_num_10_bigS.npy" # not solve
    # mcts_name = "pcd2_grasp_54_obj_num_10_bigS.npy" # hard 3min
    # mcts_name = "pcd2_grasp_115_obj_num_10_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_65_obj_num_10_bigSS.npy"
    # mcts_name = "pcd2_grasp_101_obj_num_10_bigSS.npy" # hard
    # mcts_name = "pcd2_grasp_50_obj_num_10_bigSS.npy" # depend
    # mcts_name = "pcd2_grasp_169_obj_num_10_bigSS.npy"
    # mcts_name = "pcd2_grasp_115_obj_num_10_bigSS.npy" # depend

    # mcts_name = "pcd2_grasp_101_obj_num_11_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_50_obj_num_11_bigS.npy"
    # mcts_name = "pcd2_grasp_169_obj_num_11_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_54_obj_num_11_bigS.npy" # 25s
    # mcts_name = "pcd2_grasp_115_obj_num_11_bigS.npy" # depend

    mcts_name = "pcd2_grasp_51_obj_num_12_bigS.npy"
    # mcts_name = "pcd2_grasp_6_obj_num_12_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_80_obj_num_12_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_45_obj_num_12_bigS.npy" # hard unsolve
    # mcts_name = "pcd2_grasp_160_obj_num_12_bigS.npy"
    # mcts_name = "pcd2_grasp_16_obj_num_12_bigS.npy"
    # mcts_name = "pcd2_grasp_65_obj_num_12_bigS.npy" # no collision
    # mcts_name = "pcd2_grasp_101_obj_num_12_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_169_obj_num_12_bigS.npy"
    # mcts_name = "pcd2_grasp_115_obj_num_12_bigS.npy"

    # mcts_name = "pcd2_grasp_6_obj_num_13_bigS.npy" # hard
    # mcts_name = "pcd2_grasp_80_obj_num_13_bigS.npy" # hard
    # mcts_name = "pcd2_grasp_45_obj_num_13_bigS.npy"
    # mcts_name = "pcd2_grasp_160_obj_num_13_bigS.npy" # hard
    # mcts_name = "pcd2_grasp_16_obj_num_13_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_65_obj_num_13_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_101_obj_num_13_bigS.npy"
    # mcts_name = "pcd2_grasp_169_obj_num_13_bigS.npy" # depend
    # mcts_name = "pcd2_grasp_54_obj_num_13_bigS.npy"
    # mcts_name = "pcd2_grasp_115_obj_num_13_bigS.npy" # hard

    # time_con, time_con_og = check_MCTS(mcts_root, mcts_name, file_path)


    # print(total_time)
    # print(total_steps)
    # print(total_length_travelled)
    # print(total_length_displacement)


    # data = np.load("test_data/MCTS_result/pcd2_grasp_4_obj_num6.npy", allow_pickle=True)
    # mcts = data[0]
    # mcts_root = "test_data/MCTS_input/"
    # mcts_name = "test1.npy" # perfect
    # # mcts_name = "pcd2_grasp_4_obj_num6.npy" # perfect

    # MCTS_path = mcts_root + mcts_name
    # data = np.load(MCTS_path, allow_pickle=True)
    # pdb.set_trace()











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
                
