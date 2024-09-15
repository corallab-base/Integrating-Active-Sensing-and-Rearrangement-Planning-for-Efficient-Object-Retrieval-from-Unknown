#
# File:          ur5e_refactor.py
# Brief:         main program for ur5e simulation
# Author:        Hanwen Ren -- ren221@purdue.edu
# Date:          2022-01-04
# Last Modified: 2022-02-28
#

from scipy.spatial.transform import Rotation as R
import math
import time
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from PIL import Image
import numpy as np
from trac_ik_python.trac_ik import IK
import sys
import os
import open3d as o3d
import fcl
import cv2
import copy
# from grasp_util.robot_arm_configuration import robot_arm_configuration
from grasp_util.robot_arm_configuration import path_planner
from grasp_util.robot_arm_configuration import ur5e_valid

import pdb
import robot_arm_configuration_copy as RC

import MCTS_algo_ICRA as mct
import MCTS_algo_ICRA_OG as mct_OG
import MCTS_algo_ICRA_OG2 as mct_OG2
import MCTS_algo_ICRA_base1 as mct_base1
import MCTS_algo_ICRA_base2 as mct_base2
from rearrangement_planning_util_ICRA import write_result


file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../util')
grasp_util_dir = os.path.join(file_dir, './grasp_util')
#learning_dir = os.path.join(file_dir, './object_selection_learning')
scorenet_dir = os.path.join(file_dir, '../learning')
#sys.path.append(learning_dir)
sys.path.append(util_dir)
sys.path.append(scorenet_dir)
sys.path.append(grasp_util_dir)
sys.path.append('/home/j0k/coral/ompl-1.5.2/py-bindings')

import ompl.base as ob
import ompl.util as ou
import ompl.geometric as og
from stl_reader import stl_reader
from obj_reader import obj_reader
from global_scene import global_scene
from grasp_util.global_scene import global_scene
from grasp_util.pc_extractor_grasp import pc_extractor_grasp

from test_module.camera_view import camera
from runner import feed_forward
#from object_selection_learning.runner import feed_forward

#define parameters
#*************************************************************************************************#
#global settings
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))

#env settings
#x_dim [0.8, 1.0]
#y_dim [1.2, 1.5]
#z_dim [0.05, 0.2]
#table_dims = gymapi.Vec3(np.random.rand()*0.2 + 0.8, np.random.rand()*0.2 + 1.0,
#                         np.random.rand()*0.05 + 0.05)
# table_dims = gymapi.Vec3(0.56, 0.56, 0.10) # S
# table_dims = gymapi.Vec3(0.56, 0.86, 0.10) # M
# table_dims = gymapi.Vec3(0.76, 1.16, 0.10) # L

table_dims = gymapi.Vec3(0.56, 0.84, 0.05) # L
max_drawer_height = 0.44
min_drawer_height = 0.44
NUM_OF_OBJECTS = 6
NUM_SWEPT_COLLISION = 2
TARGET_OBJ_INDEX = [3] # mustard
# TARGET_OBJ_INDEX = [5] # banana
GAP_TO_BOX = 0.3

piece_width = 0.03
max_scaling_factor = 0
fall_height = table_dims.z
ADD_COVER = True

# TARGET_OBJ_INDEX = [3, 5]
OBSTACLE_OBJ_INDEX = [0, 2]
MIN_RADIUS = 0.03471716871486391

# MIN_NUM_OBSTACLES = 5
# MAX_NUM_OBSTACLES = 8
# # NUM_OF_OBJECTS = 11
# # NUM_OF_OBJECTS = np.random.randint(MIN_NUM_OBSTACLES + 1, MAX_NUM_OBSTACLES + 1)


#*************************************************************************************************#

#helper functions
#*************************************************************************************************#
def write_to_cam_pose(cam_rotation, cam_translation, name):
    cam_info = np.concatenate((cam_rotation, cam_translation), axis=None)
    np.save(name, cam_info)

def write_to_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw//4
    new_image = np.zeros((x_dim, y_dim, 3), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            offset = j*4
            for k in range(3):
                new_image[i][j][k] = raw_image[i][offset+k]
    img = Image.fromarray(new_image, 'RGB')
    img.save(image_name)
    return new_image

def convert_rgb_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw//4
    new_image = np.zeros((x_dim, y_dim, 3), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            offset = j*4
            for k in range(3):
                new_image[i][j][k] = raw_image[i][offset+k]
    return new_image

def write_to_seg_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw
    new_image = np.zeros((x_dim, y_dim), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            new_image[i][j] = raw_image[i][j]
    img = Image.fromarray(new_image)
    img.save(image_name)

def convert_seg_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw
    new_image = np.zeros((x_dim, y_dim), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            new_image[i][j] = raw_image[i][j]
    return new_image

def write_to_depth_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    maxi = -sys.maxsize
    mini = sys.maxsize
    for i in range(x_dim_raw):
        for j in range(y_dim_raw):
            maxi = max(maxi, raw_image[i][j])
            mini = min(mini, raw_image[i][j])
    x_dim, y_dim = x_dim_raw, y_dim_raw
    new_image = np.zeros((x_dim, y_dim, 1))
    for i in range(x_dim):
        for j in range(y_dim):
            if raw_image[i][j] != mini:
                new_image[i][j][0] = - int(raw_image[i][j]*1000)
            else:
                new_image[i][j][0] = 65535

    cv2.imwrite(image_name, new_image.astype(np.uint16))
    return new_image[:,:,0]

def convert_depth_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    maxi = -sys.maxsize
    mini = sys.maxsize
    for i in range(x_dim_raw):
        for j in range(y_dim_raw):
            maxi = max(maxi, raw_image[i][j])
            mini = min(mini, raw_image[i][j])
    x_dim, y_dim = x_dim_raw, y_dim_raw
    new_image = np.zeros((x_dim, y_dim, 1), dtype = np.uint16)
    for i in range(x_dim):
        for j in range(y_dim):
            if raw_image[i][j] != mini:
                # new_image[i][j][0] = - int(raw_image[i][j]*1000)
                new_image[i][j][0] = -(raw_image[i][j]*1000).astype(int)
            else:
                new_image[i][j][0] = 65535
    return new_image

def write_for_contact_grasp(color_image, seg_image, depth_image, k, cam_rot, cam_tran, name):
    arr = np.array({'rgb':color_image, 'depth':depth_image, 'K':k, 'seg': seg_image, 'cam_rot': cam_rot, 'cam_tran': cam_tran})
    np.save(name, arr)        

def global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, -coord2 + offset2)

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)

def get_random_loc(x_min, x_max, y_min, y_max, z_min, z_max):
    x_can = np.random.random()*(x_max - x_min) + x_min
    y_can = np.random.random()*(y_max - y_min) + y_min
    z_can = np.random.random()*(z_max - z_min) + z_min
    return gymapi.Vec3(x_can, y_can, z_can)

def get_best_cam_pose(scene, camera_pose_list):
    best_score = - sys.maxsize
    best_index = None
    for t in range(len(camera_pose_list)):
        pose_candidate = camera_pose_list[t]
        candidate_score = feed_forward(scene.scene_, pose_candidate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_index = t
    print (f'best score is : {best_score}')
    return best_index

def get_best_cam_pose_point_check(camera_pose_list, points, dofs=None):
    best_score = - sys.maxsize
    best_index = None
    for t in range(len(camera_pose_list)):
        pose_candidate = camera_pose_list[t]
        cam = camera(pose_candidate[4:], pose_candidate[:4])
        cam.build()
        
        in_cam = 0
        for point in points:
            if len(point) == 3:
                target_location = point
            else:
                target_location = [point[1]/100, -point[0]/100, 0.06]

            flag, depth, location = cam.inside_frame(target_location)
            if flag:
                in_cam += 1

        print("score: ", in_cam)
        # if dofs is not None:
        #     rac.check_collision_models(dofs[t], scene_info=scene_info)
        #     pdb.set_trace()

        if best_score < in_cam:
            best_score = in_cam
            best_index = t
    
    return best_index

def cam_loc_selection_for_clusters(sim, env, test_cam, center, end_points, scene_info, points):
    camera_pose_list = []
    camera_setting_list = []
    cam_height = scene_info[3] * 0.90
    floor_height = scene_info[2] + 0.01

    dist = 0
    while len(camera_pose_list) < 10:
        for end_point in end_points:
            loc_vec = (end_point - center) / np.linalg.norm(end_point - center) / 100
            cam_loc = end_point + loc_vec * dist

            camera_loc = gymapi.Vec3(cam_loc[0], cam_loc[1], cam_height)
            camera_focus = gymapi.Vec3(center[0], center[1], floor_height)
            gym.set_camera_location(test_cam, env, 
                                    camera_loc, 
                                    camera_focus)
            target_pos = gym.get_camera_transform(sim, env, test_cam).p
            target_quat = gym.get_camera_transform(sim, env, test_cam).r
            camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                    target_pos.x, target_pos.y, target_pos.z])

            r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
            cam_offset_vector = np.array([0.11, 0, 0.08])
            rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
            converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                     target_pos.y - rot_cam_offset_vector[1],
                                                     target_pos.z - rot_cam_offset_vector[2], 
                                                     ur5e_pose.p.x, 
                                                     ur5e_pose.p.y,
                                                     ur5e_pose.p.z)
            converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

            seed_state = [0.0]*ik_solver2.number_of_joints
            dof_result = ik_solver2.get_ik(seed_state, 
                                           converted_coord[0],
                                           converted_coord[1],
                                           converted_coord[2],
                                           converted_quat.x, 
                                           converted_quat.y,
                                           converted_quat.z,
                                           converted_quat.w)

            print('dist:', dist, 'loc', cam_loc, cam_height, "dof result", True if dof_result else False, 'len', len(camera_pose_list))

            if dof_result:
                end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)

                if end_state_collision_free:
                    camera_pose_list.append(camera_pose)
                    camera_setting_list.append([camera_loc, camera_focus, dof_result])

        if cam_loc[0] < 0.05:
            cam_height -= 0.01
            dist = 0

        if cam_height <= 0.20:
            break

        dist += 1

    CAM_SETTING_LIST = [cam_set[2] for cam_set in camera_setting_list]
    print("cluster")
    best_cam_pose_index = get_best_cam_pose_point_check(camera_pose_list, points, CAM_SETTING_LIST)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]


def random_sample_swept_volume_selection(sim, env, test_cam, swept_center, swept_points):
    camera_pose_list = []
    camera_setting_list = []
    while len(camera_pose_list) < 50:
        camera_loc = get_random_loc(0 + 0.2, table_dims.x + 0.3,
                                    -table_dims.y*0.5 + 0.02, table_dims.y*0.5 - 0.02,
                                    table_dims.z, table_dims.z + drawer_height - 0.02)
        camera_focus = gymapi.Vec3(swept_center[0], swept_center[1], swept_center[2])
        gym.set_camera_location(test_cam, env, 
                                camera_loc, 
                                camera_focus)
        target_pos = gym.get_camera_transform(sim, env, test_cam).p
        target_quat = gym.get_camera_transform(sim, env, test_cam).r
        camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                target_pos.x, target_pos.y, target_pos.z])

        r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
        cam_offset_vector = np.array([0.11, 0, 0.08])
        rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
        converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                 target_pos.y - rot_cam_offset_vector[1],
                                                 target_pos.z - rot_cam_offset_vector[2], 
                                                 ur5e_pose.p.x, 
                                                 ur5e_pose.p.y,
                                                 ur5e_pose.p.z)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

        seed_state = [0.0]*ik_solver2.number_of_joints
        dof_result = ik_solver2.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat.x, 
                                       converted_quat.y,
                                       converted_quat.z,
                                       converted_quat.w)
        if dof_result:
            end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)

            if end_state_collision_free:
                camera_pose_list.append(camera_pose)
                camera_setting_list.append([camera_loc, camera_focus, dof_result])
    print("swept")
    best_cam_pose_index = get_best_cam_pose_point_check(camera_pose_list, swept_points)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]


def random_sample_guided_selection(sim, env, test_cam, scene):
    camera_pose_list = []
    camera_setting_list = []
    while len(camera_pose_list) < 50:
        camera_loc = get_random_loc(0 + 0.2, table_dims.x + 0.3,
                                    -table_dims.y*0.5 + 0.02, table_dims.y*0.5 - 0.02,
                                    table_dims.z, table_dims.z + drawer_height - 0.02)
        camera_focus = get_random_loc(0 + 0.3, table_dims.x + 0.3,
                                      -table_dims.y*0.5 + 0.02, table_dims.y*0.5 - 0.02,
                                      table_dims.z, camera_loc.z)
        gym.set_camera_location(test_cam, env, 
                                camera_loc, 
                                camera_focus)
        target_pos = gym.get_camera_transform(sim, env, test_cam).p
        target_quat = gym.get_camera_transform(sim, env, test_cam).r
        camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                target_pos.x, target_pos.y, target_pos.z])

        r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
        cam_offset_vector = np.array([0.11, 0, 0.08])
        rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
        converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                 target_pos.y - rot_cam_offset_vector[1],
                                                 target_pos.z - rot_cam_offset_vector[2], 
                                                 ur5e_pose.p.x, 
                                                 ur5e_pose.p.y,
                                                 ur5e_pose.p.z)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

        seed_state = [0.0]*ik_solver2.number_of_joints
        dof_result = ik_solver2.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat.x, 
                                       converted_quat.y,
                                       converted_quat.z,
                                       converted_quat.w)
        if dof_result:
            end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)

            if end_state_collision_free:
                camera_pose_list.append(camera_pose)
                camera_setting_list.append([camera_loc, camera_focus, dof_result])
         
    best_cam_pose_index = get_best_cam_pose(scene, camera_pose_list)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]

def swept_coverage_check(scene, swept_verts, rac, scene_info, max_height):
    covered = 0
    new_verts = []
    for i, verts in enumerate(swept_verts):
        idx = verts * 100
        idx[0] -= 30
        idx[1] += 60
        idx = np.rint(idx).astype(int)
        checked = scene.scene_[idx[0], idx[1], idx[2]]
        if checked < 0:
            swept_verts.pop(i)
        if checked > 0:
            covered += 1
        else:
            new_verts.append(verts)

    next_center, _ = rac.get_swept_center([new_verts], scene_info, max_height)
    return covered / len(swept_verts), next_center

def get_swept_volume_size(main_swept):
    min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
    max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
    for tx, ty, tz in main_swept:
        min_x = min(min_x, tx)
        min_y = min(min_y, ty)
        min_z = min(min_z, tz)
        max_x = max(max_x, tx)
        max_y = max(max_y, ty)
        max_z = max(max_z, tz)

    return max_y - min_y
    # return max_x - min_x, max_y - min_y, max_z - min_z


def get_unobserved_area(scene):
    floor = scene.scene_[:scene.x_limit_, scene.y_left_+1 :(scene.y_left_ + scene.y_limit_-1), scene.g_height_]
    unknown_area = np.argwhere(floor == 0)[:,:2]
    unknown_area[:, 0] += 29
    unknown_area[:, 1] -= int((floor.shape[1]) / 2)

    temp = copy.deepcopy(unknown_area)
    unknown_area[:, 0] = -temp[:, 1]
    unknown_area[:, 1] = temp[:, 0]
    return unknown_area

def get_max_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST):
    max_height = -sys.maxsize
    for i in range(len(OBJ_FILE_IDX_LIST) - 1):
        obj_name = object_asset_files[OBJ_FILE_IDX_LIST[i]]
        obj_name = asset_root + "/".join(obj_name.split('/')[:3]) + "/textured_vhacd.obj"
    
        mesh = o3d.io.read_triangle_mesh(obj_name)
        verts = np.asarray(mesh.vertices)
        min_z = sys.maxsize
        max_z = -sys.maxsize
        for tx, ty, tz in verts:
            min_z = min(min_z, tz)
            max_z = max(max_z, tz)
        height = int((max_z - min_z) * 100)

        if max_height < height:
            max_height = height
    
    return max_height + 1

def get_min_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST):
    min_height = sys.maxsize
    for i in range(len(OBJ_FILE_IDX_LIST) - 1):
        obj_name = object_asset_files[OBJ_FILE_IDX_LIST[i]]
        obj_name = asset_root + "/".join(obj_name.split('/')[:3]) + "/textured_vhacd.obj"
    
        mesh = o3d.io.read_triangle_mesh(obj_name)
        verts = np.asarray(mesh.vertices)
        min_z = sys.maxsize
        max_z = -sys.maxsize
        for tx, ty, tz in verts:
            min_z = min(min_z, tz)
            max_z = max(max_z, tz)
        height = int((max_z - min_z) * 100) # 15, 0.14119999739341438

        if min_height > height:
            min_height = height
    
    return min_height

def get_unobserved_area_w_height(scene, asset_root, object_asset_files, OBJ_FILE_IDX_LIST):
    min_height = get_min_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST)
    floor = scene.scene_[:scene.x_limit_, scene.y_left_+1 :(scene.y_left_ + scene.y_limit_-1), scene.g_height_: scene.g_height_ + min_height]

    unknown_layer = set(map(tuple, np.argwhere(floor[:,:,0] == 0)[:,:2]))
    for i in range(1, min_height):
        temp_unknown = set(map(tuple, np.argwhere(floor[:,:,i] == 0)[:,:2]))
        unknown_layer = unknown_layer.intersection(temp_unknown)
    unknown_area = np.array(list(unknown_layer))

    unknown_area[:, 0] += 29
    unknown_area[:, 1] -= int((floor.shape[1]) / 2)

    temp = copy.deepcopy(unknown_area)
    unknown_area[:, 0] = -temp[:, 1]
    unknown_area[:, 1] = temp[:, 0]

    return unknown_area

def convert_0_to_end(point):
    new_point = [0,0,point[2]]
    new_point[1] = -point[0]
    new_point[0] = point[1]
    new_point[0] -= 29
    new_point[1] += int(scene_info[1] / 2 * 100)
    new_point[0] /= 100
    new_point[1] /= 100
    new_point[2] /= 100

    return new_point


def scale_config(config):
    for pos in config:
        for i in range(3):
            pos[i] = pos[i] * 100
        temp = pos[0]
        pos[0] = -pos[1]
        pos[1] = temp

    return config

def save_scene(init2grasp_path, grasp2init_path, obj_pos_list, gt_obj_pos_list, NUM_OF_OBJECTS, scene_info,
               target_mesh, obj_mesh, target_pos, gt_target_pos, obstacles_num, w_target,
               unknown_area, valid_area, potential_centers, method, MCTS_result=None):
    gt_save_info = {"idx" : 0,
                    "init2grasp_path" : init2grasp_path,
                    "grasp2init_path" : grasp2init_path,
                    "obj_pos_list" : gt_obj_pos_list[:NUM_OF_OBJECTS-1],
                    "obj_mesh" : obj_mesh,
                    "scene_info" : scene_info,
                    "w_target" : w_target,
                    "test_name" : None,
                    "target_mesh" : target_mesh,
                    "obstacles_num" : obstacles_num,
                    "target_pos" : gt_target_pos,
                    "unknown_area" : unknown_area,
                    "valid_area" : valid_area,
                    "potential_centers" : potential_centers}
    
    save_info = {"idx" : 0,
                 "init2grasp_path" : init2grasp_path,
                 "grasp2init_path" : grasp2init_path,
                 "obj_pos_list" : obj_pos_list,
                 "obj_mesh" : obj_mesh,
                 "scene_info" : scene_info,
                 "w_target" : w_target,
                 "test_name" : None,
                 "target_mesh" : target_mesh,
                 "obstacles_num" : obstacles_num,
                 "target_pos" : target_pos,
                 "unknown_area" : unknown_area,
                 "valid_area" : valid_area,
                 "potential_centers" : potential_centers}
    
    comp = np.array([save_info])

    end = "_success" if MCTS_result else "_failed"
    if MCTS_result is None:
        end = "None"

    name = new_folder + method + "temp_scene" + str(sequence_count) + end
    np.save(name, comp)

    comp = np.array([gt_save_info])
    name = new_folder + method + "groud_truth_scene" + str(sequence_count) + end
    np.save(name, comp)

def update_MCTS_val(ML_MCTS_ins, curr_config, target_pos_MCT, obj_mesh, unknown_area, valid_area, potential_centers):
    # update MCTS values
    ML_MCTS_ins.curr_config_ = copy.deepcopy(curr_config)
    ML_MCTS_ins.goal_config_ = copy.deepcopy(curr_config)
    ML_MCTS_ins.target_pos = copy.deepcopy(target_pos_MCT)
    ML_MCTS_ins.obj_mesh = copy.deepcopy(obj_mesh)

    ML_MCTS_ins.unknown_area = copy.deepcopy(unknown_area)
    ML_MCTS_ins.valid_area = copy.deepcopy(valid_area)
    ML_MCTS_ins.potential_centers = copy.deepcopy(potential_centers)

def update_rac_val(rac, target_obj_mesh, obj_mesh_MCTS, obj_pos_MCTS):
    rac.target_mesh = copy.deepcopy(target_obj_mesh)
    rac.obj_mesh = copy.deepcopy(list(obj_mesh_MCTS.values()))
    rac.obj_pos_list = copy.deepcopy(list(obj_pos_MCTS.values()))
    rac.obstacles_num = len(rac.obj_pos_list)

#*************************************************************************************************#

if __name__ == '__main__':
    #initialize gym
    #*************************************************************************************************#
    gym = gymapi.acquire_gym()
    #*************************************************************************************************#

    # parse arguments
    #*************************************************************************************************#
    args = gymutil.parse_arguments(description="ur5e example", custom_parameters = [{'name':'--env_id', 'type':int, 'help':'env_id', 'default':0}])
    env_id = int(args.env_id)
    #*************************************************************************************************#

    #create a simulator
    #*************************************************************************************************#
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    #*************************************************************************************************#

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, 
                        args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()
    #*************************************************************************************************#

    #configure a ground plane
    #*************************************************************************************************#
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    #*************************************************************************************************#

    #all assets
    #*************************************************************************************************#
    asset_root = "/home/j0k/Project/Imsa/assets/"
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    ur5e_collision_parts = ["urdf/ur5e/meshes/collision/base.stl",
                            "urdf/ur5e/meshes/collision/shoulder.stl",
                            "urdf/ur5e/meshes/collision/upperarm.stl",
                            "urdf/ur5e/meshes/collision/forearm.stl",
                            "urdf/ur5e/meshes/collision/wrist1.stl",
                            "urdf/ur5e/meshes/collision/wrist2.stl",
                            "urdf/ur5e/meshes/collision/wrist3.stl"]

    object_asset_files = []
    object_collision_files = []
    object_offset = []
    object_centroid_m = []
    object_common_prefix = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        for line in f:
            object_asset_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        for line in f:
            object_collision_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt") as f:
        for line in f:
            div = line[:-1].split(" ")
            object_offset.append([float(x) for x in div])

    #setup all collision meshes
    #setup is done outside the env loop since all robots are the same
    ur5e_collision_models = []
    ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
                    R.from_euler('xy', [90, 180], degrees = True),
                    R.from_euler('xy', [180, 180], degrees = True),
                    R.from_euler('z',  [-180], degrees = True),
                    R.from_euler('x',  [-180], degrees = True),
                    R.from_euler('x',  [90], degrees = True),
                    R.from_euler('z',  [-90], degrees = True)]
    ur5e_translations = [[0, 0, 0], 
                        [0, 0, 0],
                        [0, -0.138, 0],
                        [0, -0.007, 0],
                        [0, 0.127, 0],
                        [0, 0, 0],
                        [0, 0, 0]]
    for i in range(len(ur5e_collision_parts)):
        parts_path = ur5e_collision_parts[i]
        collision_mesh = stl_reader(asset_root + parts_path)
        m = fcl.BVHModel()
        collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    object_collision_lib = []
    #*************************************************************************************************#


    #calculate Inverse Kinematics
    #*************************************************************************************************#
    urdf_str = ''
    with open("../assets/urdf/ur5e/ur5e_mimic_real_gripper_test.urdf") as f:
        urdf_str = f.read()

    #*************************************************************************************************#

    # create viewer using the default camera properties
    #*************************************************************************************************#
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')
    #*************************************************************************************************#

    #set up the environment grid
    #*************************************************************************************************#
    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)
    #*************************************************************************************************#

    #load asset
    #*************************************************************************************************#
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x,
                                    table_dims.y,
                                    table_dims.z,
                                    asset_options)

    #size of left/right cover will be decided by table size
    drawer_height = np.random.random()*(max_drawer_height - min_drawer_height) + min_drawer_height
    side_cover_dims = gymapi.Vec3(table_dims.x, piece_width, drawer_height)
    left_cover_asset = gym.create_box(sim, side_cover_dims.x,
                                        side_cover_dims.y,
                                        side_cover_dims.z,
                                        asset_options)
    right_cover_asset = gym.create_box(sim, side_cover_dims.x,
                                            side_cover_dims.y,
                                            side_cover_dims.z,
                                            asset_options)

    #upper cover
    upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
    upper_cover_asset = gym.create_box(sim, upper_cover_dims.x,
                                            upper_cover_dims.y,
                                            upper_cover_dims.z,
                                            asset_options)
    
    left_door_dims = gymapi.Vec3(0.16, 1.0, drawer_height)
    left_door_asset = gym.create_box(sim, left_door_dims.x,
                                            left_door_dims.y,
                                            left_door_dims.z,
                                            asset_options)
    
    right_door_dims = gymapi.Vec3(0.16, 1.0, drawer_height)
    right_door_asset = gym.create_box(sim, left_door_dims.x,
                                            left_door_dims.y,
                                            left_door_dims.z,
                                            asset_options)
    
    

    saved_env_name = './saved_as_result/env_' + str(env_id) + '_scene_info.npy'
    np.save(saved_env_name, np.array([table_dims.x, table_dims.y, table_dims.z, drawer_height]))

    asset_options.fix_base_link = False
    object_assets = []
    test_assets = []
    for ob in object_asset_files:
        object_assets.append(gym.load_asset(sim, asset_root, ob, asset_options))
    asset_options.fix_base_link = True
    test_assets.append(gym.load_asset(sim, asset_root, object_asset_files[0], asset_options))
    asset_options.fix_base_link = False

    #*************************************************************************************************#

    #initial pose
    #*************************************************************************************************#
    ur5e_pose = gymapi.Transform()
    # ur5e_pose.p = gymapi.Vec3(np.random.rand()*0.3 - 0.2, np.random.rand()*0.4 - 0.2, 0.0)
    ur5e_pose.p = gymapi.Vec3(-0.12, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x*0.5 + GAP_TO_BOX, 0.0, table_dims.z*0.5)

    left_cover_pose = gymapi.Transform()
    left_cover_pose.p = gymapi.Vec3(table_pose.p.x, table_dims.y*0.5 - 0.015, 
                                    table_dims.z + side_cover_dims.z/2.0)

    right_cover_pose = gymapi.Transform()
    right_cover_pose.p = gymapi.Vec3(table_pose.p.x, -table_dims.y*0.5 + 0.015, 
                                    table_dims.z + side_cover_dims.z/2.0)

    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + side_cover_dims.z + 0.015)

    left_door_pose = gymapi.Transform()
    left_door_pose.p = gymapi.Vec3(0.3, -(table_dims.y/2 + left_door_dims.y/2), table_dims.z + drawer_height/2)

    right_door_pose = gymapi.Transform()
    right_door_pose.p = gymapi.Vec3(0.3, table_dims.y/2 + right_door_dims.y/2, table_dims.z + drawer_height/2)

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    #set all environment collision models

    plane_normal = np.array([0.0, 0.0, 1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)

    col_left_cover = fcl.Box(side_cover_dims.x,
                            side_cover_dims.y,
                            side_cover_dims.z)
    trans_left_cover = fcl.Transform(np.array([table_pose.p.x, table_dims.y*0.5 - 0.015, 
                                            table_dims.z + side_cover_dims.z/2.0]))
    left_cover_obj = fcl.CollisionObject(col_left_cover, trans_left_cover)

    col_right_cover = fcl.Box(side_cover_dims.x,
                            side_cover_dims.y,
                            side_cover_dims.z)
    trans_right_cover = fcl.Transform(np.array([table_pose.p.x, -table_dims.y*0.5 + 0.015, 
                                                table_dims.z + side_cover_dims.z/2.0]))
    right_cover_obj = fcl.CollisionObject(col_right_cover, trans_right_cover)

    left_door_fcl_box = fcl.Box(left_door_dims.x,
                   left_door_dims.y,
                   left_door_dims.z)
    trans_left_door = fcl.Transform(np.array([0.3 - left_door_dims.x/2.0, -(table_dims.y/2 + left_door_dims.y/2), table_dims.x + drawer_height/2]))
    left_door_fcl = fcl.CollisionObject(left_door_fcl_box, trans_left_door)


    right_door_fcl_box = fcl.Box(right_door_dims.x,
                   right_door_dims.y,
                   right_door_dims.z)
    trans_right_door = fcl.Transform(np.array([0.3 - right_door_dims.x/2.0, -(table_dims.y/2 + right_door_dims.y/2), table_dims.x + drawer_height/2]))
    right_door_fcl = fcl.CollisionObject(right_door_fcl_box, trans_right_door)


    object_collision_models = [table_obj, left_cover_obj, right_cover_obj, left_door_fcl, right_door_fcl]

    if ADD_COVER:
        col_upper_cover = fcl.Box(upper_cover_dims.x,
                                upper_cover_dims.y,
                                upper_cover_dims.z)
        trans_upper_cover = fcl.Transform(np.array([table_pose.p.x, 0.0, 
                                        table_dims.z + side_cover_dims.z + 0.015]))
        upper_cover_obj = fcl.CollisionObject(col_upper_cover, trans_upper_cover)
        object_collision_models.append(upper_cover_obj)

    #*************************************************************************************************#

    #create environment
    #*************************************************************************************************#
    #create location candidates
    location_candidates = []
    start_i = 0.4
    while start_i <= table_dims.x - 0.1 + 0.3:
        start_j = - table_dims.y * 0.5 + 0.1
        temp_candidates = []
        while start_j <= table_dims.y*0.5 - 0.1:
            temp_candidates.append([start_i, start_j, table_dims.z + 0.1])
            start_j += 0.2
        location_candidates.append(temp_candidates)
        start_i += 0.3

    region_candidates = []
    start_i = 0.4
    while start_i <= table_dims.x - 0.1 + 0.3:
        start_j = - table_dims.y * 0.5 + 0.1
        temp_candidates = []
        while start_j <= table_dims.y*0.5 - 0.1:
            temp_candidates.append([start_i, start_j, table_dims.z + 0.1])
            start_j += 0.1
        region_candidates.append(temp_candidates)
        start_i += 0.1

    print(len(region_candidates), len(region_candidates[0]))

    envs = []
    ur5e_handles = []
    body_cam_handles = []
    camera_candidates = []
    # chosen_object = []
    chosen_scale = []
    object_normalize = []

    observed_objects = []
    gripper_location = None
    object_status_list = []
    object_reader_tracker = []
    for i in range(num_of_envs):
        envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))

        #get joint handler
        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
        likj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_knuckle_joint")
        lifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_finger_joint")
        lokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_outer_knuckle_joint")
        rikj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_knuckle_joint")
        rifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_finger_joint")
        rokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_outer_knuckle_joint")

        #attach body camera sensor
        cam_link = gym.find_actor_rigid_body_handle(envs[-1], ur5e_handles[-1], "wrist_3_link")

        #right in front of D435i model
        cam_offset_x = 0.11
        cam_offset_z = 0.08
        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        body_cam_transform = gymapi.Transform()
        body_cam_transform.p = gymapi.Vec3(cam_offset_x, 0, cam_offset_z)
        gym.attach_camera_to_body(body_cam_handles[-1], envs[-1], cam_link, body_cam_transform, 
                                gymapi.CameraFollowMode.FOLLOW_TRANSFORM)

        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
        gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)
        if ADD_COVER:
            gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)
            gym.create_actor(envs[-1], left_door_asset, left_door_pose, "left_door" + str(i), 0, 1)
            gym.create_actor(envs[-1], right_door_asset, right_door_pose, "right_door" + str(i), 0, 1)


        file_path = '../assets/urdf/ur5e/meshes/collision/'
        scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        rac = RC.robot_arm_configuration(file_path, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info)

        cam_move_plan = np.load('test_data/test_real_experiment/real_test2/MCTS*/test_results/cam_dofs_3.npy')


        cam_planed_move = []
        for i in range(1, len(cam_move_plan) + 1):
            if i == 1:
                start_pos = [0.7, -2, 2.5, -0.3, 0.7, 0]
            else:
                start_pos = cam_move_plan[i-1]

            if i == len(cam_move_plan):
                RC.get_path2start(rac, cam_move_plan[-1], target_mesh=None, time_limit=60, given_static_model=object_collision_models)
            
            # end_state_collision_free = rac.arm_collision_free(cam_move_plan[i], plane_obj, object_collision_models, [])
            # print(end_state_collision_free)
            # rac.check_collision_models(start_pos, scene_info=scene_info)
            # rac.check_collision_models(cam_move_plan[i], scene_info=scene_info)

            cam_move_temp = RC.get_patha2b(rac, start_pos, cam_move_plan[i], target_mesh=None, time_limit=60, given_static_model=object_collision_models)
            cam_planed_move.append(cam_move_temp)


        with open('/home/j0k/Project/Imsa/main_code/test_data/test_real_experiment/real_test2/MCTS*/test_results/cam_plan.txt', 'w') as f:
            for i in cam_planed_move:
                f.write(str(i) + '\n')

        pdb.set_trace()

        # Choose target & obstacle objects-------------------------------------------------------------------------------
        target_file_idx = np.random.choice(TARGET_OBJ_INDEX, 1)
        object_file_idx = np.random.choice(OBSTACLE_OBJ_INDEX, NUM_OF_OBJECTS - 1)
        OBJ_FILE_IDX_LIST = np.concatenate((object_file_idx, target_file_idx), axis=0)

        # chosen_object.append(OBJ_FILE_IDX_LIST)
        object_handles = []

        with open("object_name.txt", 'a') as f:
            for k in range(NUM_OF_OBJECTS):
                f.write(object_asset_files[OBJ_FILE_IDX_LIST[k]])

        object_scaling_factor = np.random.randint(0, max_scaling_factor+1, size = NUM_OF_OBJECTS)/10.0 + 1.0

        # set up objects--------------------------------------------------------------------------------------------------------------------
        # creating manager
        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []
        GT_OBJ_MESH_LIST = []
        # GT_TARGET_POS = [np.random.uniform(0.20 + table_dims.x/2, table_dims.x),
        #                  np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2),
        #                  table_dims.z + 0.08]
        GT_TARGET_POS = [table_dims.x/2 + 0.3, 0, table_dims.z + 0.08]

        # init setup
        ik_solver2 = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
        target_quat = gymapi.Quat(0.446, 0.560, -0.433, 0.549)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)
        file_path = '../assets/urdf/ur5e/meshes/collision/'
        scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        MAX_HEIGHT = get_max_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST)
        rac = RC.robot_arm_configuration(file_path, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info)
        target_obj_pos = copy.deepcopy(GT_TARGET_POS)
        with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
            for idx, line in enumerate(f):
                if idx != target_file_idx[0]:
                    continue
                i = line.find('/')
                print(idx)
                obj_name = asset_root + object_common_prefix + line[:i] + '/textured_vhacd.obj'

        target_obj_mesh = obj_reader(obj_name)
        target_obj_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[-1]][:2] + [0])
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
            for grasp_idx in grasp_list:
                target_grasp_pos = grasp_data[grasp_idx]['target_pos']
                target_grasp_quat = grasp_data[grasp_idx]['target_quat']
                target_grasp_pos[:2] = target_grasp_pos[:2] + target_obj_pos[:2]
                init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
                grasp2init_angels_temp = rac.grasp_verify(target_grasp_pos + [0,0,0.02], target_grasp_quat)
                if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
                    skip_grasp += 1
                    assert skip_grasp < 15, "imposible grasp"
                    print("skip imposible grasp")
                    continue
                # rac.check_collision_models(init2grasp_angels_temp, scene_info=scene_info)

                init2grasp_path_temp = RC.get_path2grasp(rac, init2grasp_angels_temp, scene_info, target_mesh=target_obj_mesh, time_limit=30, given_static_model=object_collision_models)
                if init2grasp_path_temp is None:
                    print("No path generated\n")
                    continue

                grasp_dof = rac.grasp_verify(target_grasp_pos, target_grasp_quat, new_offset=True)
                init2grasp_path_temp.append(grasp_dof)
                temp_mod_bbox = rac.modify_grasp_bbox(grasp_dof, target_obj_mesh, visualize=False)
                
                grasp2init_path_temp = RC.get_path2start(rac, grasp2init_angels_temp, temp_mod_bbox, scene_info, time_limit=20, given_static_model=object_collision_models)
                if init2grasp_path_temp is None or grasp2init_path_temp is None:
                    print("No path generated\n")
                    continue

                swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(init2grasp_path_temp, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(grasp2init_path_temp, w_target=temp_mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                num_grasp += 1
                print("\n----------grasp---------------\n")

                # compare swept volumes
                swept_center_temp, swept_verts_temp = rac.get_swept_center(swept_verts1_temp+swept_verts2_temp, scene_info, MAX_HEIGHT)
                temp_swept_size = get_swept_volume_size(swept_verts_temp)
                if temp_swept_size < swept_size:
                    swept_size = temp_swept_size
                    MAIN_swept_volume1 = swept_volume1_temp
                    MAIN_swept_volume2 = swept_volume2_temp
                    init2grasp_path = init2grasp_path_temp
                    grasp2init_path = grasp2init_path_temp
                    swept_center = swept_center_temp
                    swept_verts = swept_verts_temp
                    W_TARGET = temp_mod_bbox
                if num_grasp == 1: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    grasp_dof = rac.grasp_verify(target_grasp_pos, target_grasp_quat, new_offset=True)
                    init2grasp_path.append(grasp_dof)

                    # img_id = int(sys.argv[1])
                    curr_time = time.localtime()
                    new_folder = 'test_data/test_real_experiment/target_path_saved/' + str(curr_time[1]) + '.' + str(curr_time[2]) + '.' + str(curr_time[3]) + '.' + str(curr_time[4]) + '/'
                    os.makedirs(new_folder)

                    np.save(new_folder + "/init2grasp_path.npy", np.array(init2grasp_path))
                    np.save(new_folder + "/grasp2init_path.npy", np.array(grasp2init_path))
                    np.save(new_folder + "/grasp_idx.npy", np.array(grasp_idx))
                    np.save(new_folder + "/grasp_bbox_verts.npy", temp_mod_bbox[0])
                    np.save(new_folder + "/grasp_bbox_faces.npy", temp_mod_bbox[1])

                    sys.exit(1)
                    break
            if num_grasp > 0:
                break


        init2grasp_path = np.load("test_data/test_real_experiment/target_path_saved/banana/grasp2init_path_banana_center.npy", allow_pickle=True)
        grasp2init_path = np.load("test_data/test_real_experiment/target_path_saved/banana/init2grasp_path_banana_center.npy", allow_pickle=True)
        
        MAIN_swept_volume1, swept_verts1_temp = rac.get_swept_volume(init2grasp_path, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
        W_TARGET = rac.modify_grasp_bbox(init2grasp_path[-1], target_obj_mesh, visualize=False)
        MAIN_swept_volume2, swept_verts2_temp = rac.get_swept_volume(grasp2init_path, w_target=W_TARGET, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)

        swept_center, swept_verts = rac.get_swept_center(swept_verts1_temp+swept_verts2_temp, scene_info, MAX_HEIGHT)
        swept_size = get_swept_volume_size(swept_verts)

        s_time = time.time()
        num_in_swept = 0
        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            is_collision = True

            # add target obj
            if k == NUM_OF_OBJECTS - 1:
                object_pose.p = gymapi.Vec3(GT_TARGET_POS[0], GT_TARGET_POS[1], GT_TARGET_POS[2])

                file_path = object_collision_files[OBJ_FILE_IDX_LIST[-1]]
                collision_mesh = obj_reader(asset_root + file_path)
                collision_mesh.set_scale(object_scaling_factor[-1])
                collision_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[-1]])
                verts, tris = collision_mesh.get_bounding_box_mesh()
                temp_center = collision_mesh.get_center()
                temp_bounding_box = collision_mesh.get_bounding_box()

                m = fcl.BVHModel()
                m.beginModel(len(verts), len(tris))
                m.addSubModel(verts, tris)
                m.endModel()
                t = fcl.Transform(np.array(GT_TARGET_POS))
                is_collision = False

            # random selec obj location
            while is_collision:
                tx = np.random.uniform(0.35, table_dims.x + 0.2)
                ty = np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2)
                tz = table_dims.z + 0.08

                object_pose.p = gymapi.Vec3(tx, ty, tz)

                # file_path = object_collision_files[OBJ_FILE_IDX_LIST[k]]
                # collision_mesh = obj_reader(asset_root + file_path)
                # collision_mesh.set_scale(object_scaling_factor[k])
                # collision_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[k]])
                # collision_mesh.add_offset([tx,ty,tz])

                file_path = object_collision_files[OBJ_FILE_IDX_LIST[k]]
                collision_mesh = obj_reader(asset_root + file_path)
                collision_mesh.set_scale(object_scaling_factor[k])
                collision_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[k]][:2] + [0])
                collision_mesh.add_offset([tx,ty,0.1])

                # obstacle_obj_mesh = copy.deepcopy(collision_mesh.add_offset([tx,tz,0.1]))
                
                verts, tris = collision_mesh.get_bounding_box_mesh()
                temp_center = collision_mesh.get_center()
                temp_bounding_box = collision_mesh.get_bounding_box()

                # new obj
                m = fcl.BVHModel()
                m.beginModel(len(verts), len(tris))
                m.addSubModel(verts, tris)
                m.endModel()
                # t = fcl.Transform(np.array([tx,ty,tz]))
                t = fcl.Transform()
                

                # check collision
                req = fcl.CollisionRequest()
                rdata = fcl.CollisionData(request = req)
                objs_manager.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)
                is_collision = rdata.result.is_collision # update collision status

                MAIN_swept_volume1.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)
                is_collision_swept1 = rdata.result.is_collision
                MAIN_swept_volume2.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)
                is_collision_swept2 = rdata.result.is_collision

                if not is_collision:
                    dist = np.sqrt((tx - GT_TARGET_POS[0])**2 + (ty - GT_TARGET_POS[1])**2)
                    if dist <= 0.2:
                        is_collision = True
                        print("target contact recalc")
                        continue

                    for obj in GT_OBJ_POS_LIST:
                        dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                        if dist <= 0.16:
                            is_collision = True
                            print("recalc")
                            continue

                    if not is_collision_swept1 and not is_collision_swept2 and num_in_swept < NUM_SWEPT_COLLISION:
                        is_collision = True
                        assert time.time() - s_time < 60, "time out"
                        continue
            
            if is_collision_swept1 or is_collision_swept2 and k != NUM_OF_OBJECTS - 1:
                num_in_swept += 1
            
            GT_OBJ_MESH_LIST.append([verts, tris])
            GT_OBJ_POS_LIST.append([object_pose.p.x, object_pose.p.y])

            object_handles.append(gym.create_actor(envs[-1], 
                                                object_assets[OBJ_FILE_IDX_LIST[k]], 
                                                object_pose, 
                                                "object" + str(k) + str(i), 0, 2**(k+1), k+1))
            gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
            object_reader_tracker.append(collision_mesh)
            object_status_list.append([temp_center, temp_bounding_box])
            object_collision_lib.append(m)
            obstacle_objs.append(fcl.CollisionObject(m, t))
            objs_manager.registerObjects(obstacle_objs)
            objs_manager.setup()

        #set up global camera to record configuration
        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        viewpoint_candidate = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(body_cam_handles[-1], envs[-1], 
                                viewpoint_candidate, 
                                camera_focus)
        
        GT_OBJ_MESH_LIST[NUM_OF_OBJECTS - 1] = target_obj_mesh


    # rac.target_mesh = copy.deepcopy(GT_OBJ_MESH_LIST[-1])
    # rac.obj_mesh = copy.deepcopy(GT_OBJ_MESH_LIST[:-1])
    # rac.obj_pos_list = copy.deepcopy(GT_OBJ_POS_LIST[:-1])
    # rac.check_collision_models(init2grasp_angels_temp, scene_info=scene_info)

    # os.makedirs('test_data/test_real_experiment/target_path_saved/')
    # save_scene(init2grasp_path, grasp2init_path, rac.obj_pos_list, GT_OBJ_POS_LIST, NUM_OF_OBJECTS, scene_info,
    #                 rac.target_mesh, rac.obj_mesh, target_obj_pos, GT_TARGET_POS, rac.obstacles_num, W_TARGET,
    #                 None, None, None, 'test_data/test_real_experiment/target_path_saved/1.npy', False)
    
    #*************************************************************************************************#

    #*************************************************************************************************#
    cam_pos = gymapi.Vec3(2.2, 0, 0.5)
    cam_target = gymapi.Vec3(0, 0, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                    gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                    gymapi.Vec3(1.0, 0.0, 0.0))
    #*************************************************************************************************#

    # empty loop here to test functionalities
    #*************************************************************************************************#
    real_position = False
    flex_collision_models = []
    object_mesh = []
    t = 0
    #while not gym.query_viewer_has_closed(viewer):
    for t in range(2000):
        if not real_position:
            gym.set_dof_target_position(envs[-1], spj, 0)
            gym.set_dof_target_position(envs[-1], slj, -math.pi/2)
            gym.set_dof_target_position(envs[-1], ej,  0)
            gym.set_dof_target_position(envs[-1], wj1, -math.pi/2)
            gym.set_dof_target_position(envs[-1], wj2, 0)
            gym.set_dof_target_position(envs[-1], wj3, 0)
            real_position = True

        if t == 999:
            for i in range(len(object_handles)):
                element = object_handles[i]
                states = gym.get_actor_rigid_body_states(envs[-1], element, 1)
                rotation = np.array(states[0][0][1])
                translation = np.array(states[0][0][0])
                rotation = np.array(rotation.item())
                translation = np.array(translation.item())
                object_status_list[i][0] += translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                flex_collision_models.append([fcl.CollisionObject(object_collision_lib[i], tf), 0])

                temp_obj = object_reader_tracker[i]
                temp_obj.set_offset(translation)
                all_lines = [] 
                vertices, faces = temp_obj.get_bounding_box_mesh()
                object_mesh.append([vertices, faces])
                for v1, v2, v3 in faces:
                    all_lines += list(vertices[v1])
                    all_lines += list(vertices[v2])
                    all_lines += list(vertices[v1])
                    all_lines += list(vertices[v3])
                    all_lines += list(vertices[v2])
                    all_lines += list(vertices[v3])

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        gym.sync_frame_time(sim)
    #*************************************************************************************************#

    seed_state = [0.0]*ik_solver2.number_of_joints
    dof_result = None
    trial = 0
    potential_result = None
    counter = 0
    test_cam = gym.create_camera_sensor(envs[-1], camera_props)
    coverage_score = 0
    sequence_count = 0
    acquire_counter = 0
    need_acquire = False
    object_dict = {}

    saved_scene_vertices = None
    saved_scene_faces = None

    # creating new folder
    curr_time = time.localtime()
    new_folder = 'test_data/test_real_experiment/' + str(curr_time[1]) + '.' + str(curr_time[2]) + '.' + str(curr_time[3]) + '.' + str(curr_time[4]) + '/'

    os.makedirs(new_folder + 'MCTS*/test_results/')
    os.makedirs(new_folder + 'MCTS*/test_image/')
    os.makedirs(new_folder + 'MCTS*/test_seg_image/')
    os.makedirs(new_folder + 'MCTS*/test_depth_image/')
    os.makedirs(new_folder + 'MCTS*/test_cam_info/')

    print("\n\n--------------------------- Complete Sensing MCTS* ---------------------------")
    test_name = "MCTS*/"
    save_scene_folder = test_name +"test_results/"
    test_result_folder = new_folder + test_name + "test_results/"
    image_folder = new_folder + test_name

    seed_state = [0.0]*ik_solver2.number_of_joints
    dof_result = None
    trial = 0
    potential_result = None
    counter = 0
    test_cam = gym.create_camera_sensor(envs[-1], camera_props)
    coverage_score = 0
    sequence_count = 0
    acquire_counter = 0
    need_acquire = False
    object_dict = {}
    mcts_attempts = 0
    cam_dofs = []
    total_view_time_comsumption = 0
    cam_time = 0
    saved_scene_vertices = None
    saved_scene_faces = None
    ML_MCTS_ins = mct.multi_level_MCTS_algo(None, None, scene_info=scene_info, swept_volume1=None, swept_volume2=None, obj_mesh=rac.obj_mesh)
    OFFSET = np.array([0.3, -0.6, 0.05])
    scene = global_scene(1.0, 1.2, 0.85, OFFSET, table_dims.x, table_dims.y - 0.04, drawer_height, table_dims.z - 0.05)
    ML_MCTS_ins.swept_volume1 = MAIN_swept_volume1
    ML_MCTS_ins.swept_volume2 = MAIN_swept_volume2
    #active sensing here
    obj_pos_MCTS = {}
    obj_mesh_MCTS = {}
    target_pos_MCT = None
    target_obj_mesh = None
    target_obj_pos = None
    obj_pcd = {}
    MAX_HEIGHT = get_max_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST)
    valid_area_cluster = None
    potential_center_cluster = None
    is_tunnel_covered = False
    is_target_detected = False
    is_cluster_covered = False
    run_mcts = False
    mcts_out_angle = None
    start_time = time.time()

    while not gym.query_viewer_has_closed(viewer):
        if run_mcts and is_target_detected and not need_acquire:
            print("--------------------PLANNING START------------------------")
            ML_MCTS_ins.scenario_check() # terminate scenario if it's not valid
            total_view_time_comsumption += time.time() - start_time
            mcts_attempts += 1

            is_plan_success, child_node_list = ML_MCTS_ins.run_mcts(30)

            save_scene(init2grasp_path, grasp2init_path, rac.obj_pos_list, GT_OBJ_POS_LIST, NUM_OF_OBJECTS, scene_info,
                           rac.target_mesh, rac.obj_mesh, target_obj_pos, GT_TARGET_POS, rac.obstacles_num, W_TARGET,
                           ML_MCTS_ins.unknown_area, ML_MCTS_ins.valid_area, ML_MCTS_ins.potential_centers, save_scene_folder, is_plan_success)

            if is_plan_success:
                print("!!!!!!!!!!!!!!MCTS* planning Success!!!!!!!!!!!!!!!!!!!!!")
                ML_MCTS_ins.global_optimization()
                # ML_MCTS_ins.animate_whole_sequence()
                res_plan = ML_MCTS_ins.save_planning_results()
                num_collision_obj_ = len(ML_MCTS_ins.track_level_steps_[0][0].check_collision_w_swept())
                write_result(new_folder, save_scene_folder, scene.num_observation, len(curr_config), ML_MCTS_ins.time_consumption_, ML_MCTS_ins.total_steps_, ML_MCTS_ins.calculate_total_length_travelled(), ML_MCTS_ins.calculate_total_length_displacement(), num_collision_obj_, mcts_attempts, total_view_time_comsumption, cam_dofs, cam_time, res_plan)
                ML_MCTS_ins.track_level_steps_[0][0].scene_saver(test_result_folder + 'start_layout.png')
                ML_MCTS_ins.track_level_steps_[-1][-1].scene_saver(test_result_folder + 'final_layout.png')
                break
            else:
                ML_MCTS_ins.MCTS_ins.MCTS_tree_.scene_saver(test_result_folder + 'failed_case' + str(scene.num_observation))
                num_collision_obj_ = len(ML_MCTS_ins.MCTS_ins.MCTS_tree_.check_collision_w_swept())
                write_result(new_folder, save_scene_folder, scene.num_observation, len(curr_config), ML_MCTS_ins.time_consumption_, "Failed", "Failed", "Failed", num_collision_obj_, mcts_attempts, total_view_time_comsumption, cam_dofs, cam_time, None)
                
                if len(potential_center_cluster) == 0:
                    print("!!!!!!!!!!!!!!Planning Faild without unobserved area!!!!!!!!!!!!!!!!!!!!!")
                    is_cluster_covered = True
                    break
                    
                else:
                    cam_time_start = time.time()
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
                    for idx in region_list:
                        mcts_out_angle = RC.cal_cam_angle_for_area(valid_area_cluster[idx], ML_MCTS_ins.curr_config_ + [target_pos_MCT], scene_info, visualize=False)
                        if mcts_out_angle is not None:
                            break

                    if mcts_out_angle is None:
                        print("------ RUN FAILED WITH UNOBSERVABLE AREA ------")
                        break

                    mcts_selected_cluster = valid_area_cluster[idx]
                    run_mcts = False
                    cam_time += time.time() - cam_time_start
                    
        if need_acquire:
            if acquire_counter > 500:
                gym.clear_lines(viewer)
                gym.render_all_camera_sensors(sim)
                for q in body_cam_handles:
                    color_image = gym.get_camera_image(sim, envs[-1], body_cam_handles[q], gymapi.IMAGE_COLOR)
                    depth_image = gym.get_camera_image(sim, envs[-1], body_cam_handles[q], gymapi.IMAGE_DEPTH)
                    seg_image = gym.get_camera_image(sim, envs[-1], body_cam_handles[q], gymapi.IMAGE_SEGMENTATION)

                    if (q == 0):
                        K = np.array([[911.445649104, 0, 641.169],
                            [0, 891.51236121, 352.77],
                            [0, 0, 1]])

                        color_img_saved = write_to_image(color_image, image_folder + 'test_image/' + str(sequence_count) + '.png')
                        write_to_seg_image(seg_image, image_folder + 'test_seg_image/' + str(sequence_count) + '.png')
                        write_to_depth_image(depth_image, image_folder + 'test_depth_image/' + str(sequence_count) + '.png')
        
                        temp_cam = body_cam_handles[q] 
                        cam_rotation = gym.get_camera_transform(sim, envs[-1], temp_cam).r
                        cam_translation = gym.get_camera_transform(sim, envs[-1], temp_cam).p
                        new_rgb_image = convert_rgb_image(color_image)
                        new_seg_image = convert_seg_image(seg_image)
                        new_depth_image = convert_depth_image(depth_image)

                        new_cam_rotation = np.array([cam_rotation.x,
                                                    cam_rotation.y,
                                                    cam_rotation.z,
                                                    cam_rotation.w])
                        new_cam_translation = np.array([cam_translation.x,
                                                        cam_translation.y,
                                                        cam_translation.z])
                        
                        write_to_cam_pose(new_cam_rotation, new_cam_translation, image_folder + 'test_cam_info/' + str(sequence_count) + '.npy')

                        # write_for_contact_grasp(color_img_saved, seg_image, -depth_image, K, new_cam_rotation, new_cam_translation, new_folder + 'test_npy/' + str(sequence_count)+'.npy')
                        dist1 = np.linalg.norm(final_rotation - new_cam_rotation)
                        dist3 = np.linalg.norm(final_rotation - new_cam_rotation*-1)
                        dist2 = np.linalg.norm(final_translation - new_cam_translation)

                        if (dist1 < 1e-2 or dist3 < 1e-2) and dist2 < 1e-2:
                            print ('----- start point cloud extraction -----')
                            pc_extractor_grasp(new_rgb_image, new_depth_image, new_seg_image, new_cam_rotation, new_cam_translation, object_dict, table_dims.z)
                            
                            for i in object_dict:
                                if i != NUM_OF_OBJECTS:
                                    obj_pos_MCTS[i-1] = GT_OBJ_POS_LIST[i-1]
                                    obj_mesh_MCTS[i-1] = GT_OBJ_MESH_LIST[i-1]
                                else:
                                    target_obj_pos = GT_OBJ_POS_LIST[i-1]
                                    target_obj_mesh = GT_OBJ_MESH_LIST[i-1]
                                    is_target_detected = True
                                # mask = new_seg_image == 1
                                # temp_seg_image = copy.deepcopy(new_seg_image)
                                # mask = new_seg_image == i
                                # temp_seg_image[~mask] = 0
                                # temp_seg_image[mask] = 1

                                # point_cloud, pcd = RC.write_to_pointcloud(new_rgb_image, new_depth_image, temp_seg_image, new_cam_rotation, new_cam_translation, visualization=False)
                                # if i in obj_pcd.keys():
                                #     obj_pcd[i] += pcd
                                # else:
                                #     obj_pcd[i] = pcd

                                # # find matching object mesh file
                                # downpcd = obj_pcd[i].voxel_down_sample(voxel_size=0.005) # downsampe pcd
                                # # downpcd = obj_pcd[i]

                                # if i != NUM_OF_OBJECTS:
                                #     # obstacle object matching
                                #     obstacle_obj_mesh, obj_pos, dist, obj_name = RC.get_matching_mesh(downpcd, OBSTACLE_OBJ_INDEX, visualize=False)
                                #     obj_pos = obj_pos - object_offset[OBJ_FILE_IDX_LIST[i-1]]

                                #     if dist < 3:
                                #         print("idx",i,"mesh added")
                                #         obj_pos_MCTS[i] = obj_pos[0:2].tolist()
                                #         obj_mesh_MCTS[i] = obstacle_obj_mesh

                                #         dy = obj_pos_MCTS[i][1] - GT_OBJ_POS_LIST[i-1][1]
                                #         dx = obj_pos_MCTS[i][0] - GT_OBJ_POS_LIST[i-1][0]
                                #         print("diff:", np.sqrt(dy**2 + dx**2), "dist", dist, '\n')
                                # else:
                                #     # target obj matching
                                #     temp_target_obj_mesh, temp_target_obj_pos, dist, obj_name = RC.get_matching_mesh(downpcd, TARGET_OBJ_INDEX, visualize=False)
                                #     temp_target_obj_pos = temp_target_obj_pos - object_offset[OBJ_FILE_IDX_LIST[i-1]]

                                #     if dist < 3:
                                #         print("idx",i,"mesh added")
                                #         target_obj_mesh = copy.deepcopy(temp_target_obj_mesh)
                                #         target_obj_pos = copy.deepcopy(temp_target_obj_pos[0:2].tolist())

                                #         dy = target_obj_pos[1] - GT_OBJ_POS_LIST[i-1][1]
                                #         dx = target_obj_pos[0] - GT_OBJ_POS_LIST[i-1][0]
                                #         print("diff:", np.sqrt(dy**2 + dx**2), "dist", dist, '\n')

                                #     if not is_target_detected and dist < 3:
                                #         # read grasp data
                                #         grasp_file = "/".join(obj_name.split('/')[:5]) + "/grasp_dict.npy"
                                #         grasp_data = np.load(grasp_file, allow_pickle=True)

                                #         # generate swept volume
                                #         num_grasp = 0
                                #         swept_size = sys.maxsize
                                #         grasp_list = np.arange(len(grasp_data))
                                #         np.random.shuffle(np.arange(len(grasp_list)))

                                #         for grasp_idx in grasp_list[:20]:
                                #             target_grasp_pos = grasp_data[grasp_idx]['target_pos']
                                #             target_grasp_quat = grasp_data[grasp_idx]['target_quat']
                                #             target_grasp_pos[:2] = target_grasp_pos[:2] + target_obj_pos[:2]

                                #             init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
                                #             grasp2init_angels_temp = rac.grasp_verify(target_grasp_pos + [0,0,0.01], target_grasp_quat)

                                #             if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
                                #                 print("skip imposible grasp")
                                #                 continue

                                #             init2grasp_path_temp = RC.get_path2grasp(rac, init2grasp_angels_temp, scene_info, target_mesh=target_obj_mesh, time_limit=30)
                                #             temp_mod_bbox = rac.modify_grasp_bbox(init2grasp_angels_temp, target_obj_mesh, visualize=False)
                                            # new_grasp2init_path = RC.get_path2start(rac, grasp2init_angels_temp, temp_mod_bbox, scene_info)
                                #             grasp2init_path_temp = RC.get_path2start(rac, grasp2init_angels_temp, temp_mod_bbox, scene_info, time_limit=30)

                                #             if init2grasp_path_temp is None or grasp2init_path_temp is None:
                                #                 print("No path generated\n")
                                #                 continue

                                #             swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(init2grasp_path_temp, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                                #             swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(grasp2init_path_temp, w_target=temp_mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                                #             num_grasp += 1
                                #             is_target_detected = True

                                #             # compare swept volumes
                                #             swept_center_temp, swept_verts_temp = rac.get_swept_center(swept_verts1_temp+swept_verts2_temp, scene_info, MAX_HEIGHT)
                                #             temp_swept_size = get_swept_volume_size(swept_verts_temp)
                                #             if temp_swept_size < swept_size:
                                #                 swept_size = temp_swept_size
                                #                 ML_MCTS_ins.swept_volume1 = swept_volume1_temp
                                #                 ML_MCTS_ins.swept_volume2 = swept_volume2_temp
                                #                 init2grasp_path = init2grasp_path_temp
                                #                 grasp2init_path = grasp2init_path_temp
                                #                 swept_center = swept_center_temp
                                #                 swept_verts = swept_verts_temp
                                #                 W_TARGET = temp_mod_bbox

                                #             if num_grasp == 5:
                                #                 break
                                #         print("\n!!!!!!!!!!!!!!!!!!!", num_grasp ,'grasp generated!!!!!!!!!!!!!!!!!!!!!!!\n')

                            _ = scene.register_camera_view(list(new_cam_rotation), list(new_cam_translation), new_depth_image, object_dict)

                            if is_target_detected:
                                coverage_score, _ = swept_coverage_check(scene, swept_verts, rac, scene_info, MAX_HEIGHT)
                                print("----- Swept volume covered", coverage_score, '-----')
                                
                                # get obj pose
                                obj_mesh_MCTS = dict(sorted(obj_mesh_MCTS.items()))
                                obj_pos_MCTS = dict(sorted(obj_pos_MCTS.items()))
                                update_rac_val(rac, target_obj_mesh, obj_mesh_MCTS, obj_pos_MCTS)

                                # unobserved area processing
                                curr_config, target_pos_MCT = rac.get_MCT_config(copy.deepcopy(rac.obj_pos_list), copy.deepcopy(rac.obj_mesh), copy.deepcopy(target_obj_pos), copy.deepcopy(rac.target_mesh))
                                unknown_area = get_unobserved_area_w_height(scene, asset_root, object_asset_files, OBJ_FILE_IDX_LIST)

                                # Process unknown area
                                unknown_area, potential_center_cluster, valid_area_cluster = RC.process_unknown_area(unknown_area, curr_config, target_pos_MCT, MIN_RADIUS * 100, valid_center_num=5)
                                potential_centers = np.array(sum(potential_center_cluster, []))
                                valid_area = np.array(sum(valid_area_cluster, []))

                                # initialize & check scenario
                                update_MCTS_val(ML_MCTS_ins, curr_config, target_pos_MCT, rac.obj_mesh, unknown_area, valid_area, potential_centers)
                                ML_MCTS_ins.init_MCTS()

                                # scene save
                                ML_MCTS_ins.MCTS_ins.MCTS_tree_.scene_saver(test_result_folder + 'scene_capture' + str(scene.num_observation) + '.png')

                            sequence_count += 1

                need_acquire = False
                acquire_counter = 0
            else:
                acquire_counter += 1
        else:
            #add scene surface mesh first
            flexible_collision_models = []
            
            scene_mesh = scene.get_surface_collision_mesh()
            scene_vertices = np.asarray(scene_mesh.vertices)
            scene_faces = np.asarray(scene_mesh.triangles)

            saved_scene_vertices = copy.deepcopy(scene_vertices)
            saved_scene_faces = copy.deepcopy(scene_faces)

            all_lines = []
            for v1, v2, v3 in scene_faces:
                all_lines += list(scene_vertices[v1])
                all_lines += list(scene_vertices[v2])
                all_lines += list(scene_vertices[v1])
                all_lines += list(scene_vertices[v3])
                all_lines += list(scene_vertices[v2])
                all_lines += list(scene_vertices[v3])

            gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])

            m = fcl.BVHModel()
            m.beginModel(len(scene_vertices), len(scene_faces))
            m.addSubModel(scene_vertices, scene_faces)
            m.endModel()

            flexible_collision_models.append(fcl.CollisionObject(m))
            print ('There are {0} objects detected\n'.format(len(object_dict)))

            for obj_id, obj_ins in object_dict.items():
                if obj_id != 0:
                    object_vertices, object_faces = obj_ins.get_collision_mesh()
                    m = fcl.BVHModel()
                    m.beginModel(len(object_vertices), len(object_faces))
                    m.addSubModel(object_vertices, object_faces)
                    m.endModel()

                    flexible_collision_models.append(fcl.CollisionObject(m))

            # print ('finish adding flexible collision model')

            end_state_collision_free = False
            while not end_state_collision_free:
                if coverage_score >=0.85: # swept volume observed
                    cam_time_start = time.time()
                    if not is_tunnel_covered:
                        obj_idx, collision_points = ML_MCTS_ins.unknown_tunnel_check()

                        if not collision_points:
                            print("\n----- Tunnel Covered -----")
                            is_tunnel_covered = True
                            run_mcts = True
                            break
                        
                        print("\n----- covering grasp tunnel -----")
                        for cluster in valid_area_cluster:
                            exist = np.any(np.isin(cluster, collision_points).all(1))
                            if exist:
                                check_cluster = cluster
                                break
                        if check_cluster is None:
                            print("ERROR!!!!!!!!!!!!!!!!!!!!!!")
                            pdb.set_trace()

                        cluster_to_view = check_cluster
                        cluster_angles = RC.cal_cam_angle_for_area(check_cluster, curr_config + [target_pos_MCT], scene_info, visualize=False)

                    elif not is_cluster_covered:
                        run_mcts = True
                        if len(ML_MCTS_ins.valid_area) == 0:
                            print("\n----- No unknown areas -----")
                            is_cluster_covered = True
                            continue

                        print("\n----- Checking suggested unknown_area from MCTS -----")
                        cluster_angles = mcts_out_angle
                        cluster_to_view = mcts_selected_cluster

                    else:
                        print("\n----- No observation needed -----")
                        run_mcts = True
                        continue

                    # choosing biggest cluster
                    end_points = cluster_angles['loc']
                    focus_point = cluster_angles['foc'][0]
                    camera_loc, camera_focus, dof_result = cam_loc_selection_for_clusters(sim, envs[-1], test_cam, focus_point, end_points, scene_info, cluster_to_view)
                    cam_time += time.time() - cam_time_start
                        
                elif not is_target_detected: # target is not detected, normal active sensing
                    cam_time_start = time.time()
                    camera_loc, camera_focus, dof_result = random_sample_guided_selection(sim, envs[-1], test_cam, scene)
                    cam_time += time.time() - cam_time_start

                else: # target is detected, covering swept volume
                    cam_time_start = time.time()
                    if swept_center is not None:
                        focus_point = swept_center
                        swept_center = None
                    else: # continue tracking swept volume
                        _, focus_point = swept_coverage_check(scene, swept_verts, rac, scene_info, MAX_HEIGHT)
                    
                    camera_loc, camera_focus, dof_result = random_sample_swept_volume_selection(sim, envs[-1], test_cam, focus_point, swept_verts)
                    cam_time += time.time() - cam_time_start

                gym.set_camera_location(test_cam, envs[-1], camera_loc, camera_focus)
                target_pos = gym.get_camera_transform(sim, envs[-1], test_cam).p
                target_quat = gym.get_camera_transform(sim, envs[-1], test_cam).r

                if dof_result:
                    end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)

                    if end_state_collision_free:
                        final_translation = np.array([target_pos.x, target_pos.y, target_pos.z])
                        final_rotation = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
                        gym.set_dof_target_position(envs[-1], spj, dof_result[0])
                        gym.set_dof_target_position(envs[-1], slj, dof_result[1])
                        gym.set_dof_target_position(envs[-1], ej,  dof_result[2])
                        gym.set_dof_target_position(envs[-1], wj1, dof_result[3])
                        gym.set_dof_target_position(envs[-1], wj2, dof_result[4])
                        gym.set_dof_target_position(envs[-1], wj3, dof_result[5])
                        need_acquire = True
                        cam_dofs.append(dof_result)
    
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    print('Done')
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    sys.exit(1)