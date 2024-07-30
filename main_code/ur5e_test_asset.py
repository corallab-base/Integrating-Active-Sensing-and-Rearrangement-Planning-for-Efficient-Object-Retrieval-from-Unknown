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
import robot_arm_configuration as RC
import MCTS_algo_ICRA as mct
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
table_dims = gymapi.Vec3(0.56, 0.86, 0.10)
piece_width = 0.03
min_num_of_objects = 15
max_num_of_objects = 20
max_scaling_factor = 0
fall_height = table_dims.z
max_drawer_height = 0.5
min_drawer_height = 0.5
ADD_COVER = True
#*************************************************************************************************#

#helper functions
#*************************************************************************************************#
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

def cam_loc_selection_for_clusters(sim, env, test_cam, scene, center, end_point, scene_info):
    camera_pose_list = []
    camera_setting_list = []
    cam_height = scene_info[3] * 0.80
    floor_height = scene_info[2] + 0.01

    dist = 0
    while len(camera_pose_list) < 10:
        loc_vec = (end_point - center) / np.linalg.norm(end_point - center) / 100
        cam_loc = end_point + loc_vec * dist
        dist += 1

        if cam_loc[0] < 0.0: break

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
        if dof_result:
            end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)


            if end_state_collision_free:
                camera_pose_list.append(camera_pose)
                camera_setting_list.append([camera_loc, camera_focus, dof_result])
         
    best_cam_pose_index = get_best_cam_pose(scene, camera_pose_list)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]


def random_sample_swept_volume_selection(sim, env, test_cam, scene, swept_center):
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
         
    best_cam_pose_index = get_best_cam_pose(scene, camera_pose_list)
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

def swept_coverage_check(scene, swept_verts, rac, scene_info):
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

    next_center, _ = rac.get_swept_center([new_verts], scene_info)
    return covered / len(swept_verts), next_center

def get_unobserved_area(scene):
    floor = scene.scene_[:scene.x_limit_, scene.y_left_+1 :(scene.y_left_ + scene.y_limit_-1), scene.g_height_]
    unknown_area = np.argwhere(floor == 0)[:,:2]
    unknown_area[:, 0] += 29
    unknown_area[:, 1] -= int((floor.shape[1]) / 2)

    temp = copy.deepcopy(unknown_area)
    unknown_area[:, 0] = -temp[:, 1]
    unknown_area[:, 1] = temp[:, 0]
    return unknown_area

def get_unobserved_area_w_height(scene):
    # mesh = o3d.io.read_triangle_mesh(obstacle_files)
    # verts = np.asarray(mesh.vertices)
    # min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
    # max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
    # for tx, ty, tz in verts:
    #     min_x = min(min_x, tx)
    #     min_y = min(min_y, ty)
    #     min_z = min(min_z, tz)
    #     max_x = max(max_x, tx)
    #     max_y = max(max_y, ty)
    #     max_z = max(max_z, tz)
    # height = int((max_z - min_z) * 100) + 1 # 15, 0.14119999739341438
    
    floor = scene.scene_[:scene.x_limit_, scene.y_left_+1 :(scene.y_left_ + scene.y_limit_-1), scene.g_height_: scene.g_height_ + 15]

    unknown_layer = set(map(tuple, np.argwhere(floor[:,:,0] == 0)[:,:2]))
    for i in range(1,15):
        temp_unknown = set(map(tuple, np.argwhere(floor[:,:,i] == 0)[:,:2]))
        unknown_layer = unknown_layer.intersection(temp_unknown)
    unknown_area = np.array(list(unknown_layer))

    unknown_area[:, 0] += 29
    unknown_area[:, 1] -= int((floor.shape[1]) / 2)

    temp = copy.deepcopy(unknown_area)
    unknown_area[:, 0] = -temp[:, 1]
    unknown_area[:, 1] = temp[:, 0]

    return unknown_area

def clustering(unknown_area, visualize=False):
    if len(unknown_area) == 0:
        return unknown_area
    point_list = copy.deepcopy(unknown_area)
    cluster_list = []
    while True:
        idx = np.random.randint(len(point_list))
        point = point_list[idx]
        point_list = np.delete(point_list, idx, axis=0)
        cluster = [point.tolist()]
        for x, y in cluster:
            check1 = [x+1,y]
            if not check1 in cluster:
                idx1 = np.argwhere((point_list == np.array(check1)).all(1))
                if idx1.size != 0:
                    cluster.append(check1)
                    point_list = np.delete(point_list, idx1, axis=0)

            check2 = [x-1,y]
            if not check2 in cluster:
                idx2 = np.argwhere((point_list == np.array(check2)).all(1))
                if idx2.size != 0:
                    cluster.append(check2)
                    point_list = np.delete(point_list, idx2, axis=0)

            check3 = [x,y+1]
            if not check3 in cluster:
                idx3 = np.argwhere((point_list == np.array(check3)).all(1))
                if idx3.size != 0:
                    cluster.append(check3)
                    point_list = np.delete(point_list, idx3, axis=0)

            check4 = [x,y-1]
            if not check4 in cluster:
                idx4 = np.argwhere((point_list == np.array(check4)).all(1))
                if idx4.size != 0:
                    cluster.append(check4)
                    point_list = np.delete(point_list, idx4, axis=0)

            check5 = [x+1,y+1]
            if not check5 in cluster:
                idx5 = np.argwhere((point_list == np.array(check5)).all(1))
                if idx5.size != 0:
                    cluster.append(check5)
                    point_list = np.delete(point_list, idx5, axis=0)

            check6 = [x+1,y-1]
            if not check6 in cluster:
                idx6 = np.argwhere((point_list == np.array(check6)).all(1))
                if idx6.size != 0:
                    cluster.append(check6)
                    point_list = np.delete(point_list, idx6, axis=0)

            check7 = [x-1,y+1]
            if not check7 in cluster:
                idx7 = np.argwhere((point_list == np.array(check7)).all(1))
                if idx7.size != 0:
                    cluster.append(check7)
                    point_list = np.delete(point_list, idx7, axis=0)

            check8 = [x-1,y-1]
            if not check8 in cluster:
                idx8 = np.argwhere((point_list == np.array(check8)).all(1))
                if idx8.size != 0:
                    cluster.append(check8)
                    point_list = np.delete(point_list, idx8, axis=0)

        cluster_list.append(cluster)
        if point_list.size == 0:
            break

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        for cluster in cluster_list:
            plt.scatter(np.array(cluster)[:,0], np.array(cluster)[:,1])
        plt.show()

    return cluster_list

def delete_obj_spots(curr_config, target_pos, unknown_area, visualize=False):
    if len(unknown_area) == 0:
        return unknown_area
    
    pos_list = curr_config + [target_pos]
    offset_list = []
    p_list = [[0,i] for i in range(6)]
    n_list = [[0,-i] for i in range(6)]
    for degree in range(0, 180, 1):
        theta = np.radians(degree)
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array(((cos,-sin), (sin, cos)))
        for i in range(6):
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
    valid_points = []
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
        valid_points += point

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        plt.scatter(np.array(cluster)[:,0], np.array(cluster)[:,1], color='black')
        if valid_center:
            plt.scatter(np.array(valid_points)[:, 0], np.array(valid_points)[:, 1], color='green')
            plt.scatter(np.array(valid_center)[:, 0], np.array(valid_center)[:, 1], color='red')
        plt.show()

    return valid_points, valid_center

def get_filtered_clusters(cluster_list, visualize=True):
    filtered_cluster = []
    valid_area = []
    potential_centers = []
    for cluster in cluster_list:
        valid_points, valid_center = check_obj_fit(cluster, 5, visualize=False)
        if len(valid_points) > 5:
            filtered_cluster += cluster
            valid_area += valid_points
            potential_centers += valid_center

    filtered_cluster = np.array(filtered_cluster)
    valid_area = np.array(valid_area)
    potential_centers = np.array(potential_centers)

    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        if len(filtered_cluster) > 0:
            plt.scatter(np.array(filtered_cluster)[:,0], np.array(filtered_cluster)[:,1], color='black')
            plt.scatter(np.array(valid_area)[:,0], np.array(valid_area)[:,1], color='green')
            plt.scatter(np.array(potential_centers)[:,0], np.array(potential_centers)[:,1], color='red')
        plt.show()

    return filtered_cluster, valid_area, potential_centers

def cal_cam_angle_for_area(valid_points, curr_config, scene_info, visualize=False):
    valid_list = clustering(np.array(valid_points), visualize=False)
    valid_list = sorted(valid_list, key=len, reverse=True)
    left_point =  int(-scene_info[1]/2 * 100)
    right_point = int( scene_info[1]/2 * 100)

    if visualize:
        line_list = []

    cluster_angles = {'foc':[], 'loc':[]}
    for cluster in valid_list:
        center = np.median(cluster, axis=0)

        max_dist = 0
        max_line = None
        no_obj = True
        for point in np.arange(left_point, right_point + 1, int(scene_info[1] * 100) / 30):
            vec = ([point, 25] - center)

            is_collision = False
            dist_list = []
            for obj in curr_config:
                obj_pos = [-obj[1] * 100, obj[0] * 100]
                check_range = np.dot(obj_pos - np.array([point, 25]), -vec/np.linalg.norm(vec))
                if abs(check_range) > np.linalg.norm(vec):
                    continue
                no_obj = False

                obj_vec = obj_pos - center
                radius = obj[2] * 100
                dist = abs((obj_vec[0] * vec[1] - obj_vec[1] * vec[0]) / np.linalg.norm(vec))
                if dist <= radius + 1:
                    is_collision = True
                    break
                dist_list.append(dist)

            if not is_collision:
                if dist_list:
                    dist_to_wall = abs(left_point - point) if point <= 0 else abs(right_point - point)
                    min_dist = min(dist_list) if min(dist_list) < dist_to_wall else dist_to_wall
                else:
                    min_dist = abs(left_point - point) if point <= 0 else abs(right_point - point)
                    
                if min_dist > max_dist:
                    max_dist = min_dist
                    max_line = point

                if visualize:
                    line_list.append([[center[0], point], [center[1], 25]])

        if no_obj:
            print("no obj")
            max_line = 0

        cluster_angles["foc"].append(np.array([center[1], -center[0]]) / 100)
        cluster_angles["loc"].append(np.array([25, -max_line]) / 100)
        
    if visualize:
        plt.figure(figsize=(20,20))
        plt.axis([-43,43,0,86])
        for cluster in valid_list:
            plt.scatter(np.array(cluster)[:,0], np.array(cluster)[:,1], color='orange')
        for obj in curr_config:
            obj_pos = [-obj[1] * 100, obj[0] * 100]
            temp_circle = mpatches.Circle((obj_pos), radius, color = obj[3])
            plt.gca().add_patch(temp_circle)

        for line in line_list:
            plt.plot(line[0], line[1], marker='o', color='black')

        for i in range(len(cluster_angles['loc'])):
            loc = cluster_angles['loc'][i]
            foc = cluster_angles['foc'][i]
            plt.plot([-loc[1] * 100, -foc[1] * 100], [loc[0] * 100, foc[0] * 100], marker='o', color='red')

        plt.show()

    return cluster_angles

def scale_config(config):
    for pos in config:
        for i in range(3):
            pos[i] = pos[i] * 100
        temp = pos[0]
        pos[0] = -pos[1]
        pos[1] = temp

    return config
    
#*************************************************************************************************#


#initialize gym
#*************************************************************************************************#
gym = gymapi.acquire_gym()
#*************************************************************************************************#

#parse arguments

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
#with open(asset_root + "urdf/ycb/all_centroid_m_new.txt") as f:
#    for line in f:
#        div = line[:-1].split(" ")
#        div_f = [float(x) for x in div]
#        object_centroid_m.append([np.array([div_f[0], div_f[1], div_f[2]]), div_f[3]])


#object_asset_files = ["urdf/ycb/025_mug/025_mug.urdf",
#                      "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
#                      "urdf/ycb/006_mustard_bottle/textured.urdf"]
#object_collision_files = ["urdf/ycb/025_mug/mug_collision.obj",
#                          "urdf/ycb/010_potted_meat_can/collision.obj",
#                          "urdf/ycb/006_mustard_bottle/textured_vhacd.obj"]


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
#ik_solver = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
#seed_state = [0.0]*ik_solver.number_of_joints
#test_quat = gymapi.Quat()
#quat_result = test_quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi)
#print (quat_result)
#dof_result = ik_solver.get_ik(seed_state, 
#                              0.5, 0.5, 0.5,
#                              quat_result.x, 
#                              quat_result.y,
#                              quat_result.z,
#                              quat_result.w)
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
ur5e_pose.p = gymapi.Vec3(0, 0, 0)
ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5)

left_cover_pose = gymapi.Transform()
left_cover_pose.p = gymapi.Vec3(table_pose.p.x, table_dims.y*0.5 - 0.015, 
                                table_dims.z + side_cover_dims.z/2.0)

right_cover_pose = gymapi.Transform()
right_cover_pose.p = gymapi.Vec3(table_pose.p.x, -table_dims.y*0.5 + 0.015, 
                                 table_dims.z + side_cover_dims.z/2.0)

upper_cover_pose = gymapi.Transform()
upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + side_cover_dims.z + 0.015)

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

object_collision_models = [table_obj, left_cover_obj, right_cover_obj]

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
#print (location_candidates)
#print (len(location_candidates))

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
chosen_object = []
chosen_scale = []
object_normalize = []
# num_of_objects = np.random.randint(min_num_of_objects, max_num_of_objects+1)

observed_objects = []
gripper_location = None
object_status_list = []
object_reader_tracker = []
for i in range(num_of_envs):
    envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
    ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))
    #ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 1))

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

    # Choose target & obstacle objects-------------------------------------------------------------------------------
    num_of_objects = 4
    target_file_idx = 3

    # target_file_idx = np.random.choice([1, ])
    object_index = np.array([0] * (num_of_objects-1) + [target_file_idx])

    chosen_object.append(object_index)
    #object_index = [0]
    # object_loc = np.random.normal(0, table_dims.x/10.0, size=(2, num_of_objects))

    object_handles = []

    with open("object_name.txt", 'a') as f:
        for k in range(num_of_objects):
            f.write(object_asset_files[object_index[k]])
 
    # object_loc = []
    # for depth in range(len(location_candidates)-2, -1, -1):
    #     max_loc_count = len(location_candidates[depth])
    #     chosen_loc = np.random.choice(max_loc_count, int(max_loc_count*np.random.uniform(0.5, 0.8)), replace = False)
    #     for element in chosen_loc: object_loc.append(location_candidates[depth][element])

    # object_index = np.random.randint(len(object_asset_files) - 1, size = len(object_loc)-1)
    # object_index = np.insert(object_index, 0, len(object_asset_files)-1, axis = 0)
    # num_of_objects = len(object_loc)
        

    object_scaling_factor = np.random.randint(0, max_scaling_factor+1, size = num_of_objects)/10.0 + 1.0

    #object_loc_index = np.random.choice(len(location_candidates), num_of_objects, replace = False)
    #object_loc_index = [14, 11, 2, 13, 1]
        

    # set up objects///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # creating manager
    objs_manager = fcl.DynamicAABBTreeCollisionManager()
    objs_manager.setup()
    obstacle_objs = []
    obj_pos_list = []
    target_pos = [0.5, 0, table_dims.z + 0.08]

    for k in range(num_of_objects):
        object_pose = gymapi.Transform()
        is_collision = True

        # add target obj
        if k == num_of_objects - 1:
            object_pose.p = gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2])
            # object_pose.p = gymapi.Vec3(0.5, 0, 0)

            file_path = object_collision_files[object_index[-1]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[-1])
            collision_mesh.add_offset(object_offset[object_index[-1]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()

            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform(np.array(target_pos))
            is_collision = False

        # random selec obj location
        while is_collision:
            tx = np.random.uniform(0.35, table_dims.x + 0.2)
            ty = np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2)
            tz = table_dims.z + 0.08

            object_pose.p = gymapi.Vec3(tx, ty, tz)

            file_path = object_collision_files[object_index[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[object_index[k]])
            
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()

            # new obj
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform(np.array([tx,ty,tz]))
            

            # check collision
            req = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request = req)
            objs_manager.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)

            is_collision = rdata.result.is_collision # update collision status

            if not is_collision:
                    dist = np.sqrt((tx - target_pos[0])**2 + (ty - target_pos[1])**2)
                    if dist <= 0.16:
                        is_collision = True
                        print("target contact recalc")
                        continue

                    for obj in obj_pos_list:
                        dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                        # print("idx:", i, "dist:", dist)
                        if dist <= 0.16:
                            is_collision = True
                            print("recalc")
                            continue


        # if k == 0: gripper_location = gymapi.Vec3(tx, ty, tz + 0.2)
        #object_pose.p = get_random_loc(0.3 + table_dims.x*0.2, 0.3 + table_dims.x*0.8,
        #                               -table_dims.y*0.4, table_dims.y*0.4,
        #                               table_dims.z, table_dims.z + drawer_height*0.5)
        # pdb.set_trace()
        # offset = object_offset[object_index[k]]
        # obj_pos = [object_pose.p.x + object_offset[object_index[k]][0], object_pose.p.y]
        obj_pos_list.append([object_pose.p.x, object_pose.p.y])

        object_handles.append(gym.create_actor(envs[-1], 
                                               object_assets[object_index[k]], 
                                               object_pose, 
                                               "object" + str(k) + str(i), 0, 2**(k+1), k+1))
        gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
        #object_normalize.append(object_centroid_m[object_index[k]])
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

        #target_pos = gym.get_camera_transform(sim, envs[-1], local_camera_handles[-1]).p
        #target_quat = gym.get_camera_transform(sim, envs[-1], local_camera_handles[-1]).r
        #converted_coord = global_coord_converter(viewpoint_candidate.x - cam_offset_x,
        #                                         viewpoint_candidate.y,
        #                                         viewpoint_candidate.z - cam_offset_z, 0, 0, 0)
        #converted_quat = quaternion_multiply(gymapi.Quat(-0.707, 0, 0, 0.707), target_quat)

        #ik_solver = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
        #seed_state = [0.0]*ik_solver.number_of_joints
        #dof_result = ik_solver.get_ik(seed_state, 
        #                              converted_coord[0],
        #                              converted_coord[1],
        #                              converted_coord[2],
        #                              converted_quat.x, 
        #                              converted_quat.y,
        #                              converted_quat.z,
        #                              converted_quat.w)


        #if dof_result:
        #    #gym.set_dof_target_position(envs[-1], spj, dof_result[0]) 
        #    #gym.set_dof_target_position(envs[-1], slj, dof_result[1]) 
        #    #gym.set_dof_target_position(envs[-1], ej,  dof_result[2]) 
        #    #gym.set_dof_target_position(envs[-1], wj1, dof_result[3]) 
        #    #gym.set_dof_target_position(envs[-1], wj2, dof_result[4]) 
        #    #gym.set_dof_target_position(envs[-1], wj3, dof_result[5])
        #    #final_ori = gym.get_joint_transform(envs[-1], wj3)
        #    #test_ori = gym.get_rigid_transform(envs[-1], cam_link)
        #    #res = gym.get_actor_rigid_body_states(envs[-1], ur5e_handles[-1], 1)
        #    print (converted_coord, converted_quat)
        #    print (dof_result)
        #    #print (test_ori.p, test_ori.r)
        #    print ("found solution")
        ##else:
        ##    gym.set_dof_target_position(envs[-1], ej, -math.pi * 0.5)
        ##    #gym.set_dof_target_position(envs[-1], wj1, 0.966)
        ##    #gym.set_dof_target_position(envs[-1], wj2, math.pi * 0.5)


#*************************************************************************************************#

#class ur5e_valid(ob.StateValidityChecker):
#    def __init__(self, si, real_offset):
#        super().__init__(si)
#        self.real_offset_ = real_offset
#
#    def isValid(self, dof_state):
#        pose_array = get_pose_from_dof(dof_state)
#
#        ur5e_self_col = []
#        rots = []
#        trans = []
#
#        #print (dof_state[0], dof_state[1], dof_state[2], dof_state[3], dof_state[4], dof_state[5])
#        #real_offset = np.array(state_tensor[0][:3])
#        for t in range(8):
#            rotation = np.array(pose_array[t][1])
#            translation = np.array(pose_array[t][0] + self.real_offset_)
#            rots.append(rotation)
#            trans.append(translation)
#            r1 = R.from_quat(rotation)
#            tf = fcl.Transform(r1.as_matrix(), translation)
#            ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))
#
#
#        request = fcl.CollisionRequest()
#        result = fcl.CollisionResult()
#        self_collision_flag = False
#
#        for t in range(8):
#            if t != 0:
#                if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
#                    self_collision_flag = True
#                    break
#            col_with_other_part = False
#            for q in range(8):
#                if q < t-1 or q > t + 1:
#                    if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
#                        col_with_other_part = True
#                        break
#            if col_with_other_part:
#                self_collision_flag = True
#                break
#
#        env_collision_flag = False
#        manager1 = fcl.DynamicAABBTreeCollisionManager()
#        manager1.registerObjects(ur5e_self_col)
#        manager1.setup()
#
#        manager2 = fcl.DynamicAABBTreeCollisionManager()
#        manager2.registerObjects(object_collision_models + flexible_collision_models)
#        manager2.setup()
#
#        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
#        rdata = fcl.CollisionData(request = req)
#        manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
#        if rdata.result.is_collision:
#            env_collision_flag = True
#
#        return self_collision_flag == False and env_collision_flag == False
#
#
#def ur5e_in_collision(dof_result, real_offset):
#
#    pose_array = get_pose_from_dof(dof_result)
#
#    ur5e_self_col = []
#    rots = []
#    trans = []
#    #real_offset = np.array(state_tensor[0][:3])
#    for t in range(8):
#        rotation = np.array(pose_array[t][1])
#        translation = np.array(pose_array[t][0] + real_offset)
#        rots.append(rotation)
#        trans.append(translation)
#        r1 = R.from_quat(rotation)
#        tf = fcl.Transform(r1.as_matrix(), translation)
#        ur5e_self_col.append(fcl.CollisionObject(ur5e_collision_models[t], tf))
#
#
#    request = fcl.CollisionRequest()
#    result = fcl.CollisionResult()
#    self_collision_flag = False
#
#    for t in range(8):
#        if t != 0:
#            if fcl.collide(ur5e_self_col[t], plane_obj, request, result):
#                self_collision_flag = True
#                break
#        col_with_other_part = False
#        for q in range(8):
#            if q < t-1 or q > t + 1:
#                if fcl.collide(ur5e_self_col[t], ur5e_self_col[q], request, result):
#                    col_with_other_part = True
#                    break
#        if col_with_other_part:
#            self_collision_flag = True
#            break
#
#    env_collision_flag = False
#    manager1 = fcl.DynamicAABBTreeCollisionManager()
#    manager1.registerObjects(ur5e_self_col)
#    manager1.setup()
#
#    manager2 = fcl.DynamicAABBTreeCollisionManager()
#    manager2.registerObjects(object_collision_models + flexible_collision_models)
#    manager2.setup()
#
#    req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact = True)
#    rdata = fcl.CollisionData(request = req)
#    manager1.collide(manager2, rdata, fcl.defaultCollisionCallback)
#    if rdata.result.is_collision:
#        env_collision_flag = True
#
#    return self_collision_flag == True or env_collision_flag == True


#*************************************************************************************************#
cam_pos = gymapi.Vec3(2.2, 0, 0.5)
cam_target = gymapi.Vec3(0, 0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                 gymapi.Vec3(-1.0, 0.0, 0.0))
gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                 gymapi.Vec3(1.0, 0.0, 0.0))
#*************************************************************************************************#

#empty loop here to test functionalities
#*************************************************************************************************#
real_position = False
flex_collision_models = []
object_mesh = []
t = 0
#while not gym.query_viewer_has_closed(viewer):
for t in range(2000):
    if not real_position:
        gym.set_dof_target_position(envs[-1], spj, 0.7)
        gym.set_dof_target_position(envs[-1], slj, -2)
        gym.set_dof_target_position(envs[-1], ej,  2.5)
        gym.set_dof_target_position(envs[-1], wj1, -0.3)
        gym.set_dof_target_position(envs[-1], wj2, 0.7)
        gym.set_dof_target_position(envs[-1], wj3, 0)

        # # check grasp
        # gym.set_dof_target_position(envs[-1], spj, init2grasp_path[-1][0])
        # gym.set_dof_target_position(envs[-1], slj, init2grasp_path[-1][1])
        # gym.set_dof_target_position(envs[-1], ej,  init2grasp_path[-1][2])
        # gym.set_dof_target_position(envs[-1], wj1, init2grasp_path[-1][3])
        # gym.set_dof_target_position(envs[-1], wj2, init2grasp_path[-1][4])
        # gym.set_dof_target_position(envs[-1], wj3, init2grasp_path[-1][5])

        # reset motion
        # gym.set_dof_target_position(envs[-1], spj, 0)
        # gym.set_dof_target_position(envs[-1], slj, -math.pi/2)
        # gym.set_dof_target_position(envs[-1], ej,  0)
        # gym.set_dof_target_position(envs[-1], wj1, -math.pi/2)
        # gym.set_dof_target_position(envs[-1], wj2, 0.7)
        # gym.set_dof_target_position(envs[-1], wj3, 0)

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
            #gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])



    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)
#*************************************************************************************************#

ik_solver2 = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
# target_pos = gripper_location
# print (target_pos)
# converted_coord = global_coord_converter(target_pos.x ,
#                                          target_pos.y ,
#                                          target_pos.z , 
#                                          ur5e_pose.p.x, 
#                                          ur5e_pose.p.y, 
#                                          ur5e_pose.p.z)

target_quat = gymapi.Quat(0.446, 0.560, -0.433, 0.549)
converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)
file_path = '../assets/urdf/ur5e/meshes/collision/'

# data = np.load("test_data/MCTS_input/ur5_test.npy", allow_pickle=True)
# init2grasp_path = data[0]["init2grasp_path"]
# grasp2init_path = data[0]["grasp2init_path"]
# w_target = data[0]["w_target"]
# target_pos = data[0]["target_pos"]
# target_mesh = data[0]["target_mesh"]
scene_info = [table_dims.x, table_dims.y, table_dims.z, 0.5]
# print("SCENE INFO", scene_info)

# rac = RC.robot_arm_configuration('../assets/urdf/ur5e/meshes/collision/', np.array([0.0, 0, 0]), scene_info, target_mesh=target_mesh, obstacles_num=0, target_pos=target_pos) # point_cloud=point_cloud
# swept_volume1, swept_verts1 = rac.get_swept_volume(init2grasp_path, None, 0, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False, with_scene=True)
# swept_volume2, swept_verts2 = rac.get_swept_volume(grasp2init_path, None, 0, w_target=w_target, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False, with_scene=True)
# swept_center, swept_verts = rac.get_swept_center(swept_verts1+swept_verts2, scene_info)

obstacle_files = asset_root + "urdf/ycb/002_master_chef_can/textured_vhacd.obj"
rac = RC.robot_arm_configuration(file_path, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info)


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
scene = global_scene(1.0, 1.2, 0.85, np.array([0.3, -0.6, 0.05]), table_dims.x, table_dims.y - 0.04, drawer_height, table_dims.z - 0.05)

saved_scene_vertices = None
saved_scene_faces = None

# no active sensing part
# while not gym.query_viewer_has_closed(viewer):
#      # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
    
#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
    
#     gym.sync_frame_time(sim)

# creating new folder
curr_time = time.localtime()
new_folder = 'test_data/test_scenes/' + str(curr_time[1]) + '.' + str(curr_time[2]) + '.' + str(curr_time[3]) + '.' + str(curr_time[4]) + '/'
os.makedirs(new_folder + 'test_image/')
os.makedirs(new_folder + 'test_seg_image/')
os.makedirs(new_folder + 'test_depth_image/')
os.makedirs(new_folder + 'test_npy/')

# init MCTS
ML_MCTS_ins = mct.multi_level_MCTS_algo(None, None, scene_info=scene_info, swept_volume1=None, swept_volume2=None, obj_mesh=rac.obj_mesh)

#active sensing here
obj_pos_MCTS = {}
obj_mesh_MCTS = {}
obj_pcd = {}
is_tunnel_covered = False
is_target_detected = False
target_template = None
while not gym.query_viewer_has_closed(viewer):#///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if is_target_detected and is_cluster_covered:
        is_plan_success = ML_MCTS_ins.run_mcts()
        break
    if need_acquire:
        if acquire_counter > 500:
            gym.clear_lines(viewer)
            gym.render_all_camera_sensors(sim)
            for q in body_cam_handles:
                color_image = gym.get_camera_image(sim, envs[-1], 
                                                                                     body_cam_handles[q], 
                                                                                     gymapi.IMAGE_COLOR)
                depth_image = gym.get_camera_image(sim, envs[-1], 
                                                   body_cam_handles[q], 
                                                                                     gymapi.IMAGE_DEPTH)
                seg_image = gym.get_camera_image(sim, envs[-1],
                                                                                 body_cam_handles[q],
                                                                                 gymapi.IMAGE_SEGMENTATION)
                if (q == 0):
                    # view_matrix = np.matrix(gym.get_camera_view_matrix(sim, envs[-1], body_cam_handles[q]))
    #                 k = np.array([[912.72143555,   0.        , 649.00366211],
    #    [  0.        , 912.7409668 , 363.25247192],
    #    [  0.        ,   0.        ,   1.        ]])
                    K = np.array([[911.445649104, 0, 641.169],
                         [0, 891.51236121, 352.77],
                         [0, 0, 1]])
                    # print("view_matrix:", view_matrix)
                    # print("View_matrix size:", view_matrix.shape)

                    color_img_saved = write_to_image(color_image, new_folder + 'test_image/' + str(sequence_count) + '.png')
                    write_to_seg_image(seg_image, new_folder + 'test_seg_image/' + str(sequence_count) + '.png')
                    write_to_depth_image(depth_image, new_folder + 'test_depth_image/' + str(sequence_count) + '.png')
       
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
                    
                    write_for_contact_grasp(color_img_saved, seg_image, -depth_image, K, new_cam_rotation, new_cam_translation, new_folder + 'test_npy/' + str(sequence_count)+'.npy')

                    dist1 = np.linalg.norm(final_rotation - new_cam_rotation)
                    dist3 = np.linalg.norm(final_rotation - new_cam_rotation*-1)
                    dist2 = np.linalg.norm(final_translation - new_cam_translation)

                    if (dist1 < 1e-2 or dist3 < 1e-2) and dist2 < 1e-2:
                        print ('start pc extraction')
                        pc_extractor_grasp(new_rgb_image, new_depth_image, new_seg_image, new_cam_rotation, new_cam_translation, object_dict, table_dims.z)

                        # print("obj", object_dict)
                        # plt.imshow(new_rgb_image); plt.show()
                        # plt.imshow(new_seg_image); plt.show()
                        
                        for i in object_dict:
                            mask = new_seg_image == 1
                            temp_seg_image = copy.deepcopy(new_seg_image)
                            mask = new_seg_image == i
                            temp_seg_image[~mask] = 0
                            temp_seg_image[mask] = 1

                            point_cloud, pcd = RC.write_to_pointcloud(new_rgb_image, new_depth_image, temp_seg_image, new_cam_rotation, new_cam_translation, visualization=False)
                            if i in obj_pcd.keys():
                                obj_pcd[i] += pcd
                            else:
                                obj_pcd[i] = pcd

                            # find matching object mesh file
                            downpcd = obj_pcd[i].voxel_down_sample(voxel_size=0.005) # downsampe pcd

                            if i != num_of_objects:
                                # obstacle object matching
                                mesh = o3d.io.read_triangle_mesh(obstacle_files)
                                mesh.compute_vertex_normals()
                                source_pcd = mesh.sample_points_uniformly(number_of_points=20000)
                                dist, obj_trans = RC.pcd_matching(downpcd, source_pcd, False)

                                # calc inverse transform 
                                inv_rot = obj_trans[:3,:3].T
                                inv_trans = -inv_rot @ obj_trans[:3, 3]
                                inv_rot = R.from_matrix(inv_rot.copy())

                                # make object model
                                verts_no_rotations = np.asarray(mesh.vertices)
                                face = np.asarray(mesh.triangles)
                                verts = inv_rot.apply(verts_no_rotations) + inv_trans
                                obj_mesh = [verts, face]

                                obj_pos = inv_trans - object_offset[object_index[i-1]]
                            else:
                                # target obj matching
                                obj_mesh, obj_pos, dist, obj_name = RC.get_matching_mesh(downpcd, visualize=False)
                                obj_pos = obj_pos - object_offset[object_index[i-1]]

                                if not is_target_detected:
                                    # read grasp data
                                    grasp_file = "/".join(obj_name.split('/')[:5]) + "/grasp_dict.npy"
                                    grasp_data = np.load(grasp_file, allow_pickle=True)

                                    # generate swept volume
                                    num_grasp = 0
                                    swept_size = sys.maxsize
                                    grasp_list = np.arange(len(grasp_data))
                                    np.random.shuffle(np.arange(len(grasp_list)))

                                    # for grasp_idx in np.random.randint(len(grasp_data), size=10):
                                    for grasp_idx in grasp_list:
                                        target_pos = grasp_data[grasp_idx]['target_pos']
                                        target_quat = grasp_data[grasp_idx]['target_quat']
                                        target_pos[:2] = target_pos[:2] + obj_pos[:2]

                                        init2grasp_angels = rac.grasp_verify(target_pos, target_quat)
                                        grasp2init_angels = rac.grasp_verify(target_pos + [0,0,0.01], target_quat)

                                        if init2grasp_angels is None or grasp2init_angels is None:
                                            print("skip imposible grasp")
                                            continue

                                        # rac.check_collision_models(grasp2init_angels, scene_info=scene_info)

                                        init2grasp_path_temp = RC.get_path2grasp(rac, init2grasp_angels, scene_info, target_mesh=obj_mesh)
                                        mod_bbox = rac.modify_grasp_bbox(init2grasp_angels, obj_mesh, visualize=False)
                                        grasp2init_path_temp = RC.get_path2start(rac, grasp2init_angels, mod_bbox, scene_info)

                                        if init2grasp_path_temp is None or grasp2init_path_temp is None:
                                            print("No path generated")
                                            continue

                                        swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(init2grasp_path_temp, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                                        swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(grasp2init_path_temp, w_target=mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=True)
                                        num_grasp += 1
                                        is_target_detected = True
                                        print("!!!!!!!!!!!!!!!!!!!TRUE", num_grasp, '!!!!!!!!!!!!!!!!!!!!!!!')


                                        # compare swept volumes
                                        swept_center_temp, swept_verts_temp = rac.get_swept_center(swept_verts1_temp+swept_verts2_temp, scene_info)
                                        if len(swept_verts_temp) < swept_size:
                                            swept_size = len(swept_verts_temp)
                                            ML_MCTS_ins.swept_volume1 = swept_volume1_temp
                                            ML_MCTS_ins.swept_volume2 = swept_volume2_temp
                                            swept_center = swept_center_temp
                                            swept_verts = swept_verts_temp

                                        if num_grasp == 5:
                                            break

                            # # obj matching
                            # obj_mesh, obj_pos, dist = RC.get_matching_mesh(downpcd, visualize=False)
                            # obj_pos = obj_pos - object_offset[object_index[i-1]]
                                
                            if dist < 3:
                                print("idx",i,"mesh added")
                                obj_pos_MCTS[i] = obj_pos[0:2].tolist()
                                obj_mesh_MCTS[i] = obj_mesh

                                dy = obj_pos_MCTS[i][1] - obj_pos_list[i-1][1]
                                dx = obj_pos_MCTS[i][0] - obj_pos_list[i-1][0]
                                print("diff:", np.sqrt(dy**2 + dx**2), "dist", dist)

                        _ = scene.register_camera_view(list(new_cam_rotation), list(new_cam_translation), new_depth_image, object_dict)

                        if is_target_detected:
                            coverage_score, _ = swept_coverage_check(scene, swept_verts, rac, scene_info)
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!score", coverage_score)

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

        print ('finish adding flexible collision model')

        end_state_collision_free = False
        while not end_state_collision_free:
            # camera_loc, camera_focus, dof_result = random_sample_guided_selection(sim, envs[-1], test_cam, scene)
            # if coverage_score >=0.96 and len(obj_pos_MCTS) == num_of_objects:
            if coverage_score >=0.96: # swept volume observed
                # get obj pose
                obj_mesh_MCTS = dict(sorted(obj_mesh_MCTS.items()))
                obj_pos_MCTS = dict(sorted(obj_pos_MCTS.items()))

                rac.target_mesh = obj_mesh_MCTS[4]
                target_pos_MCT = obj_pos_MCTS[4]
                rac.obstacles_num = num_of_objects - 1
                rac.obj_mesh = list(obj_mesh_MCTS.values())[:num_of_objects - 1]
                rac.obj_pos_list = list(obj_pos_MCTS.values())[:num_of_objects - 1]

                # unobserved area processing
                curr_config, target_pos_MCT = rac.get_MCT_config(copy.deepcopy(rac.obj_pos_list), copy.deepcopy(rac.obj_mesh), copy.deepcopy(target_pos_MCT), copy.deepcopy(rac.target_mesh))
                unknown_area = get_unobserved_area(scene)

                np.save(new_folder + "unknown_area" + str(sequence_count), unknown_area)

                unknown_area = delete_obj_spots(curr_config, target_pos_MCT, unknown_area, visualize=False)
                cluster_list = clustering(unknown_area, visualize=False)
                filtered_cluster, valid_area, potential_centers = get_filtered_clusters(cluster_list, visualize=False)

                ML_MCTS_ins.curr_config_ = copy.deepcopy(curr_config)
                ML_MCTS_ins.goal_config_ = copy.deepcopy(curr_config)
                ML_MCTS_ins.target_pos = copy.deepcopy(target_pos_MCT)

                ML_MCTS_ins.unknown_area = unknown_area
                ML_MCTS_ins.valid_area = valid_area
                ML_MCTS_ins.potential_centers = potential_centers

                ML_MCTS_ins.obj_mesh = rac.obj_mesh

                ML_MCTS_ins.init_MCTS()
                ML_MCTS_ins.scenario_check() # terminate scenario if it's not valid

                if not is_tunnel_covered:
                    print("covering grasp tunnel")
                    obj_idx, centers = ML_MCTS_ins.unknown_tunnel_check()
                    print("centers", centers)
                    # pdb.set_trace()

                    if not centers:
                        print("Tunnel Covered")
                        is_tunnel_covered = True
                        need_acquire = True
                        break


                    plt.figure(figsize=(20,20))
                    plt.axis([-43,43,0,86])
                    tunnel_color = ['b', 'r']
                    tunnel_counter_ = 0
                    for j, obj_i in enumerate(obj_idx):
                        cx, cy, radius, color = ML_MCTS_ins.MCTS_ins.MCTS_tree_.curr_config_[obj_i]
                        temp_circle = mpatches.Circle((cx, cy), radius, color = color)
                        plt.gca().add_patch(temp_circle)

                        tunnel = ML_MCTS_ins.MCTS_ins.MCTS_tree_.get_tunnel(ML_MCTS_ins.MCTS_ins.MCTS_tree_.robot_, ML_MCTS_ins.MCTS_ins.MCTS_tree_.curr_config_[obj_i][:2])
                        start_corner, width, height, angle, v2_start, v2_end, v3_start, v3_end = tunnel
                        tunnel_shape = mpatches.Rectangle(start_corner, width, height, angle, alpha = 0.5, color = tunnel_color[tunnel_counter_])
                        plt.gca().add_patch(tunnel_shape)
                        plt.plot([v2_start[0], v2_end[0]], [v2_start[1], v2_end[1]], color = 'r')
                        plt.plot([v3_start[0], v3_end[0]], [v3_start[1], v3_end[1]], color = 'r')
                        tunnel_counter_ += 1
                    plt.scatter(np.array(centers)[:,0], np.array(centers)[:,1], color='black')
                    plt.show()
                    # pdb.set_trace()
                    
                    cluster_angles = cal_cam_angle_for_area(centers, curr_config + [target_pos_MCT], scene_info, visualize=False)

                else:
                    print("Checking other unknown areas")
                    cluster_list = clustering(unknown_area, visualize=False)
                    filtered_cluster, valid_points, potential_centers = get_filtered_clusters(cluster_list, visualize=False)

                    if len(filtered_cluster) == 0:
                        print("No unknown areas!!!!!!")
                        is_cluster_covered = True
                        need_acquire = True
                        break

                    # get possilbe cam pose
                    cluster_angles = cal_cam_angle_for_area(potential_centers, curr_config + [target_pos_MCT], scene_info, visualize=True)
                    
                # choosing biggest cluster 
                end_point = cluster_angles['loc'][0]
                focus_point = cluster_angles['foc'][0]
                camera_loc, camera_focus, dof_result = cam_loc_selection_for_clusters(sim, envs[-1], test_cam, scene, focus_point, end_point, scene_info)
                    

            # elif sequence_count > 1:
            elif not is_target_detected: # target is not detected, normal active sensing
                camera_loc, camera_focus, dof_result = random_sample_guided_selection(sim, envs[-1], test_cam, scene)
            else: # target is detected, covering swept volume
                if swept_center is not None:
                    focus_point = swept_center
                    swept_center = None
                else:
                    _, focus_point = swept_coverage_check(scene, swept_verts, rac, scene_info)
                camera_loc, camera_focus, dof_result = random_sample_swept_volume_selection(sim, envs[-1], test_cam, scene, focus_point)

            print (camera_loc, camera_focus)
            gym.set_camera_location(test_cam, envs[-1], camera_loc, camera_focus)
            target_pos = gym.get_camera_transform(sim, envs[-1], test_cam).p
            target_quat = gym.get_camera_transform(sim, envs[-1], test_cam).r

            #r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
            #cam_offset_vector = np.array([0.11, 0, 0.08])
            #rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
            #converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
            #                                                                                 target_pos.y - rot_cam_offset_vector[1],
            #                                                                                 target_pos.z - rot_cam_offset_vector[2],
            #                                                                                 ur5e_pose.p.x,
            #                                                                                 ur5e_pose.p.y,
            #                                                                                 ur5e_pose.p.z)
            #converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

            #seed_state = [0.0]*ik_solver2.number_of_joints

            #dof_result = ik_solver2.get_ik(seed_state,
            #                                                             converted_coord[0],
            #                                                             converted_coord[1],
            #                                                             converted_coord[2],
            #                                                             converted_quat.x,
            #                                                             converted_quat.y,
            #                                                             converted_quat.z,
            #                                                             converted_quat.w)

            print (dof_result)
            if dof_result:
                end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)
                print (end_state_collision_free)

                if end_state_collision_free:

                    print ('enters here')

                    final_translation = np.array([target_pos.x, target_pos.y, target_pos.z])
                    final_rotation = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
                    gym.set_dof_target_position(envs[-1], spj, dof_result[0])
                    gym.set_dof_target_position(envs[-1], slj, dof_result[1])
                    gym.set_dof_target_position(envs[-1], ej,  dof_result[2])
                    gym.set_dof_target_position(envs[-1], wj1, dof_result[3])
                    gym.set_dof_target_position(envs[-1], wj2, dof_result[4])
                    gym.set_dof_target_position(envs[-1], wj3, dof_result[5])

                    need_acquire = True
  
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    gym.sync_frame_time(sim)

# MCTS/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
rac.target_mesh = obj_mesh_MCTS.pop(num_of_objects)
target_pos_MCT = obj_pos_MCTS.pop(num_of_objects)
rac.obstacles_num = num_of_objects - 1
rac.obj_mesh = list(obj_mesh_MCTS.values())
rac.obj_pos_list = list(obj_pos_MCTS.values())
# rac.check_collision_models(init2grasp_path[-1], scene_info=scene_info)

gt_save_info = {"idx" : 0,
             "init2grasp_path" : init2grasp_path,
             "grasp2init_path" : grasp2init_path,
             "obj_pos_list" : obj_pos_list[:num_of_objects-1],
             "obj_mesh" : rac.obj_mesh,
             "scene_info" : scene_info,
             "w_target" : None,
             "test_name" : None,
             "target_mesh" : rac.target_mesh,
             "obstacles_num" : rac.obstacles_num,
             "target_pos" : obj_pos_list[-1]}


save_info = {"idx" : 0,
             "init2grasp_path" : init2grasp_path,
             "grasp2init_path" : grasp2init_path,
             "obj_pos_list" : rac.obj_pos_list,
             "obj_mesh" : rac.obj_mesh,
             "scene_info" : scene_info,
             "w_target" : None,
             "test_name" : None,
             "target_mesh" : rac.target_mesh,
             "obstacles_num" : rac.obstacles_num,
             "target_pos" : target_pos_MCT}

comp = np.array([save_info])
name = new_folder + "temp_scene"
np.save(name, comp)

comp = np.array([gt_save_info])
name = new_folder + "groud_truth_scene"
np.save(name, comp)

unknown_area = get_unobserved_area(scene)
np.save(new_folder + "unknown_area", unknown_area)

sys.exit(1)

cluster_list = clustering(unknown_area)
curr_config, target_pos_MCT = rac.get_MCT_config(rac.obj_pos_list, rac.obj_mesh, target_pos_MCT, rac.target_mesh)
pdb.set_trace()
ML_MCTS_ins = mct.multi_level_MCTS_algo(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info, swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh, target_pos=target_pos_MCT, unknown_area=unknown_area)


curr_config, target_pos_MCT = rac.get_MCT_config(obj_pos_list[:num_of_objects-1], rac.obj_mesh, obj_pos_list[-1], rac.target_mesh)
ML_MCTS_ins = mct.multi_level_MCTS_algo(copy.deepcopy(curr_config), copy.deepcopy(curr_config), scene_info=scene_info, swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=rac.obj_mesh, target_pos=target_pos_MCT, unknown_area=unknown_area)

ML_MCTS_ins.global_optimization()
pdb.set_trace()
ML_MCTS_ins.animate_whole_sequence()



   
saved_region_name = './saved_as_result_narrow/env_' + str(env_id) + '_region_info.npy'
region_data = []

for depth in range(len(region_candidates)):
    for can in region_candidates[depth]:
        can_vis = int(scene.is_visible(can[0], can[1], can[2]))
        region_data.append([can[0], can[1], can[2], can_vis])

np.save(saved_region_name, np.array(region_data))


saved_obj_name = './saved_as_result/env_' + str(env_id) + '_object_info.npy'
object_data = np.array([]).reshape(0, 8, 3)

#visualize all the detected objects
draw_flag = False
if not draw_flag:
    for key,value in object_dict.items():
        object_handler = value
        vertices, faces = object_handler.get_collision_mesh()
        print(vertices.shape)
        object_data = np.concatenate((object_data, vertices.reshape(1, 8, 3)), axis = 0)

np.save(saved_obj_name, object_data)

saved_scene_vertices_name = './saved_as_result_narrow/env_' + str(env_id) + '_scene_vertices_info.npy'
saved_scene_faces_name = './saved_as_result_narrow/env_' + str(env_id) + '_scene_faces_info.npy'

#np.save(saved_scene_vertices_name, saved_scene_vertices)
#np.save(saved_scene_faces_name, saved_scene_faces)

sys.exit(1)
#test_cam = gym.create_camera_sensor(envs[-1], camera_props)
#while not gym.query_viewer_has_closed(viewer):
#    camera_loc = get_random_loc(0.2, 0.3 + table_dims.x,
#                                          -table_dims.y * 0.5, table_dims.y * 0.5,
#                                          table_dims.z,  table_dims.z + drawer_height)
#    camera_focus = get_random_loc(camera_loc.x, table_dims.x + 0.3,
#                                            -table_dims.y*0.5, table_dims.y*0.5,
#                                            table_dims.z, camera_loc.z)
#
#    gym.set_camera_location(test_cam, envs[-1], 
#                                                                        camera_loc, 
#                                                                        camera_focus)
#    target_pos = gym.get_camera_transform(sim, envs[-1], test_cam).p
#    target_quat = gym.get_camera_transform(sim, envs[-1], test_cam).r
#    gripper_offset_vector = np.array([0.25, 0, 0])
#    r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
#    rot_gripper_offset_vector = r_rot.apply(gripper_offset_vector)
#    converted_coord = global_coord_converter(target_pos.x - rot_gripper_offset_vector[0],
#                                                       target_pos.y - rot_gripper_offset_vector[1],
#                                                       target_pos.z - rot_gripper_offset_vector[2], 
#                                                       ur5e_pose.p.x, 
#                                                       ur5e_pose.p.y, 
#                                                       ur5e_pose.p.z)
#                        
#    converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)
#                        
#    seed_state = [0.0]*ik_solver2.number_of_joints
#    dof_result = None
#    dof_result = ik_solver2.get_ik(seed_state,
#                                     converted_coord[0],
#                                     converted_coord[1],
#                                     converted_coord[2],
#                                     converted_quat.x, 
#                                     converted_quat.y,
#                                     converted_quat.z,
#                                                 converted_quat.w)
#    if dof_result and rac.arm_collision_free(dof_result, plane_obj, object_collision_models, [], 0): 
#        print(dof_result)

search_space_low = [-0.48 - 1.26, -1.19 - 0.98, 1.67 - 1.8, -3.14, -0.25 - 1.54, -3.14]
search_space_high = [-0.48 + 1.26, -1.19 + 0.98, 1.67 + 1.8, 3.14, -0.25 + 1.54, 3.14]
while not potential_result and counter < 10000:
    counter += 1
    dof_result = ik_solver2.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat.x, 
                                       converted_quat.y,
                                       converted_quat.z,
                                                   converted_quat.w)
    if dof_result and rac.arm_collision_free(dof_result, plane_obj, object_collision_models, [], 0): 
        flag = True
        #print (dof_result)
        for k in range(6):
            if search_space_low[k] <= dof_result[k] <= search_space_high[k]:
                pass
            else:
                #print (dof_result[k], k)
                flag = False
                break
        if flag:
            potential_result = dof_result
            for t in range(100):
                gym.set_dof_target_position(envs[-1], spj, potential_result[0])
                gym.set_dof_target_position(envs[-1], slj, potential_result[1])
                gym.set_dof_target_position(envs[-1], ej,  potential_result[2])
                gym.set_dof_target_position(envs[-1], wj1, potential_result[3])
                gym.set_dof_target_position(envs[-1], wj2, potential_result[4])
                gym.set_dof_target_position(envs[-1], wj3, potential_result[5])
    
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
            break

if not potential_result:
    sys.exit(1)


#try feed forward module
robot_data = [ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]
gripper_data = [gripper_location.x, gripper_location.y, gripper_location.z, 0.446, 0.560, -0.433, 0.549]
object_data = []
for t in range(len(object_handles)):
    ox, oy, oz = object_status_list[t][0]
    dx, dy, dz = object_status_list[t][1]
    object_data.append([ox, oy, oz, dx, dy, dz])
object_choice = feed_forward(robot_data, gripper_data, object_data)
if object_choice < len(object_handles) - 1:
    print (object_status_list[object_choice+1])
else:
    print ('invalid choice')

#while not gym.query_viewer_has_closed(viewer):
#    element = object_handles[object_choice + 1]
#    states = gym.get_actor_rigid_body_states(envs[-1], element, 1)
#    rotation = np.array(states[0][0][1])
#    translation = np.array(states[0][0][0])
#    rotation = np.array(rotation.item())
#    translation = np.array(translation.item())
#    #print (rotation, translation)
#    #object_status_list[i][0] += translation
#    r1 = R.from_quat(rotation)
#    tf = fcl.Transform(r1.as_matrix(), translation)
#    flex_collision_models.append([fcl.CollisionObject(object_collision_lib[i], tf), 0])
#
#    temp_obj = object_reader_tracker[object_choice + 1]
#    #temp_obj.set_offset(translation)
#    all_lines = []
#    vertices = temp_obj.get_vertices()
#    for v1, v2, v3 in temp_obj.get_faces():
#        all_lines += list(vertices[v1])
#        all_lines += list(vertices[v2])
#        all_lines += list(vertices[v1])
#        all_lines += list(vertices[v3])
#        all_lines += list(vertices[v2])
#        all_lines += list(vertices[v3])
#    gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])
#
#        # step the physics
#    gym.simulate(sim)
#    gym.fetch_results(sim, True)
#
#    # update the viewer
#    gym.step_graphics(sim)
#    gym.draw_viewer(viewer, sim, True)
#
#    gym.sync_frame_time(sim)



#sys.exit(1)


#real_position = False
#index = 0
#for t in range(10000000000000):
#    t += 1
#
#    if t % 300 == 0 and index < len(potential_result):
#        gym.set_dof_target_position(envs[-1], spj, potential_result[index][0])
#        gym.set_dof_target_position(envs[-1], slj, potential_result[index][1])
#        gym.set_dof_target_position(envs[-1], ej,  potential_result[index][2])
#        gym.set_dof_target_position(envs[-1], wj1, potential_result[index][3])
#        gym.set_dof_target_position(envs[-1], wj2, potential_result[index][4])
#        gym.set_dof_target_position(envs[-1], wj3, potential_result[index][5])
#        index += 1
#        #gym.set_dof_target_position(envs[-1], likj, 0.3)
#        #gym.set_dof_target_position(envs[-1], lifj, 0.3)
#        #gym.set_dof_target_position(envs[-1], lokj,  0.3)
#        #gym.set_dof_target_position(envs[-1], rikj, -0.3)
#        #gym.set_dof_target_position(envs[-1], rifj, -0.3)
#        #gym.set_dof_target_position(envs[-1], rokj, -0.3)
#
#        real_position = True
#
#    # step the physics
#    gym.simulate(sim)
#    gym.fetch_results(sim, True)
#
#    # update the viewer
#    gym.step_graphics(sim)
#    gym.draw_viewer(viewer, sim, True)
#
#    gym.sync_frame_time(sim)
#
#sys.exit(1)

collision_counts = [0]*len(object_handles)
pp = path_planner(rac, plane_obj, object_collision_models)
path_list = []
plan_index = 0
counter = 0
all_path_points = []
while plan_index < 5 and counter < 5:
    path = pp.plan(potential_result, flex_collision_models, 2**plan_index)
    if path:
        plan_index += 1
        temp_path = []
        print ([x[1] for x in flex_collision_models])
        for t in range(path.getStateCount()):
            state = path.getState(t)
            temp_path.append([state[0], state[1], state[2], 
                                                state[3], state[4], state[5]])
        path_list.append(temp_path)
        

        #for t in range(len(temp_path)):
        #    next_pose = temp_path[t]
        #    for k in range(100):
        #        gym.set_dof_target_position(envs[-1], spj, next_pose[0])
        #        gym.set_dof_target_position(envs[-1], slj, next_pose[1])
        #        gym.set_dof_target_position(envs[-1], ej,  next_pose[2])
        #        gym.set_dof_target_position(envs[-1], wj1, next_pose[3])
        #        gym.set_dof_target_position(envs[-1], wj2, next_pose[4])
        #        gym.set_dof_target_position(envs[-1], wj3, next_pose[5])

        #        gym.simulate(sim)
        #        gym.fetch_results(sim, True)
        #        gym.step_graphics(sim)
        #        gym.draw_viewer(viewer, sim, True)
        #        gym.sync_frame_time(sim)

        for t in range(len(temp_path)-1):
            start, end = temp_path[t], temp_path[t+1]
            distance = sum([(x-y)**2 for x,y in zip(start, end)])
            div = int(max(distance // 10, 1))
            delta_angle = [(y - x)/div for x, y in zip(start, end)]
            current_angle = start[:]
            for k in range(div):
                current_angle = [current_angle[i] + delta_angle[i] for i in range(6)]
                res_points = rac.apply_transform(rac.calculate_transform_from_angles(current_angle))
                scene_occu.register_points(res_points)
                all_path_points += res_points
                #print (current_angle)
    else:
        counter += 1

if plan_index < 5: sys.exit(1)

#region proposal starts here
#create path mesh
object_choice = None
object_counts = -sys.maxsize
for t in range(1, len(flex_collision_models)):
    if flex_collision_models[t][1] > object_counts:
        object_counts = flex_collision_models[t][1]
        object_choice = t

#while not gym.query_viewer_has_closed(viewer):
#    element = object_handles[object_choice]
#    states = gym.get_actor_rigid_body_states(envs[-1], element, 1)
#    rotation = np.array(states[0][0][1])
#    translation = np.array(states[0][0][0])
#    rotation = np.array(rotation.item())
#    translation = np.array(translation.item())
#    #print (rotation, translation)
#    #object_status_list[i][0] += translation
#    r1 = R.from_quat(rotation)
#    tf = fcl.Transform(r1.as_matrix(), translation)
#    flex_collision_models.append([fcl.CollisionObject(object_collision_lib[i], tf), 0])
#
#    temp_obj = object_reader_tracker[object_choice]
#    #temp_obj.set_offset(translation)
#    all_lines = []
#    vertices = temp_obj.get_vertices()
#    for v1, v2, v3 in temp_obj.get_faces():
#        all_lines += list(vertices[v1])
#        all_lines += list(vertices[v2])
#        all_lines += list(vertices[v1])
#        all_lines += list(vertices[v3])
#        all_lines += list(vertices[v2])
#        all_lines += list(vertices[v3])
#    gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])
#
#        # step the physics
#    gym.simulate(sim)
#    gym.fetch_results(sim, True)
#
#    # update the viewer
#    gym.step_graphics(sim)
#    gym.draw_viewer(viewer, sim, True)
#
#    gym.sync_frame_time(sim)



path_points_only = []
for element in all_path_points:
    path_points_only.append(element[0])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(path_points_only)
pcd = pcd.voxel_down_sample(voxel_size = 0.08)
pcd.estimate_normals()
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
path_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.08, tetra_mesh, pt_map)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
#o3d.visualization.draw_geometries([path_mesh, pcd, coord_frame])

old_x, old_y, _ = object_status_list[object_choice][0]
dx, dy, dz = object_status_list[object_choice][1]
chosen_vertices, chosen_faces = object_mesh[object_choice]

for depth in range(len(region_candidates)):
    for new_x, new_y, _ in region_candidates[depth]:
        print (new_x, new_y)
        offset_x, offset_y = new_x - old_x, new_y - old_y
        temp_offset = np.array([offset_x, offset_y, 0])
        new_vertices = np.array(chosen_vertices)
        for i in range(len(new_vertices)):
            new_vertices[i] += temp_offset
        all_lines = []
        for v1, v2, v3 in chosen_faces:
            all_lines += list(new_vertices[v1])
            all_lines += list(new_vertices[v2])
            all_lines += list(new_vertices[v1])
            all_lines += list(new_vertices[v3])
            all_lines += list(new_vertices[v2])
            all_lines += list(new_vertices[v3])

        gym.add_lines(viewer, envs[-1], len(all_lines)//6, all_lines, [1, 0, 0])
        
        for t in range(100):
        
                # step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)
        
            # update the viewer
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
        
            gym.sync_frame_time(sim)

sys.exit(1)

step_count = 0
path_index = 0
#scene_occu.vis_scene()
print (object_status_list)
with open("./object_selection_learning/dataset1/env" + str(env_id) + '.txt', 'w') as f:
    #robot location
    f.write(' '.join([str(x) for x in [ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]]))
    f.write('\n')
    #gripper end
    f.write(' '.join([str(x) for x in ([gripper_location.x, gripper_location.y, gripper_location.z] + [0.446, 0.560, -0.433, 0.549])]))
    f.write('\n')
    for i in range(len(object_handles)):
        loc_x, loc_y, loc_z = object_status_list[i][0]
        dx, dy, dz = object_status_list[i][1]
        temp_arr = [loc_x, loc_y, loc_z, dx, dy, dz, bin(flex_collision_models[i][1])[2:].count('1')]
        f.write(' '.join([str(x) for x in temp_arr]))
        f.write('\n')
sys.exit(1)



while not gym.query_viewer_has_closed(viewer):

    if step_count > 100 and step_count % 100 == 0 and path_index < len(path_list):
        gym.set_dof_target_position(envs[-1], spj, path_list[path_index][0])
        gym.set_dof_target_position(envs[-1], slj, path_list[path_index][1])
        gym.set_dof_target_position(envs[-1], ej,  path_list[path_index][2])
        gym.set_dof_target_position(envs[-1], wj1, path_list[path_index][3])
        gym.set_dof_target_position(envs[-1], wj2, path_list[path_index][4])
        gym.set_dof_target_position(envs[-1], wj3, path_list[path_index][5])
        path_index += 1
    

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    gym.sync_frame_time(sim)
    
    step_count += 1


#*************************************************************************************************#

sys.exit(1)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


