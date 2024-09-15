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
import MCTS_algo_ICRA_OG as mct_OG
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
table_dims = gymapi.Vec3(0.76, 1.16, 0.10) # L

# table_dims = gymapi.Vec3(np.random.uniform(0.7, 0.9), np.random.uniform(1, 1.2), 0.10)
# max -> 1.0, 1.2
# For testing X: 0.7 + 0~0.2    Y: 1 + 0~0.2 
piece_width = 0.03
# min_num_of_objects = 15
# max_num_of_objects = 20
max_scaling_factor = 0
fall_height = table_dims.z
max_drawer_height = 0.5
min_drawer_height = 0.5
ADD_COVER = True

OBSTACLE_OBJ_INDEX = [0, 2]
MIN_RADIUS = 0.03471716871486391

MIN_NUM_OBSTACLES = 5
MAX_NUM_OBSTACLES = 8
NUM_OF_OBJECTS = 11

scene_choose_idx = 2
GAP_TO_BOX = 0

if scene_choose_idx == 0:
# Large scene ----------------------------------------------------------------------------------------
    print("---------- Large scene ----------")
    table_dims = gymapi.Vec3(0.8, 1.2, 0.15)
    max_drawer_height = 0.5
    min_drawer_height = 0.5
    NUM_OF_OBJECTS = 11
    NUM_SWEPT_COLLISION = 4
    TARGET_OBJ_INDEX = [3]
# ----------------------------------------------------------------------------------------------------
elif scene_choose_idx == 1:
# Medium scene ----------------------------------------------------------------------------------------
    print("---------- Medium scene ----------")
    table_dims = gymapi.Vec3(0.76, 1.1, 0.10)
    max_drawer_height = 0.4
    min_drawer_height = 0.4
    NUM_OF_OBJECTS = 9
    NUM_SWEPT_COLLISION = 3
    TARGET_OBJ_INDEX = [3, 5]
# ----------------------------------------------------------------------------------------------------

elif scene_choose_idx == 2:
# Small scene ----------------------------------------------------------------------------------------
    print("---------- Small scene ----------")
    table_dims = gymapi.Vec3(0.72, 1.0, 0.05)
    max_drawer_height = 0.35
    min_drawer_height = 0.35
    NUM_OF_OBJECTS = 7
    NUM_SWEPT_COLLISION = 3
    TARGET_OBJ_INDEX = [5]
# ----------------------------------------------------------------------------------------------------
# NUM_OF_OBJECTS = np.random.randint(MIN_NUM_OBSTACLES + 1, MAX_NUM_OBSTACLES + 1)


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
    # ur5e_collision_models = []
    # ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
    #                 R.from_euler('xy', [90, 180], degrees = True),
    #                 R.from_euler('xy', [180, 180], degrees = True),
    #                 R.from_euler('z',  [-180], degrees = True),
    #                 R.from_euler('x',  [-180], degrees = True),
    #                 R.from_euler('x',  [90], degrees = True),
    #                 R.from_euler('z',  [-90], degrees = True)]
    # ur5e_translations = [[0, 0, 0], 
    #                     [0, 0, 0],
    #                     [0, -0.138, 0],
    #                     [0, -0.007, 0],
    #                     [0, 0.127, 0],
    #                     [0, 0, 0],
    #                     [0, 0, 0]]
    # for i in range(len(ur5e_collision_parts)):
    #     parts_path = ur5e_collision_parts[i]
    #     collision_mesh = stl_reader(asset_root + parts_path)
    #     m = fcl.BVHModel()
    #     collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
    #     verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
    #     m.beginModel(len(verts), len(tris))
    #     m.addSubModel(verts, tris)
    #     m.endModel()
    #     ur5e_collision_models.append(m)

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
    table_asset = gym.create_box(sim, table_dims.x + 0.10,
                                    table_dims.y + 0.10,
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
    
    back_cover_dims = gymapi.Vec3(0.03, table_dims.y, table_dims.z + drawer_height)
    back_cover_asset = gym.create_box(sim, back_cover_dims.x,
                                            back_cover_dims.y,
                                            back_cover_dims.z,
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
    # ur5e_pose = gymapi.Transform()
    # # ur5e_pose.p = gymapi.Vec3(np.random.rand()*0.3 - 0.2, np.random.rand()*0.4 - 0.2, 0.0)
    # ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    # ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

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

    back_cover_pose = gymapi.Transform()
    back_cover_pose.p = gymapi.Vec3(table_pose.p.x - table_dims.x/2 - back_cover_dims.x/2, 0.0, drawer_height / 2 + table_dims.z/2)

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

        # back_cover = fcl.Box(upper_cover_dims.x,
        #                         upper_cover_dims.y,
        #                         upper_cover_dims.z)

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
        # ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))

        # #get joint handler
        # spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        # slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        # ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        # wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        # wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        # wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
        # likj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_knuckle_joint")
        # lifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_finger_joint")
        # lokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_outer_knuckle_joint")
        # rikj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_knuckle_joint")
        # rifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_finger_joint")
        # rokj = gym.find_actor_dof_handle(envs[if scene_choose_idx == 0:

        # body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        # body_cam_transform = gymapi.Transform()
        # body_cam_transform.p = gymapi.Vec3(cam_offset_x, 0, cam_offset_z)
        # # gym.attach_camera_to_body(body_cam_handles[-1], envs[-1], cam_link, body_cam_transform, 
        # #                         gymapi.CameraFollowMode.FOLLOW_TRANSFORM)

        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
        gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)
        if ADD_COVER:
            gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)
            gym.create_actor(envs[-1], back_cover_asset, back_cover_pose, "back_cover" + str(i), 0, 1)
            gym.set_rigid_body_color(envs[-1], 0, 0, gymapi.MeshType.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7,0.7,0.7))
            gym.set_rigid_body_color(envs[-1], 1, 0, gymapi.MeshType.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7,0.7,0.7))
            gym.set_rigid_body_color(envs[-1], 2, 0, gymapi.MeshType.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7,0.7,0.7))
            gym.set_rigid_body_color(envs[-1], 3, 0, gymapi.MeshType.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7,0.7,0.7))
            gym.set_rigid_body_color(envs[-1], 4, 0, gymapi.MeshType.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.7,0.7,0.7))

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
        # GT_TARGET_POS = [np.random.uniform(0.20 + table_dims.x/2, table_dims.x),
        #                  np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2),
        #                  table_dims.z + 0.08]
        
        GT_TARGET_POS = [0.6, 0.01, table_dims.z + 0.08]

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
                tx = np.random.uniform(0.50, table_dims.x + 0.2)
                ty = np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2)
                tz = table_dims.z + 0.08

                if k == 0:
                    tx = GT_TARGET_POS[0] + 0.16
                    ty = GT_TARGET_POS[1] + 0.01

                # if k == 1:
                #     tx = GT_TARGET_POS[0] + 0.16
                #     ty = GT_TARGET_POS[1] - 0.15

                # if k == 2:
                #     tx = GT_TARGET_POS[0] + 0.08
                #     ty = GT_TARGET_POS[1] + 0.25

                # if k == 3:
                #     tx = GT_TARGET_POS[0] + 0.03
                #     ty = GT_TARGET_POS[1] - 0.3

                # if k == 4:
                #     tx = 0.3 + table_dims.x - 0.17
                #     ty = table_dims.y/2 - 0.30

                # if k == 5:
                #     tx = 0.3 + table_dims.x - 0.10
                #     ty = -table_dims.y/2 + 0.2

                # if k == 6:
                #     tx = 0.3 + table_dims.x - 0.12
                #     ty = -0.03

                # if k == 7:
                #     tx = 0.3 + table_dims.x - 0.15
                #     ty = -table_dims.y/2 + 0.3

                # if k == 4:
                #     tx = table_dims.x + 0.07
                #     ty = table_dims.y - 0.08

                object_pose.p = gymapi.Vec3(tx, ty, tz)

                file_path = object_collision_files[OBJ_FILE_IDX_LIST[k]]
                collision_mesh = obj_reader(asset_root + file_path)
                collision_mesh.set_scale(object_scaling_factor[k])
                collision_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[k]])
                
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

                # if k not in [0,1,2,3,4, 5]:
                if k not in [0]:
                    if not is_collision:
                        dist = np.sqrt((tx - GT_TARGET_POS[0])**2 + (ty - GT_TARGET_POS[1])**2)
                        if dist <= 0.15:
                            is_collision = True
                            print("target contact recalc")
                            continue

                        for obj in GT_OBJ_POS_LIST:
                            dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                            if dist <= 0.15:
                                is_collision = True
                                print("recalc")
                                continue

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
        
        np.save("scene_obj_pose" + str(time.localtime()), np.array(GT_OBJ_POS_LIST))

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
    # for t in range(2000):
    #     if not real_position:
    #         gym.set_dof_target_position(envs[-1], spj, 0)
    #         gym.set_dof_target_position(envs[-1], slj, -math.pi/2)
    #         gym.set_dof_target_position(envs[-1], ej,  0)
    #         gym.set_dof_target_position(envs[-1], wj1, -math.pi/2)
    #         gym.set_dof_target_position(envs[-1], wj2, 0)
    #         gym.set_dof_target_position(envs[-1], wj3, 0)
    #         real_position = True

    #     if t == 999:
    #         for i in range(len(object_handles)):
    #             element = object_handles[i]
    #             states = gym.get_actor_rigid_body_states(envs[-1], element, 1)
    #             rotation = np.array(states[0][0][1])
    #             translation = np.array(states[0][0][0])
    #             rotation = np.array(rotation.item())
    #             translation = np.array(translation.item())
    #             object_status_list[i][0] += translation
    #             r1 = R.from_quat(rotation)
    #             tf = fcl.Transform(r1.as_matrix(), translation)
    #             flex_collision_models.append([fcl.CollisionObject(object_collision_lib[i], tf), 0])

    #             temp_obj = object_reader_tracker[i]
    #             temp_obj.set_offset(translation)
    #             all_lines = [] 
    #             vertices, faces = temp_obj.get_bounding_box_mesh()
    #             object_mesh.append([vertices, faces])
    #             for v1, v2, v3 in faces:
    #                 all_lines += list(vertices[v1])
    #                 all_lines += list(vertices[v2])
    #                 all_lines += list(vertices[v1])
    #                 all_lines += list(vertices[v3])
    #                 all_lines += list(vertices[v2])
    #                 all_lines += list(vertices[v3])

    #     # step the physics
    #     gym.simulate(sim)
    #     gym.fetch_results(sim, True)

    #     # update the viewer
    #     gym.step_graphics(sim)
    #     gym.draw_viewer(viewer, sim, True)

    #     gym.sync_frame_time(sim)
    #*************************************************************************************************#
    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    sys.exit(1)