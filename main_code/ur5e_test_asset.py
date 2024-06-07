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
from grasp_util.robot_arm_configuration import robot_arm_configuration
from grasp_util.robot_arm_configuration import path_planner
from grasp_util.robot_arm_configuration import ur5e_valid

import pdb


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
                new_image[i][j][0] = - int(raw_image[i][j]*1000)
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

print (len(region_candidates), len(region_candidates[0]))

envs = []
ur5e_handles = []
body_cam_handles = []
camera_candidates = []
chosen_object = []
chosen_scale = []
object_normalize = []
# num_of_objects = np.random.randint(min_num_of_objects, max_num_of_objects+1)
num_of_objects = 1

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

    # object_index = np.random.randint(len(object_asset_files), size=num_of_objects-1)
    # object_index = np.insert(object_index, 0, len(object_asset_files)-1, axis = 0)
    object_index = np.array([5])

    #object_index = np.array([8, 21, 21, 21, 21])
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
        

    #set up objects///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # creating manager
    objs_manager = fcl.DynamicAABBTreeCollisionManager()
    objs_manager.setup()
    collision_objs = []

    for k in range(num_of_objects):
        object_pose = gymapi.Transform()
        # tx, ty, tz = object_loc[k]

        # random selec obj location
        is_collision = True
        while is_collision:
            tx = np.random.uniform(0.35, table_dims.x + 0.2)
            ty = np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2)
            tz = table_dims.z + 0.08
            # tx = 0.4
            # ty = -0.2

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
            print("collision result:", is_collision)


        if k == 0: gripper_location = gymapi.Vec3(tx, ty, tz + 0.2)
        #object_pose.p = get_random_loc(0.3 + table_dims.x*0.2, 0.3 + table_dims.x*0.8,
        #                               -table_dims.y*0.4, table_dims.y*0.4,
        #                               table_dims.z, table_dims.z + drawer_height*0.5)
        object_handles.append(gym.create_actor(envs[-1], 
                                               object_assets[object_index[k]], 
                                               object_pose, 
                                               "object" + str(k) + str(i), 0, 2**(k+1), k+1))
        gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
        #object_normalize.append(object_centroid_m[object_index[k]])
        object_reader_tracker.append(collision_mesh)
        object_status_list.append([temp_center, temp_bounding_box])
        object_collision_lib.append(m)
        collision_objs.append(fcl.CollisionObject(m, t))
        objs_manager.registerObjects(collision_objs)
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
target_pos = gripper_location
print (target_pos)
converted_coord = global_coord_converter(target_pos.x ,
                                         target_pos.y ,
                                         target_pos.z , 
                                         ur5e_pose.p.x, 
                                         ur5e_pose.p.y, 
                                         ur5e_pose.p.z)

target_quat = gymapi.Quat(0.446, 0.560, -0.433, 0.549)
converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)
file_path = '../assets/urdf/ur5e/meshes/collision/'
rac = robot_arm_configuration(file_path, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]))
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

# # no active sensing part
# while not gym.query_viewer_has_closed(viewer):
#      # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
    
#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
    
#     gym.sync_frame_time(sim)

#active sensing here
while not gym.query_viewer_has_closed(viewer):
    if coverage_score >= 0.9 or sequence_count >= 10: break
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
                    color_img_saved = write_to_image(color_image, 'test_data/test_image/' + str(sequence_count) + '.png')
                    write_to_seg_image(seg_image, 'test_data/test_seg_image/' + str(sequence_count) + '.png')
                    write_to_depth_image(depth_image, 'test_data/test_depth_image/' + str(sequence_count) + '.png')
                    # pdb.set_trace()
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
                    
                    write_for_contact_grasp(color_img_saved, seg_image, -depth_image, K, new_cam_rotation, new_cam_translation,'test_data/test_npy/' + str(sequence_count)+'.npy')

                    dist1 = np.linalg.norm(final_rotation - new_cam_rotation)
                    dist3 = np.linalg.norm(final_rotation - new_cam_rotation*-1)
                    dist2 = np.linalg.norm(final_translation - new_cam_translation)

                    if (dist1 < 1e-2 or dist3 < 1e-2) and dist2 < 1e-2:
                        print ('start pc extraction')
                        pc_extractor_grasp(new_rgb_image, new_depth_image, new_seg_image, new_cam_rotation, new_cam_translation, object_dict, table_dims.z)

                        coverage_score = scene.register_camera_view(list(new_cam_rotation), list(new_cam_translation), new_depth_image, object_dict)

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
            camera_loc, camera_focus, dof_result = random_sample_guided_selection(sim, envs[-1], test_cam, scene)

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


