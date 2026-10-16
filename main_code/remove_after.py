from scipy.spatial.transform import Rotation as R
import math
import time
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import sys
import os

file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../util')
sys.path.append(util_dir)

# --- 1. CONFIGURATION ---

# Robot Configurations
ROBOT_CONFIGS = [
    {
        "name": "ur5e_1",
        "position": [0.0, 0.0, 0.0],
        "rotation_euler_xyz": [90.0, 0.0, 0.0], 
        "dof_trajectory": [
            [2.84980428, -0.94914, -0.5628, 0.206442, 0.870935, 1.974515],
            [2.84980428, -0.94914, -0.5628, 0.206442, 0.870935, 1.974515]
        ]
    },
    {
        "name": "ur5e_2",
        "position": [0.0, 0.4, 0.0],
        "rotation_euler_xyz": [90.0, 0.0, 180.0], # Facing the first robot
        "dof_trajectory": [
            [0.17279, -0.59842, -1.839979, -0.612313, -2.22562, -0.688921],
            [0.17279, -0.59842, -1.839979, -0.612313, -2.22562, -0.688921]
        ]
    },
    # {
    #     "name": "ur5e_3",
    #     "position": [0.0, -0.4, 0.0],
    #     "rotation_euler_xyz": [90.0, 0.0, 180.0], # Facing the first robot
    #     "dof_trajectory": [
    #         [0.807279, -1.79842, -1.539979, 0.612313, -0.622562, 0.288921],
    #         [0.807279, -1.79842, -1.539979, 0.612313, -0.622562, 0.288921]
    #     ]
    # },

    # {
    #     "name": "ur5e_1",
    #     "position": [0.0, 0.6, 0.0],
    #     "rotation_euler_xyz": [90.0, 0.0, -90.0], 
    #     "dof_trajectory": [
    #         [-0.10980428, -0.94914, -0.032628, 0.706442,-0.470935, -0.274515],
    #         [-0.10980428, -0.94914, -0.032628, 0.706442,-0.470935, -0.274515]
    #     ]
    # },
    # {
    #     "name": "ur5e_2",
    #     "position": [0.6, 0.0, 0.0],
    #     "rotation_euler_xyz": [90.0, 0.0, 180.0], # Facing the first robot
    #     "dof_trajectory": [
    #         [-1.17279, -0.59842, -0.539979, -0.612313, -0.622562, 0.288921],
    #         [-1.17279, -0.59842, -0.539979, -0.612313, -0.622562, 0.288921]
    #     ]
    # },
    # {
    #     "name": "ur5e_3",
    #     "position": [0.0, -0.6, 0.0],
    #     "rotation_euler_xyz": [90.0, 0.0, 90.0], # Facing the first robot
    #     "dof_trajectory": [
    #         [-0.207279, -0.19842, -0.539979, -0.612313, -0.622562, 0.288921],
    #         [-0.207279, -0.19842, -0.539979, -0.612313, -0.622562, 0.288921]
    #     ]
    # },
    # {
    #     "name": "ur5e_4",
    #     "position": [-0.6, 0.0, 0.0],
    #     "rotation_euler_xyz": [90.0, 0.0, 0.0], # Facing the first robot
    #     "dof_trajectory": [
    #         [-0.607279, -0.59842, -0.539979, -0.612313, -0.622562, 0.288921],
    #         [-0.607279, -0.59842, -0.539979, -0.612313, -0.622562, 0.288921]
    #     ]
    # }
]

# Obstacle Configurations
# Dimensions are [width(x), depth(y), height(z)]
# Position is the center of the box [x, y, z]
OBSTACLE_CONFIGS = [
    # {
    #     "name": "center_wall",
    #     "position": [0.0, 0.0, 0.2],
    #     "dimensions": [0.4, 0.15, 0.40], 
    #     "color": [1.0, 1.0, 1.0] # Red
    # },
    # {
    #     "name": "side_wall1",
    #     "position": [0.6, 0.0, 0.3],
    #     "dimensions": [0.1, 0.5, 0.6],
    #     "color": [1.0, 1.0, 1.0] # Blue
    # },
    # {
    #     "name": "side_wall2",
    #     "position": [-0.6, 0.0, 0.3],
    #     "dimensions": [0.1, 0.5, 0.6],
    #     "color": [1.0, 1.0, 1.0] # Blue
    # }
]


if __name__ == '__main__':
    # Initialize gym
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(description="Multi UR5e Environment", custom_parameters=[{'name':'--env_id', 'type':int, 'help':'env_id', 'default':0}])
    
    # Create simulator
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
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

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # Configure ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load UR5e Asset
    asset_root = "/home/j0k/Project/Imsa/assets/"
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)

    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise ValueError('*** Failed to create viewer')

    # Set up environment
    spacing = 3.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)
    
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Setup Robots
    robot_dof_handles = []
    for config in ROBOT_CONFIGS:
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*config["position"])
        r = R.from_euler('xyz', config["rotation_euler_xyz"], degrees=True).as_quat()
        pose.r = gymapi.Quat(r[0], r[1], r[2], r[3])

        actor_handle = gym.create_actor(env, ur5e_asset, pose, config["name"], 0, 1)

        handles = {
            "spj": gym.find_actor_dof_handle(env, actor_handle, "shoulder_pan_joint"),
            "slj": gym.find_actor_dof_handle(env, actor_handle, "shoulder_lift_joint"),
            "ej":  gym.find_actor_dof_handle(env, actor_handle, "elbow_joint"),
            "wj1": gym.find_actor_dof_handle(env, actor_handle, "wrist_1_joint"),
            "wj2": gym.find_actor_dof_handle(env, actor_handle, "wrist_2_joint"),
            "wj3": gym.find_actor_dof_handle(env, actor_handle, "wrist_3_joint")
        }
        robot_dof_handles.append(handles)

    # Setup Obstacles
    for obs_config in OBSTACLE_CONFIGS:
        # Create a unique box asset for each obstacle based on dimensions
        obs_options = gymapi.AssetOptions()
        obs_options.fix_base_link = True # Makes the obstacle static
        
        dim_x, dim_y, dim_z = obs_config["dimensions"]
        obs_asset = gym.create_box(sim, dim_x, dim_y, dim_z, obs_options)
        
        obs_pose = gymapi.Transform()
        obs_pose.p = gymapi.Vec3(*obs_config["position"])
        
        # We put obstacles in collision group 0, but you can change this if you need specific filtering
        obs_handle = gym.create_actor(env, obs_asset, obs_pose, obs_config["name"], 0, 1)
        
        # Apply color if specified
        if "color" in obs_config:
            color = gymapi.Vec3(*obs_config["color"])
            gym.set_rigid_body_color(env, obs_handle, 0, gymapi.MESH_VISUAL, color)


    # Set Camera
    cam_pos = gymapi.Vec3(3.0, 0.75, 1.5)
    cam_target = gymapi.Vec3(0, 0.75, 0.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Simulation Variables
    trajectory_step = 0
    frame_counter = 0
    frames_per_step = 60

    print("Starting Simulation...")
    while not gym.query_viewer_has_closed(viewer):
        
        for i, config in enumerate(ROBOT_CONFIGS):
            traj = config["dof_trajectory"]
            current_target = traj[min(trajectory_step, len(traj) - 1)]
            handles = robot_dof_handles[i]

            gym.set_dof_target_position(env, handles["spj"], current_target[0])
            gym.set_dof_target_position(env, handles["slj"], current_target[1])
            gym.set_dof_target_position(env, handles["ej"],  current_target[2])
            gym.set_dof_target_position(env, handles["wj1"], current_target[3])
            gym.set_dof_target_position(env, handles["wj2"], current_target[4])
            gym.set_dof_target_position(env, handles["wj3"], current_target[5])

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        frame_counter += 1
        if frame_counter >= frames_per_step:
            trajectory_step += 1
            frame_counter = 0

    print('Test Completed Successfully!!')
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    sys.exit(1)