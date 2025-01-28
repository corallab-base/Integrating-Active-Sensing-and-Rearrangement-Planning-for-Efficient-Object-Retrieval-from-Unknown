import os
import sys
import numpy as np
import math
import open3d as o3d
from scipy.spatial.transform import Rotation as R

file_dir = os.path.dirname(__file__)
util_dir = os.path.join(file_dir, '../util')
learning_dir = os.path.join(file_dir, '../learning')
sys.path.append(util_dir)
# sys.path.append('/home/j0k/coral/ompl-1.5.2/py-bindings')
# sys.path.append(learning_dir)
# import ompl.base as ob
# import ompl.util as ou
# import ompl.geometric as og
# from stl_reader import stl_reader
# from obj_reader import obj_reader
from pc_extractor_real_cam import pc_extractor_real_cam
from pc_extractor_real_cam import visualize_scene
from pc_extractor_real_cam import save_object
# from pc_extractor_real_cam import save_object_ind
# from pc_extractor_real_cam import save_object_no_bg
from global_scene_real import global_scene_real
# from runner import feed_forward

def get_real_rotation(rx, ry, rz):

    vector = np.array([rx, ry, rz])
    mag = np.linalg.norm(vector)
    vector /= mag
    rot1 = R.from_rotvec(mag * vector)
    rot2 = R.from_euler("YX", [-math.pi/2, math.pi/2])

    rot2 = rot1 * rot2

    t1, t2, t3, t4 = rot2.as_quat()

    rot3 = R.from_quat([-t1, -t2, t3, t4])

    #print (rot3.apply([-1, 0, 0]))
    #print (rot3.apply([0, -1, 0]))
    #print (rot3.apply([0, 0, 1]))

    rot4 = R.from_euler('Z', math.pi)
    rot4 = rot3*rot4

    #print (rot4.apply([1, 0, 0]))
    #print (rot4.apply([0, 1, 0]))
    #print (rot4.apply([0, 0, 1]))

    #print(rot4.as_quat())

    return rot4.as_quat()

def get_real_location(x, y, z, rotation):
    offset = R.from_quat(rotation).apply([0.035, 0.03, 0.07])
    print ("offset is: ")
    print (offset)
    #return [-0.19 - x + offset[0], -0.18 - y + offset[1], z + offset[2]]
    return [ - x + offset[0], - y + offset[1], z + offset[2]]



if __name__ == '__main__':
    scene = global_scene_real(1.0, 1.2, 0.85, np.array([0.3, -0.6, 0.05]), 0.63, 1.12*0.8 - 0.04, 0.5, 0.05)

    coverage_score = 0

    object_dict = {}

    vp_method = 3

    env_id = 10000

    sequence_count = 0

    get_real_rotation(3.86, 1.781, -0.908)
    root = "test_data/revisit_data_example/3"
    while True:
        next_id = input('Enter the next id: ')

        # div = next_id.split()

        # real_id = div[0]
        # sequence_count = div[1]

        #robot_rot = [float(x) for x in div[1:4]]
        #robot_trans = [float(x) for x in div[4:7]]

        #rotation = get_real_rotation(robot_rot[0], robot_rot[1], robot_rot[2])

        #translation = get_real_location(robot_trans[0], robot_trans[1], robot_trans[2], rotation)

        #rotation = [float(x) for x in div[1:5]]
        #translation = [float(x) for x in div[5:8]]
                                
        #print ("real rotation: ", rotation)
        #print ("real translation:", translation)
        img_color_file = root + '/test_image' + '/' + str(next_id) + '.png'
        img_depth_file = root + '/test_depth_image' + '/' + str(next_id) + '.png'
        img_seg_file = root + '/test_seg_image' + '/' + str(next_id) + '.png'
        img_rot_file = root + '/test_cam_info' + '/' + str(next_id) + '.npy'
        rot_data = np.load(img_rot_file)
        rotation = rot_data[:4]
        translation = rot_data[4:7]

        new_rgb_image = o3d.io.read_image(img_color_file)
        new_depth_image = o3d.io.read_image(img_depth_file)
        new_seg_image = o3d.io.read_image(img_seg_file)


        res = pc_extractor_real_cam(new_rgb_image, new_depth_image, new_seg_image, 
                                    rotation, translation, object_dict, False)

        visualize_scene(object_dict, False, True)
        # visualize_scene(object_dict, False, False)

        file_prefix = 'real_exp/' + str(vp_method) + '/' + str(env_id) + '/env' + str(env_id) + "_sequence" + str(sequence_count)

        # coverage_score = scene.register_camera_view(list(rotation), \
        #                                             list(translation), \
        #                                             np.asarray(new_depth_image), object_dict, file_prefix)

        # scene.vis_scene(file_prefix)
        # save_object_ind(object_dict, file_prefix)
        #save_object_no_bg(object_dict, file_prefix)

        sequence_count += 1


