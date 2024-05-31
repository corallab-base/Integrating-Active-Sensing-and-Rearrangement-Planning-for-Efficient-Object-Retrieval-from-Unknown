import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

file_dir = os.path.dirname(__file__)
root_dir = os.path.join(file_dir, '../../')
sys.path.append(root_dir)
from test_module.camera_view import camera
#from tools import object_completion_network
from grasp_object import grasp_object


def visualize_scene(object_dict):
   
    all_data = []
    for ids, object_handler in object_dict.items():
        seen_data = object_handler.get_seen()
        comp_data = object_handler.get_completion()

        pcd_seen = o3d.geometry.PointCloud()
        pcd_seen.points = o3d.utility.Vector3dVector([list(x) for x in seen_data.keys()])
        pcd_seen.colors = o3d.utility.Vector3dVector([list(x) for x in seen_data.values()])
        all_data.append(pcd_seen)
        
        if flag and len(comp_data) > 0:
            pcd_comp = o3d.geometry.PointCloud()
            pcd_comp.points = o3d.utility.Vector3dVector([list(x) for x in comp_data.keys()])
            pcd_comp.colors = o3d.utility.Vector3dVector([list(x) for x in comp_data.values()])
            #o3d.visualization.draw_geometries([pcd_comp, mesh_frame])
            all_data.append(pcd_comp)
   
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.4, origin=[0, 0, 0])
    all_data.append(mesh_frame)

    o3d.visualization.draw_geometries(all_data)


def save_object(object_dict, file_prefix):
    seen_data = []
    comp_data = []
    seen_color = []
    comp_color = []

    for ids, object_handler in object_dict.items():
        seen_obj = object_handler.get_seen()
        comp_obj = object_handler.get_completion()

        for element in seen_obj.keys():
            seen_data.append(list(element))
        for element in comp_obj.keys():
            comp_data.append(list(element))
        for element in seen_obj.values():
            seen_color.append(list(element))
        for element in comp_obj.values():
            comp_color.append(list(element))

    seen_data = np.array(seen_data)
    comp_data = np.array(comp_data)
    seen_color = np.array(seen_color)
    comp_color = np.array(comp_color)

    with open(file_prefix + "_seen_object.npy", 'wb') as f:
        np.save(f, seen_data)
    
    with open(file_prefix + "_comp_object.npy", 'wb') as f:
        np.save(f, comp_data)

    with open(file_prefix + "_seen_color.npy", 'wb') as f:
        np.save(f, seen_color)

    with open(file_prefix + "_comp_color.npy", 'wb') as f:
        np.save(f, comp_color)



    


class pc_extractor_grasp:

    #constructor
    def __init__(self, color_image, depth_image, seg_image, cam_rotation, cam_translation, object_dict, z_min):


        #environment_pc = []
        #environment_colors = []
        #for i in range(0, 57):
        #    for j in range(-46, 47):
        #        environment_pc.append([i*0.01 + 0.3, j*0.01, 0.15])
        #        environment_pc.append([i*0.01 + 0.3, j*0.01, 0.82])
        #        environment_colors.append([0.5, 0.5, 0.5])
        #        environment_colors.append([0.5, 0.5, 0.5])
        #for i in range(0, 57):
        #    for k in range(15, 83):
        #        environment_pc.append([i*0.01 + 0.3, -0.46, k*0.01])
        #        environment_pc.append([i*0.01 + 0.3, 0.46, k*0.01])
        #        environment_colors.append([0.5, 0.5, 0.5])
        #        environment_colors.append([0.5, 0.5, 0.5])


        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_image)
        seg_raw = o3d.geometry.Image(seg_image)
       
        m, n = np.asarray(seg_raw).shape
        offset = np.array(cam_translation)
        rot = R.from_quat(cam_rotation)

        #plt.imshow(depth_raw)
        #plt.show()

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                     size=0.4, origin=[0, 0, 0])


        object_list = set()
        for i in range(m):
            for j in range(n):
                object_id = np.asarray(seg_raw)[i][j]
                if object_id not in object_list and object_id != 0:
                    object_list.add(object_id)
        print (object_list)

        all_data = []
        all_data_no_comp = []

        #self.completion_network = object_completion_network()
        
        for ids in object_list:
            object_handler = None
            if ids not in object_dict:
                object_handler = grasp_object(z_min)
                object_dict[ids] = object_handler
            else:
                object_handler = object_dict[ids]
            color_copy = o3d.geometry.Image(color_raw)
            depth_copy = o3d.geometry.Image(depth_raw)
            for i in range(m):
                for j in range(n):
                    if np.asarray(seg_raw)[i][j] != ids:
                        np.asarray(color_copy)[i][j][0] = 0
                        np.asarray(color_copy)[i][j][1] = 0
                        np.asarray(color_copy)[i][j][2] = 0
                        np.asarray(depth_copy)[i][j] = 0
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_copy, depth_copy,
                                      convert_rgb_to_intensity = False)

            param = o3d.camera.PinholeCameraIntrinsic(1280, 720, 910, 910, 640, 360)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd_image,
                            o3d.camera.PinholeCameraIntrinsic(
                            param))
            pcd_data = np.array(pcd.points, dtype = np.float32)
            temp_sets = set()
            new_data = []
            num, dim = pcd_data.shape

            for i in range(num):
                element = np.round(pcd_data[i], 3)
                if tuple(element) not in temp_sets:
                    temp_sets.add(tuple(element))
                    new_data.append(element)
            if (len(new_data) < 2): continue

            pcd_data = np.array(new_data)

            pcd_data[:, [0,1,2]] = pcd_data[:, [2, 0, 1]]
            pcd_data[:, 1] *= -1
            pcd_data[:, 2] *= -1
            
            pcd_data = rot.apply(pcd_data)
            pcd_data += offset

            temp_sets = set()
            num, dim = pcd_data.shape
            for i in range(num):
                element = np.round(pcd_data[i], 3)
                if tuple(element) not in temp_sets:
                    temp_sets.add(tuple(element))

            object_translation_from_pc = np.mean(pcd_data, axis = 0)
            #create pcd
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(pcd_data)

            object_handler.add_seen(temp_sets)

        
    def get_point_cloud(self):
        return self.pts_


if __name__ == "__main__":
    
    status_dict = {}
    start = 0
    object_status = []
    with open("track_object.txt") as f:
        for line in f:
            start += 1
            divs = line.split()
            status_dict[start] = [float(x) for x in divs]
            divs = [float(x) for x in divs]
            object_status.append([np.array(divs[:4]), np.array(divs[4:7])])

    normal_dict = {}
    start = 0
    ids = [2,5,6,7,9]
    object_centroid = [[np.array([1,1,1]), 1] for _ in range(20)]
    with open("normal.txt") as f:
        for line in f:
            divs = line.split()
            normal_dict[ids[start]] = [float(x) for x in divs]
            divs = [float(x) for x in divs]
            object_centroid[ids[start]-1] = [np.array(divs[:3]), divs[3]]
            start += 1

    
    print (normal_dict.items())
    object_dict = {}
    color_raw = o3d.io.read_image('../sim/captured_images/color_test4.jpg')
    depth_raw = o3d.io.read_image('../sim/captured_images/depth_test4.png')
    seg_raw = o3d.io.read_image('../sim/captured_images/seg_test4.png')
    extractor = pc_extractor(color_raw, depth_raw, seg_raw, 
                                      [0.000005, 0.225503, 0.250454, 0.941499],
                                      [0.190782, -0.272692, 0.333791],
                                      object_status, object_centroid, object_dict)
