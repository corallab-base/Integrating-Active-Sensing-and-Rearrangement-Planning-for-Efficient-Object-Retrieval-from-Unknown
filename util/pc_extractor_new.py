import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

file_dir = os.path.dirname(__file__)
root_dir = os.path.join(file_dir, '..')
sys.path.append(root_dir)
sys.path.append("/home/corallab3/Documents/active_sensing/loc_learning/")
from test_module.camera_view import camera
from tools import object_completion_network
from YCB_object import YCB_object
from train_classification import test
from train_loc import center_network


def visualize_scene_new(object_dict):
    all_data = []

    for ids, object_handler in object_dict.items():
        seen_data = object_handler.get_seen()
        comp_data = object_handler.get_completion()

        pcd_seen = o3d.geometry.PointCloud()
        pcd_seen.points = o3d.utility.Vector3dVector(seen_data)
        pcd_seen.paint_uniform_color([1, 0, 0])
        all_data.append(pcd_seen)
        
        if comp_data.any():
            pcd_comp = o3d.geometry.PointCloud()
            pcd_comp.points = o3d.utility.Vector3dVector(comp_data)
            pcd_comp.paint_uniform_color([0, 0, 1])
            #o3d.visualization.draw_geometries([pcd_comp, mesh_frame])
            all_data.append(pcd_comp)
   
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.4, origin=[0, 0, 0])
    all_data.append(mesh_frame)

    o3d.visualization.draw_geometries(all_data)
            

class pc_extractor_new:

    #constructor
    def __init__(self, color_image, depth_image, seg_image, cam_rotation, cam_translation, \
                 object_status, object_centroid_m, object_dict, env_id, sequence_count):

        color_raw = o3d.geometry.Image(color_image)
        depth_raw = o3d.geometry.Image(depth_image)
        seg_raw = o3d.geometry.Image(seg_image)
       
        m, n = np.asarray(seg_raw).shape
        offset = np.array(cam_translation)
        rot = R.from_quat(cam_rotation)

        #plt.imshow(depth_raw)
        #plt.show()

        object_list = set()
        for i in range(m):
            for j in range(n):
                object_id = np.asarray(seg_raw)[i][j]
                if object_id != 0 and object_id not in object_list:
                    object_list.add(object_id)
        print (object_list)

        all_data = []

        self.completion_network = object_completion_network()

        self.center_network = center_network()
        
        for ids in object_list:
            object_handler = None
            if ids not in object_dict:
                object_handler = YCB_object()
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

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.4, origin=[0, 0, 0])
            mesh_frame_2 = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.4, origin=[1, 0, 0])


            param = o3d.camera.PinholeCameraIntrinsic(1280, 720, 640, 640, 640, 360)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd_image,
                            o3d.camera.PinholeCameraIntrinsic(
                            param))
            pcd_data = np.array(pcd.points, dtype = np.float32)

            temp_sets = set()
            new_data = []
            for element in np.round(pcd_data, 3):
                if tuple(element) not in temp_sets:
                    temp_sets.add(tuple(element))
                    new_data.append(element)
            pcd_data = np.array(new_data)

            pcd_data[:, [0,1,2]] = pcd_data[:, [2, 0, 1]]
            pcd_data[:, 1] *= -1
            pcd_data[:, 2] *= -1
            
            pcd_data = rot.apply(pcd_data)
            pcd_data += offset

            temp_sets = set()
            for element in np.round(pcd_data, 3):
                if tuple(element) not in temp_sets:
                    temp_sets.add(tuple(element))
            object_handler.add_seen(temp_sets)
            pcd_data = object_handler.get_seen()

            ids -= 1

            object_rotation = R.from_quat([-object_status[ids][0][0],
                                           -object_status[ids][0][1],
                                           -object_status[ids][0][2],
                                            object_status[ids][0][3]])
            back_rotation = R.from_quat([object_status[ids][0][0],
                                         object_status[ids][0][1],
                                         object_status[ids][0][2],
                                         object_status[ids][0][3]])

            write_object_rotation = np.array([object_status[ids][0][0],
                                        object_status[ids][0][1],
                                        object_status[ids][0][2],
                                        object_status[ids][0][3]])
            object_translation = np.array([object_status[ids][1][0],
                                           object_status[ids][1][1],
                                           object_status[ids][1][2]])
            centroid = np.array([object_centroid_m[ids][0][0],
                                 object_centroid_m[ids][0][1],
                                 object_centroid_m[ids][0][2]])
            object_translation += back_rotation.apply(centroid)
            ranges = object_centroid_m[ids][1]

            data_file_pc = './object_loc_data/env_' + str(env_id) + "_seq_" + str(sequence_count) + "_obj_" + str(ids) + ".npy"
            data_file_loc = './object_loc_data/env_' + str(env_id) + "_seq_" + str(sequence_count) + "_obj_" + str(ids) + ".txt"

            with open(data_file_loc, 'w') as f:
                f.write(str(object_translation[0]))
                f.write(" ")
                f.write(str(object_translation[1]))
                f.write(" ")
                f.write(str(object_translation[2]))
                f.write(" ")
                f.write(str(ranges))
                f.write("\n")

            np.save(data_file_pc, pcd_data)
            print (pcd_data.shape)


            if pcd_data.shape[0] < 1856:
                #seen_data = object_handler.get_seen()
                #
                #pcd_seen = o3d.geometry.PointCloud()
                #pcd_seen.points = o3d.utility.Vector3dVector(seen_data)
                #pcd_seen.paint_uniform_color([1, 0, 0])
                #all_data.append(pcd_seen)
                continue


            object_rotation = R.from_quat([-object_status[ids][0][0],
                                           -object_status[ids][0][1],
                                           -object_status[ids][0][2],
                                            object_status[ids][0][3]])
            back_rotation = R.from_quat([object_status[ids][0][0],
                                         object_status[ids][0][1],
                                         object_status[ids][0][2],
                                         object_status[ids][0][3]])


            #object_translation = np.array([object_status[ids][1][0],
            #                               object_status[ids][1][1],
            #                               object_status[ids][1][2]])
            
            pcd_data_2 = np.copy(pcd_data)

            print ("gt translation")
            print (object_translation)
            object_translation_from_net = self.center_network.get_center(np.copy(pcd_data))
            object_translation_from_pc = np.mean(pcd_data, axis = 0)
            print ("infer translation")
            print (object_translation)

            pcd_data -= object_translation_from_net
            pcd_data_2 -= object_translation_from_pc
            #pcd_data = object_rotation.apply(pcd_data)

            centroid = np.array([object_centroid_m[ids][0][0],
                                 object_centroid_m[ids][0][1],
                                 object_centroid_m[ids][0][2]])
            ranges = object_centroid_m[ids][1]

            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(pcd_data)
            temp_pcd_2 = o3d.geometry.PointCloud()
            temp_pcd_2.points = o3d.utility.Vector3dVector(pcd_data_2 + np.array([1, 0, 0]))

            o3d.visualization.draw_geometries([mesh_frame, mesh_frame_2, temp_pcd, temp_pcd_2])


            #pcd_data = pcd_data - centroid
            pcd_data = pcd_data/0.24
            pcd_data_2 = pcd_data_2/0.24

            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(pcd_data)
            temp_pcd_2 = o3d.geometry.PointCloud()
            temp_pcd_2.points = o3d.utility.Vector3dVector(pcd_data_2 + np.array([1, 0, 0]))

            o3d.visualization.draw_geometries([mesh_frame, mesh_frame_2, temp_pcd, temp_pcd_2])

            pcd_data = self.completion_network.complete(np.asarray([pcd_data]))
            pcd_data_2 = self.completion_network.complete(np.asarray([pcd_data_2]))
            
            pcd_data = pcd_data * 0.24
            pcd_data_2 = pcd_data_2 * 0.24
            #pcd_data = pcd_data + centroid

            #pcd_data = back_rotation.apply(pcd_data)
            pcd_data += object_translation_from_net
            pcd_data_2 += object_translation_from_pc
            
            temp_sets = set()
            new_data = []
            for element in np.round(pcd_data, 3):
                if tuple(element) not in temp_sets:
                    temp_sets.add(tuple(element))
                    new_data.append(element)
            object_handler.completion_ = set()
            object_handler.add_completion(temp_sets)

            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(pcd_data)
            temp_pcd_2 = o3d.geometry.PointCloud()
            temp_pcd_2.points = o3d.utility.Vector3dVector(pcd_data_2 + np.array([1, 0, 0]))

            o3d.visualization.draw_geometries([mesh_frame, mesh_frame_2, temp_pcd, temp_pcd_2])


        #dim_x, dim_y, dim_z = 100, 150, 50
        #offset = np.array([0, -0.75, 0])
        #scene_voxel = np.arange(dim_x * dim_y * dim_z * 3).reshape(dim_x * dim_y * dim_z, 3)
        #scene_voxel = scene_voxel.astype(np.float32)
        #for i in range(dim_x):
        #    for j in range(dim_y):
        #        for k in range(dim_z):
        #          scene_voxel[i*dim_y*dim_z + j*dim_z + k] = np.array([i,j,k], dtype = np.float32)*0.01 + \
        #                                                     np.array([0.005, 0.005, 0.005]) + \
        #                                                     offset
        #scene_pcd = o3d.geometry.PointCloud()
        #scene_pcd.points = o3d.utility.Vector3dVector(scene_voxel)
        #scene_pcd.paint_uniform_color([0.9, 0.9, 0.9])

        #for items in all_data:
        #    for points in items.points:
        #        p1, p2, p3 = points
        #        ind1 = round(p1 / 0.01)
        #        ind2 = round(p2 / 0.01) + 75
        #        ind3 = round(p3 / 0.01)
        #        scene_pcd.colors[ind1 * dim_y * dim_z + ind2 * dim_z + ind3] = np.array([1, 1, 0])

        #cam = camera(np.array([0.190782, -0.272692, 0.333791]), np.array([0.000005, 0.225503, 0.250454, 0.941499]))
        #cam.build()

        #for i in range(len(scene_pcd.points)):
        #    query_point = scene_pcd.points[i]
        #    flag, depth, location = cam.inside_frame(query_point)
        #    if flag:
        #        loc_x, loc_y = location
        #        loc_x += 512
        #        loc_y += 512
        #        if 0 <= loc_x < 1024 and 0 <= loc_y < 1024:
        #            if depth*1000 < np.asarray(depth_raw)[loc_y][loc_x]:
        #                if scene_pcd.colors[i][0] == 0.9:
        #                    scene_pcd.colors[i] = np.array([1, 0, 0])
        #            else:
        #                if scene_pcd.colors[i][0] == 0.9:
        #                    scene_pcd.colors[i] = np.array([0, 0, 1])
        #

        ##o3d.visualization.draw_geometries([mesh_frame, scene_pcd])
        #np.save("scene.npy", np.asarray(scene_pcd.points))

        
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
    extractor = pc_extractor_new(color_raw, depth_raw, seg_raw, 
                                      [0.000005, 0.225503, 0.250454, 0.941499],
                                      [0.190782, -0.272692, 0.333791],
                                      object_status, object_centroid, object_dict)
