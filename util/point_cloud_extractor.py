import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
from tools import object_completion_network

file_dir = os.path.dirname(__file__)
root_dir = os.path.join(file_dir, '..')
sys.path.append(root_dir)
from test_module.camera_view import camera



class point_cloud_extractor:

    #constructor
    def __init__(self, color_image, depth_image, seg_image, offset, rotation):
        self.pts_ = None

        color_raw = o3d.io.read_image(color_image)
        depth_raw = o3d.io.read_image(depth_image)
        seg_raw = o3d.io.read_image(seg_image)
        m, n = np.asarray(seg_raw).shape
        offset = np.array(offset)
        rot = R.from_matrix(R.from_quat(rotation).as_matrix())

        points = [np.array([1, -1, -1]), np.array([1, -1, 1]), np.array([1, 1, -1]), np.array([1, 1, 1])]

        new_points = []
        for element in points:
            new_points.append(rot.apply(element))
            print (type(new_points[-1]))

        lines = [[0, 1], [1, 3], [2, 3], [2, 0]]
        colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1,1,1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(new_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.8, origin=[0, 0, 0])

        o3d.visualization.draw_geometries([line_set, mesh_frame])
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
        

       
        for ids in object_list:
            print (ids)
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

            param = o3d.camera.PinholeCameraIntrinsic(1024, 1024, 512, 512, 512, 512)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                            rgbd_image,
                            o3d.camera.PinholeCameraIntrinsic(
                            param))
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(pcd.points)
            pcd = new_pcd

            print ("here")

            if (len(pcd.points) < 2048): continue

            for i in range(len(pcd.points)):
                c1, c2, c3 = np.asarray(pcd.points)[i]
                np.asarray(pcd.points)[i][0] = c3 
                np.asarray(pcd.points)[i][1] = -c1
                np.asarray(pcd.points)[i][2] = -c2

            copy_pcd = o3d.geometry.PointCloud(pcd)
            all_data.append(copy_pcd)

            temp_res = rot.apply(np.asarray(pcd.points))
            for i in range(len(pcd.points)):
                np.asarray(pcd.points)[i][0] = temp_res[i][0]
                np.asarray(pcd.points)[i][1] = temp_res[i][1]
                np.asarray(pcd.points)[i][2] = temp_res[i][2]
                c1, c2, c3 = np.array(pcd.points)[i]

            o3d.visualization.draw_geometries([pcd, mesh_frame])

            #for i in range(len(pcd.points)):
            #    np.asarray(pcd.points)[i] += offset

            all_data.append(pcd)

            #self.pts_ = pcd

            #original_pcd = o3d.geometry.PointCloud(pcd)            

            #object_rotation = R.from_quat([-status_dict[ids][0],
            #                              -status_dict[ids][1],
            #                              -status_dict[ids][2],
            #                               status_dict[ids][3]])
            #back_rotation = R.from_quat([status_dict[ids][0],
            #                             status_dict[ids][1],
            #                             status_dict[ids][2],
            #                             status_dict[ids][3]])


            #object_translation = np.array([status_dict[ids][4],
            #                               status_dict[ids][5],
            #                               status_dict[ids][6]])


            #pcd_ori = o3d.geometry.PointCloud(pcd)
            #for i in range(len(pcd.points)):
            #    np.asarray(pcd_ori.points)[i] -= object_translation

            #temp_res = object_rotation.apply(np.asarray(pcd_ori.points))
            #for i in range(len(pcd.points)):
            #    np.asarray(pcd_ori.points)[i][0] = temp_res[i][0]
            #    np.asarray(pcd_ori.points)[i][1] = temp_res[i][1]
            #    np.asarray(pcd_ori.points)[i][2] = temp_res[i][2]


            #data = np.asarray(pcd_ori.points)
            #print (ids)
            ##all_data.append(pcd)
            ##o3d.visualization.draw_geometries([pcd, mesh_frame])

            #centroid = np.array([normal_dict[ids][0],
            #                     normal_dict[ids][1],
            #                     normal_dict[ids][2]])
            #ranges = normal_dict[ids][3]

            #data = data - centroid

            #data = data/ranges
            #pcd_ori = o3d.geometry.PointCloud()
            #pcd_ori.points = o3d.utility.Vector3dVector(data)
            ##o3d.visualization.draw_geometries([pcd_ori, mesh_frame])
 
            ##np.save("test" + str(ids) + ".npy", np.asarray(pcd_ori.points))

            #print (np.asarray([pcd_ori.points]).shape)
            #data = self.completion_network.complete(np.asarray([pcd_ori.points]))
            #data = data*ranges
            #data = data + centroid
            #pcd_comp = o3d.geometry.PointCloud()
            #pcd_comp.points = o3d.utility.Vector3dVector(data)
 
            ##o3d.visualization.draw_geometries([pcd_comp, mesh_frame])

            #data = back_rotation.apply(data)
            #data += object_translation
            #pcd_comp = o3d.geometry.PointCloud()
            #pcd_comp.points = o3d.utility.Vector3dVector(data)
            #all_data.append(pcd_comp)
            ##o3d.visualization.draw_geometries([pcd_comp, mesh_frame])


        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.4, origin=[0, 0, 0])
        all_data.append(mesh_frame)

        o3d.visualization.draw_geometries(all_data)

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
    
    #status_dict = {}
    #start = 0
    #with open("track_object.txt") as f:
    #    for line in f:
    #        start += 1
    #        divs = line.split()
    #        status_dict[start] = [float(x) for x in divs]

    #normal_dict = {}
    #start = 0
    #ids = [2,5,6,7,9]
    #with open("normal.txt") as f:
    #    for line in f:
    #        divs = line.split()
    #        normal_dict[ids[start]] = [float(x) for x in divs]
    #        start += 1
    
    #print (normal_dict.items())
    extractor = point_cloud_extractor("../sim/poential_bug/color1.jpg",
                                      "../sim/poential_bug/depth1.png",
                                      "../sim/poential_bug/seg1.png",
                                      [0.5, 0, 0.4], 
                                      [0.0, 0.3508, 0.6139, 0.7071],
                                      )
