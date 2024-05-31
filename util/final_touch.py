import open3d as o3d
import numpy as np
import os
import sys
import math
#from YCB_object import YCB_object

def get_recon(file1, file2):
    point_file = file1
    color_file = file2

    xlimit = [0, 1]
    ylimit = [-1, 1]
    ylimit2 = [-1.3, 1.3]
    zlimit = [0.14, 1]
    zlimit2 = [-0.5, 1.1]

    point_data = np.load(point_file)
    color_data = np.load(color_file)

    new_point = []
    new_color = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_point.append([xlimit[i], ylimit2[j], zlimit2[k]])
                new_color.append([1, 0, 0])

    #for i in range(2):
    #    for j in range(2):
    #        for k in range(2):
    #            new_point.append([xlimit[i], ylimit[j], zlimit2[k]])
    #            new_color.append([1, 1, 1])


    for i in range(len(point_data)):
        x,y,z = point_data[i]
        if xlimit[0] <= x <= xlimit[1] and \
           ylimit[0] <= y <= ylimit[1] and \
           zlimit[0] <= z <= zlimit[1]:
               new_point.append(list(point_data[i]))
               new_color.append(list(color_data[i]))

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(new_point)
    pc.colors = o3d.utility.Vector3dVector(new_color)

    

    env_points = []
    env_colors = []

    for t in range(400, 1000):
        for k in range(-460, 400):
            env_points.append([t*0.001, k*0.001, 0.14])
            env_colors.append([0.5, 0.5, 0.5])

    for t in range(100, 200):
        for k in range(100, 400):
            env_points.append([1, t*0.001, k*0.001])
            env_colors.append([0.5, 0.5, 0.5])

    
    env_pc = o3d.geometry.PointCloud()
    env_pc.points = o3d.utility.Vector3dVector(env_points)
    env_pc.colors = o3d.utility.Vector3dVector(env_colors)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)
    vis.add_geometry(env_pc)
    control = vis.get_view_control()
    control.set_zoom(0.2)
    control.set_front([-2, 0, 0.6])
    control.set_lookat([0, 0, 0.5])
    control.set_up([0, 0, 1])
    #o3d.visualization.ViewControl.set_zoom(0.8)
    vis.run()

def get_object(file1, file2, file3 = None, file4 = None):

    point_file = file1
    color_file = file2
    print (file3)
    if file3:
        point_file2 = file3
        color_file2 = file4

    xlimit = [0.3, 1]
    ylimit = [-0.5, 0.5]
    ylimit2 = [-1, 1]
    zlimit = [0.0, 0.14 + 0.5]
    zlimit2 = [0, 0.8]

    point_data = np.load(point_file)
    color_data = np.load(color_file)
    if file3:
        point_data2 = np.load(point_file2)
        color_data2 = np.load(color_file2)

    new_point = []
    new_color = []
    new_point2 = []
    new_color2 = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_point.append([xlimit[i], ylimit2[j], zlimit2[k]])
                new_color.append([1, 0, 0])

    #for i in range(2):
    #    for j in range(2):
    #        for k in range(2):
    #            new_point.append([xlimit[i], ylimit[j], zlimit2[k]])
    #            new_color.append([1, 1, 1])


    for i in range(len(point_data)):
        x,y,z = point_data[i]
        if xlimit[0] <= x <= xlimit[1] and \
           ylimit[0] <= y <= ylimit[1] and \
           zlimit[0] <= z <= zlimit[1]:
               new_point.append(list(point_data[i]))
               new_color.append(list(color_data[i]))

    if file3:
        for i in range(len(point_data2)):
            x,y,z = point_data2[i]
            if xlimit[0] <= x <= xlimit[1] and \
               ylimit[0] <= y <= ylimit[1] and \
               zlimit[0] <= z <= zlimit[1]:
                   new_point2.append(list(point_data2[i]))
                   new_color2.append(list(color_data2[i]))


    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(new_point)
    pc.colors = o3d.utility.Vector3dVector(new_color)

    if file3:
        pc2 = o3d.geometry.PointCloud()
        pc2.points = o3d.utility.Vector3dVector(new_point2)
        pc2.colors = o3d.utility.Vector3dVector(new_color2)
        print (pc2)

    env_points = []
    env_colors = []

    for t in range(320, 1500):
        for k in range(-460, 460):
            env_points.append([t*0.001, k*0.001, 0.022])
            env_colors.append([0.77, 0.77, 0.77])


    #for t in range(350, 1500):
    #    for k in range(-200, 460):
    #        target_i = 0.8 - (k + 200)*(0.5/370)
    #        if t*0.001 >= target_i:
    #            env_points.append([t*0.001, k*0.001, 0.01])
    #            env_colors.append([0.22, 0.22, 0.22])
    
    #for t in range(350, 1500):
    #    for k in range(-460, 200):
    #        target_i = 0.3 + (k + 460)*(0.22/660)
    #        if t*0.001 >= target_i:
    #            env_points.append([t*0.001, k*0.001, 0.01])
    #            env_colors.append([0.22, 0.22, 0.22])

    #for t in range(300, 1500):
    #    for k in range(-500, 500):
    #        target_i = abs(k-460)/920*0.0005*1200 + 0.2
    #        if t*0.001 >= target_i:
    #            env_points.append([t*0.001, k*0.001, 0.14])
    #            env_colors.append([0.44, 0.44, 0.44])

    #for t in range(800, 1600):
    #    for k in range(100, 600):
    #        env_points.append([t*0.001, 0.50,  k*0.001])
    #        env_colors.append([0.44, 0.44, 0.44])

    
    env_pc = o3d.geometry.PointCloud()
    env_pc.points = o3d.utility.Vector3dVector(env_points)
    env_pc.colors = o3d.utility.Vector3dVector(env_colors)

    env_pc = o3d.geometry.PointCloud()
    env_pc.points = o3d.utility.Vector3dVector(env_points)
    env_pc.colors = o3d.utility.Vector3dVector(env_colors)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)
    vis.add_geometry(env_pc)
    if file3:
        vis.add_geometry(pc2)
    control = vis.get_view_control()
    control.set_zoom(0.2)
    control.set_front([-2, 0, 0.6])
    control.set_lookat([0, 0, 0.5])
    control.set_up([0, 0, 1])
    #o3d.visualization.ViewControl.set_zoom(0.8)
    vis.run()

def get_cover(file1):

    point_file = file1
    #color_file = file2

    xlimit = [0, 1]
    ylimit = [-1, 1]
    ylimit2 = [-1.3, 1.3]
    zlimit = [0.0, 1]
    zlimit2 = [-0.5, 1.1]

    point_data = np.load(point_file)
    #color_data = np.load(color_file)
    color_data = np.tile(np.array([1, 0, 0]), (len(point_data), 1))

    new_point = []
    new_color = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_point.append([xlimit[i], ylimit2[j], zlimit2[k]])
                new_color.append([1, 0, 0])

    #for i in range(2):
    #    for j in range(2):
    #        for k in range(2):
    #            new_point.append([xlimit[i], ylimit[j], zlimit2[k]])
    #            new_color.append([1, 1, 1])


    for i in range(len(point_data)):
        x,y,z = point_data[i]
        if xlimit[0] <= x <= xlimit[1] and \
           ylimit[0] <= y <= ylimit[1] and \
           zlimit[0] <= z <= zlimit[1]:
               new_point.append(list(point_data[i]))
               new_color.append(list(color_data[i]))

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(new_point)
    pc.colors = o3d.utility.Vector3dVector(new_color)

    

    env_points = []
    env_colors = []

    #for t in range(400, 1000):
    #    for k in range(-460, 400):
    #        env_points.append([t*0.001, k*0.001, 0.14])
    #        env_colors.append([0.8, 0.8, 0.8])
    
    env_pc = o3d.geometry.PointCloud()
    env_pc.points = o3d.utility.Vector3dVector(env_points)
    env_pc.colors = o3d.utility.Vector3dVector(env_colors)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)
    vis.add_geometry(env_pc)
    control = vis.get_view_control()
    control.set_zoom(0.2)
    control.set_front([-2, 0, 0.6])
    control.set_lookat([0, 0, 0.5])
    control.set_up([0, 0, 1])
    #o3d.visualization.ViewControl.set_zoom(0.8)
    vis.run()



if __name__ == '__main__':

    #get_recon(sys.argv[1], sys.argv[2])
    get_object(sys.argv[1], sys.argv[2])
    get_object(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    #get_cover(sys.argv[1])

    #print (o3d.__version__)
    #point_file = sys.argv[1]
    #color_file = sys.argv[2]

    #xlimit = [0, 1]
    #ylimit = [-1, 1]
    #ylimit2 = [-1.3, 1.3]
    #zlimit = [0.14, 1]
    #zlimit2 = [-0.5, 1.1]

    #point_data = np.load(point_file)
    #color_data = np.load(color_file)

    #new_point = []
    #new_color = []

    #for i in range(2):
    #    for j in range(2):
    #        for k in range(2):
    #            new_point.append([xlimit[i], ylimit2[j], zlimit2[k]])
    #            new_color.append([1, 0, 0])

    ##for i in range(2):
    ##    for j in range(2):
    ##        for k in range(2):
    ##            new_point.append([xlimit[i], ylimit[j], zlimit2[k]])
    ##            new_color.append([1, 1, 1])


    #for i in range(len(point_data)):
    #    x,y,z = point_data[i]
    #    if xlimit[0] <= x <= xlimit[1] and \
    #       ylimit[0] <= y <= ylimit[1] and \
    #       zlimit[0] <= z <= zlimit[1]:
    #           new_point.append(list(point_data[i]))
    #           new_color.append(list(color_data[i]))

    #pc = o3d.geometry.PointCloud()
    #pc.points = o3d.utility.Vector3dVector(new_point)
    #pc.colors = o3d.utility.Vector3dVector(new_color)

    #

    #env_points = []
    #env_colors = []

    #for t in range(400, 1000):
    #    for k in range(-460, 400):
    #        env_points.append([t*0.001, k*0.001, 0.14])
    #        env_colors.append([0.5, 0.5, 0.5])
    #
    #env_pc = o3d.geometry.PointCloud()
    #env_pc.points = o3d.utility.Vector3dVector(env_points)
    #env_pc.colors = o3d.utility.Vector3dVector(env_colors)


    #vis = o3d.visualization.Visualizer()
    #vis.create_window()
    #vis.add_geometry(pc)
    #vis.add_geometry(env_pc)
    #control = vis.get_view_control()
    #control.set_zoom(0.2)
    #control.set_front([-2, 0, 0.6])
    #control.set_lookat([0, 0, 0.5])
    #control.set_up([0, 0, 1])
    ##o3d.visualization.ViewControl.set_zoom(0.8)
    #vis.run()


    ##o3d.visualization.draw_geometries([pc, env_pc], zoom=0.1)
