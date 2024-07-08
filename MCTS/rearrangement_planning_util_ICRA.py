#
# file   rearrangement_planning_util.py
# brief  Defines search tree node and related util functions
# author Hanwen Ren
# date   2023-03-08
#

import os
import sys
import math
import time
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdb
import fcl

#IROS 2024 modifications
#**************************************************************************************************
def get_euc_distance(cx, cy, rx, ry):
    return math.sqrt(sum([(x - y)**2 for x,y in zip([cx, cy], [rx, ry])]))

def smart_LMP_test():
    #test for smart LMP
    grid = []
    for i in range(20):
        temp_arr = []
        for j in range(20):
            temp_arr.append(0)
        grid.append(temp_arr)
    
    #source_config = [[10, 8, 1, 'r'], [15, 13, 1, 'b']]
    #target_config = [[10, 8, 1, 'r'], [4, 13, 1, 'b']]
    #goal_config = [[10, 10, 1, 'r'], [5, 15, 1, 'b']]

    source_config = [[7, 5, 1, 'y'], [10, 8, 1, 'b'], [10, 12, 1, 'g'], [10, 16, 1, 'r']]
    target_config = [[7, 5, 1, 'y'], [13, 9, 1, 'b'], [10, 12, 1, 'g'], [10, 16, 1, 'r']]
    goal_config = [[7, 5, 1, 'y'], [10, 8, 1, 'b'], [10, 12, 1, 'g'], [10, 16, 1, 'r']]

    source_treenode = Tree_Node(source_config, goal_config, grid, [])
    target_treenode = Tree_Node(target_config, goal_config, grid, [])

    #source_treenode.tunnel_and_normal_visualizer(animation = False)
    #target_treenode.tunnel_and_normal_visualizer(animation = False)

    smart_LMP_motion(source_treenode, target_treenode)


def find_tunnel_side_swipe(cx, cy, radius, rx, ry, robot_width):
    left_theta, right_theta = None, None

    #from the left side
    tri_a_length = radius + robot_width
    tri_c_length = get_euc_distance(cx, cy, rx, ry)
    if tri_c_length < tri_a_length:
        return 0, math.pi, 0
    tri_b_length = math.sqrt(tri_c_length**2 - tri_a_length**2)
    left_theta_whole = np.arcsin((cy - ry)/tri_c_length)
    left_theta_deduct = np.arcsin(min(1, (radius + robot_width)/tri_b_length))
    if cx >= rx:
        left_theta = left_theta_whole - left_theta_deduct
    else:
        left_theta = math.pi - (left_theta_whole + left_theta_deduct)

    #from the right side
    if cx >= rx:
        right_theta = left_theta_whole + left_theta_deduct
    else:
        right_theta = math.pi - (left_theta_whole - left_theta_deduct)

    #maximum length
    max_length_whole = tri_c_length - radius
    max_length = math.sqrt(max_length_whole**2 - robot_width**2)
    

    return left_theta, right_theta, max_length

def vector_rotation(cx, cy, rx, ry, angle):
    temp_x, temp_y = cx - rx, cy - ry

    temp_res =  [math.cos(angle) * temp_x - math.sin(angle) * temp_y,
                           math.sin(angle) * temp_x + math.cos(angle) * temp_y]
    
    return temp_res[0] + rx, temp_res[1] + ry


def smart_LMP_motion(source_treenode, target_treenode):
    step_size = 1.0
    step_rot = 5

    source_config = source_treenode.curr_config_
    target_config = target_treenode.curr_config_
    rx, ry = source_treenode.robot_
    robot_width = source_treenode.robot_width_

    move_index = None
    for i in range(len(source_config)):
        if source_config[i] != target_config[i]:
            move_index = i
            break
    
    if move_index == None:
        print("Error: unable to find move index!\n")
        sys.exit(1)


    grasp_tunnel = source_treenode.get_tunnel(source_treenode.robot_, source_config[move_index])
    relocate_tunnel = target_treenode.get_tunnel(target_treenode.robot_, target_config[move_index])

    source_x, source_y, _, _ = source_config[move_index]
    target_x, target_y, _, _ = target_config[move_index]

    grasp_angle_degree = grasp_tunnel[3]
    relocate_angle_degree = relocate_tunnel[3]
    grasp_depth = grasp_tunnel[1]
    relocate_depth = relocate_tunnel[1]

    smaller_angle = min(grasp_angle_degree, relocate_angle_degree)
    bigger_angle = max(grasp_angle_degree, relocate_angle_degree)

    sweep_depth = grasp_depth
    if relocate_depth < grasp_depth: sweep_depth = relocate_depth

    print(smaller_angle, bigger_angle)
    for i in range(len(source_config)):
        if i != move_index:
            cx, cy, radius, _ = source_config[i]
            left_theta, right_theta, max_length = find_tunnel_side_swipe(cx, cy, radius, rx, ry, robot_width)
            left_theta_degree = left_theta / math.pi * 180
            right_theta_degree = right_theta / math.pi * 180

            #be aware that the +-3 offset to compensate for the calculation error
            if right_theta_degree <= smaller_angle+3 or left_theta_degree >= bigger_angle-3:
                pass
            else:
                print(cx, cy, left_theta_degree, right_theta_degree)
                sweep_depth = min(sweep_depth, max_length)

    
    #plt.figure(figsize = (10, 10))

    #print(sweep_depth)

    #retract
    grasp_vector = [source_x - rx, source_y - ry]
    grasp_vector_length = math.sqrt(grasp_vector[0]**2 + grasp_vector[1]**2)
    grasp_vector_unit = [(source_x - rx)*1.0/(grasp_vector_length), (source_y - ry)*1.0/(grasp_vector_length)]
    new_source_x, new_source_y = rx + (sweep_depth - 1) * grasp_vector_unit[0], ry + (sweep_depth - 1) * grasp_vector_unit[1]

    retract_length = math.sqrt((source_x - new_source_x)**2 + (source_y - new_source_y)**2)
    num_steps = max(math.ceil(retract_length/step_size),1)
    delta_x, delta_y = (new_source_x - source_x) * 1.0 /num_steps, (new_source_y - source_y) * 1.0 /num_steps

    for i in range(num_steps + 1):
        plt.clf()
        temp_tunnel = source_treenode.get_tunnel(source_treenode.robot_, source_treenode.curr_config_[move_index])
        source_treenode.tunnel_and_normal_visualizer([temp_tunnel], animation = True)
        source_treenode.curr_config_[move_index][0] += delta_x
        source_treenode.curr_config_[move_index][1] += delta_y
    source_treenode.curr_config_[move_index][0] = new_source_x
    source_treenode.curr_config_[move_index][1] = new_source_y

    #rotate
    total_angle = bigger_angle - smaller_angle
    num_steps = max(math.ceil(total_angle/step_rot),1)
    delta_angle = (relocate_angle_degree - grasp_angle_degree)/num_steps
    delta_angle = delta_angle/180*math.pi

    for i in range(num_steps + 1):
        plt.clf()
        temp_tunnel = source_treenode.get_tunnel(source_treenode.robot_, source_treenode.curr_config_[move_index])
        source_treenode.tunnel_and_normal_visualizer([temp_tunnel], animation = True)
        new_x, new_y = vector_rotation(source_treenode.curr_config_[move_index][0],
                                                                     source_treenode.curr_config_[move_index][1],
                                                                     rx, ry, delta_angle)
        source_treenode.curr_config_[move_index][0] = new_x
        source_treenode.curr_config_[move_index][1] = new_y
    new_source_x, new_source_y = vector_rotation(new_source_x, new_source_y, rx, ry, (relocate_angle_degree - grasp_angle_degree)/180*math.pi)
    source_treenode.curr_config_[move_index][0] = new_source_x
    source_treenode.curr_config_[move_index][1] = new_source_y


    #send
    #grasp_vector_length = math.sqrt(grasp_vector[0]**2 + grasp_vector[1]**2)
    #grasp_vector_unit = [(source_x - rx)*1.0/(grasp_vector_length), (source_y - ry)*1.0/(grasp_vector_length)]
    #new_source_x, new_source_y = rx + (sweep_depth - 1) * grasp_vector_unit[0], ry + (sweep_depth - 1) * grasp_vector_unit[1]

    send_length = math.sqrt((target_x - new_source_x)**2 + (target_y - new_source_y)**2)
    num_steps = max(math.ceil(send_length/step_size),1)
    delta_x, delta_y = (target_x - new_source_x) * 1.0 /num_steps, (target_y - new_source_y) * 1.0 /num_steps

    for i in range(num_steps + 1):
        plt.clf()
        temp_tunnel = source_treenode.get_tunnel(source_treenode.robot_, source_treenode.curr_config_[move_index])
        source_treenode.tunnel_and_normal_visualizer([temp_tunnel], animation = True)
        source_treenode.curr_config_[move_index][0] += delta_x
        source_treenode.curr_config_[move_index][1] += delta_y
    source_treenode.curr_config_[move_index][0] = target_x
    source_treenode.curr_config_[move_index][1] =    target_y



    #sys.exit(1)
    
    #cx, cy, radius , _ = source_config[0]
    #left_theta, right_theta, max_length = find_tunnel_side_swipe(cx, cy, radius, rx, ry, robot_width)

    #swipe_tunnel_left = source_treenode.get_tunnel_from_angle(max_length, left_theta)
    #swipe_tunnel_right = source_treenode.get_tunnel_from_angle(max_length, right_theta)
    #source_treenode.tunnel_and_normal_visualizer([swipe_tunnel_left, swipe_tunnel_right], animation = False)

    
#**************************************************************************************************

class Tree_Node():
    def __init__(self, current_config, goal_config, current_grid, static_config = [], total_distance = 0, robot= [10.0, -2.0], swept_volume1=None, swept_volume2=None, obj_mesh=None, scale=None, target_pos=[]):
        self.reward_ = 0.0
        self.visited_ = 0.0
        self.parent_ = None
        self.children_ = []
        self.curr_config_ = current_config
        self.goal_config_ = goal_config
        self.static_config_ = static_config
        if target_pos:
            self.static_config_ += target_pos
        self.grid_ = current_grid
        self.robot_ = [0,0]
        self.robot_width_ = 2
        self.object_in_collision_ = None
        self.total_distance_ = total_distance
        self.x_min_ = int(-len(current_grid[0]) / 2)
        self.x_max_ = int(len(current_grid[0]) / 2)
        if scale is not None:
            self.y_min_ = int(0.3 / scale) + 2
        else:
            self.y_min_ = 0

        self.y_max_ = len(current_grid)
        self.swept_manager1 = swept_volume1
        self.swept_manager2 = swept_volume2
        self.obj_mesh = obj_mesh
        self.scale = scale
        self.final_obj_mesh = None

        if self.obj_mesh is not None:
            self.is_goal_config_swept()

        self.distance_lookup_ = defaultdict(list)
        for t in range(-len(self.grid_) + 1, len(self.grid_)):
            for k in range(-len(self.grid_[0]) + 1, len(self.grid_[0])):
                distance = round((t)**2 + (k)**2, 3)
                # if abs(distance) <= 0.0001:
                #     continue 
                self.distance_lookup_[distance].append([t, k])
        self.distance_lookup_ = [list(x) for x in self.distance_lookup_.items()]
        self.distance_lookup_.sort(key = lambda x: x[0])

    def add_reward(self, reward):
        self.reward_ += reward
        self.visited += 1.0

    #IROS 2024
    def get_tunnel_from_angle(self, width, angle):
        
        center_point = [self.robot_[0] + np.cos(angle)*width/2.0, self.robot_[1] + np.sin(angle)*width/2.0]
        v2_start = np.array(center_point)
        v2_end = np.array([center_point[0] + np.cos(angle)*width/2.0,
                                             center_point[1] + np.sin(angle)*width/2.0])
        v3_start = np.array(v2_start)
        v3_end = np.array([center_point[0] - np.sin(angle)*self.robot_width_,
                                             center_point[1] + np.cos(angle)*self.robot_width_])

        #bottom left point, width, height, rotation degree, normal1_start, normal1_end, normal2_start, normal2_end
        return [(self.robot_[0] + self.robot_width_*np.sin(angle), self.robot_[1] - self.robot_width_*np.cos(angle)), width, self.robot_width_*2, np.degrees(angle), v2_start, v2_end, v3_start, v3_end]

    def get_tunnel(self, start, goal):
        dy = goal[1] - start[1]
        dx = goal[0] - start[0]
        if dx != 0:
            angle = np.arctan(dy*1.0/dx)
            if angle < 0:
                angle += math.pi
        else:
            angle = math.pi/2

        width = np.sqrt(dy**2 + dx**2) + 1
        
        center_point = [self.robot_[0] + np.cos(angle)*width/2.0, self.robot_[1] + np.sin(angle)*width/2.0]
        v2_start = np.array(center_point)
        v2_end = np.array([center_point[0] + np.cos(angle)*width/2.0,
                                             center_point[1] + np.sin(angle)*width/2.0])
        v3_start = np.array(v2_start)
        v3_end = np.array([center_point[0] - np.sin(angle)*self.robot_width_,
                                             center_point[1] + np.cos(angle)*self.robot_width_])

        #bottom left point, width, height, rotation degree, normal1_start, normal1_end, normal2_start, normal2_end
        return [(self.robot_[0] + self.robot_width_*np.sin(angle), self.robot_[1] - self.robot_width_*np.cos(angle)), np.sqrt(dy**2 + dx**2)+1, self.robot_width_*2, np.degrees(angle), v2_start, v2_end, v3_start, v3_end]


    def collision_tunnel_static(self, tunnel):
        _, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end = tunnel
        angle = np.radians(angle_degree)
        
        v2 = v2_end - v2_start
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3_end - v3_start
        v3 = v3 / np.linalg.norm(v3)
        
        collision_items = set()
        
        if np.dot(v2, v3) > 1e-6:
            print('tunnel axes calculation fails\n')
            sys.exit(1)
        
        for i in range(len(self.static_config_)):
            cx, cy, radius, color = self.static_config_[i]
            test_vector = np.array([cx, cy]) - v2_start
            proj_v2 = np.dot(v2, test_vector)
            proj_v3 = np.dot(v3, test_vector)
        
            if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
                 -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
                # print("true")
                return True
            
        # print("false")

        # #the goal of current MCTS objective
        # cx, cy, radius, color = self.goal_config_[0]
        # test_vector = np.array([cx, cy]) - v2_start
        # proj_v2 = np.dot(v2, test_vector)
        # proj_v3 = np.dot(v3, test_vector)
        
        # if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
        #      -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
        #     return True
            
        return False

    def collision_tunnel_object_spec_item(self, tunnel, item_index):
        _, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end = tunnel
        angle = np.radians(angle_degree)

        v2 = v2_end - v2_start
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3_end - v3_start
        v3 = v3 / np.linalg.norm(v3)

        collision_items = set()

        if np.dot(v2, v3) > 1e-6:
            print('tunnel axes calculation fails\n')
            sys.exit(1)
        
        cx, cy, radius, color = self.goal_config_[item_index]
        test_vector = np.array([cx, cy]) - v2_start
        proj_v2 = np.dot(v2, test_vector)
        proj_v3 = np.dot(v3, test_vector)

        if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
             -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
            return True
            
        return False

    def get_dependency_relation(self):
        res = defaultdict(set)
        for i in range(len(self.goal_config_)):
            relocate_tunnel = self.get_tunnel(self.robot_, self.goal_config_[i]) # fix1??
            collision_objs = self.collision_tunnel_object_goal(relocate_tunnel) # fix2??
            collision_objs = [x for x in collision_objs if x != i]
            for obj in collision_objs:
                res[obj].add(i)

        return res

    def collision_tunnel_object_goal(self, tunnel):
        _, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end = tunnel
        angle = np.radians(angle_degree)

        v2 = v2_end - v2_start
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3_end - v3_start
        v3 = v3 / np.linalg.norm(v3)

        collision_items = set()

        if np.dot(v2, v3) > 1e-6:
            print('tunnel axes calculation fails\n')
            sys.exit(1)
        
        for i in range(len(self.goal_config_)):
            cx, cy, radius, color = self.goal_config_[i]
            test_vector = np.array([cx, cy]) - v2_start
            proj_v2 = np.dot(v2, test_vector)
            proj_v3 = np.dot(v3, test_vector)

            if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
               -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
                collision_items.add(i)
            
        return list(collision_items)


    def collision_tunnel_object(self, tunnel):
        _, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end = tunnel
        angle = np.radians(angle_degree)

        v2 = v2_end - v2_start
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3_end - v3_start
        v3 = v3 / np.linalg.norm(v3)

        collision_items = set()

        if np.dot(v2, v3) > 1e-6:
            print('tunnel axes calculation fails\n')
            sys.exit(1)
        
        for i in range(len(self.curr_config_)):
            cx, cy, radius, color = self.curr_config_[i]
            test_vector = np.array([cx, cy]) - v2_start
            proj_v2 = np.dot(v2, test_vector)
            proj_v3 = np.dot(v3, test_vector)

            if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
               -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
                collision_items.add(i)
            
        return list(collision_items)

    def collision_tunnel_region(self, tunnel, new_region):
        _, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end = tunnel
        angle = np.radians(angle_degree)
    
        v2 = v2_end - v2_start
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3_end - v3_start
        v3 = v3 / np.linalg.norm(v3)

        if np.dot(v2, v3) > 1e-6:
            print('tunnel axes calculation fails\n')
            sys.exit(1)

        cx, cy = new_region[0], new_region[1]
        test_vector = np.array([cx, cy]) - v2_start
        proj_v2 = np.dot(v2, test_vector)
        proj_v3 = np.dot(v3, test_vector)

        if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
             -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
            return True
        else:
            return False


    def tunnel_and_normal_visualizer(self, tunnel_list = None, object_in_collision = None, true_color = False, animation = False):
        object_in_collision = self.check_collision_w_swept()

        region_x, region_y = [], []
        for i in range(len(self.grid_)):
            for j in range(len(self.grid_[0])):
                region_x.append(i)
                region_y.append(j)
        if animation == False:
            plt.figure(figsize = (len(self.grid_[0]), len(self.grid_)))
        plt.scatter(region_x, region_y)
        for cx, cy, radius, color in self.curr_config_:
            temp_circle = mpatches.Circle((cx, cy), radius, color = color)
            plt.gca().add_patch(temp_circle)
        for cx, cy, radius, color in self.static_config_:
            if not true_color:
                temp_circle = mpatches.Circle((cx, cy), radius, color = 'black')
                temp_circle_inner = mpatches.Circle((cx, cy), radius*0.7, color =  color)
                plt.gca().add_patch(temp_circle)
                plt.gca().add_patch(temp_circle_inner)
            else:
                temp_circle = mpatches.Circle((cx, cy), radius, color = color)
                plt.gca().add_patch(temp_circle)

        robot = mpatches.Rectangle((self.robot_[0] - 0.5, self.robot_[1] - 0.5), 1, 1)
        plt.gca().add_patch(robot)

        tunnel_counter = 0
        tunnel_color = ['b', 'r']

        if tunnel_list:
            for start_corner, width, height, angle, v2_start, v2_end, v3_start, v3_end in tunnel_list:
                tunnel_shape = mpatches.Rectangle(start_corner, width, height, angle, alpha = 0.5, color = tunnel_color[tunnel_counter])
                plt.gca().add_patch(tunnel_shape)
                plt.plot([v2_start[0], v2_end[0]], [v2_start[1], v2_end[1]], color = 'r')
                plt.plot([v3_start[0], v3_end[0]], [v3_start[1], v3_end[1]], color = 'r')
                tunnel_counter += 1

        if object_in_collision:
            for index in object_in_collision:
                # if index != 0:
                cx, cy, radius, color = self.curr_config_[index]
                temp_square = mpatches.Rectangle((cx - radius, cy - radius), radius*2, radius*2, alpha = 0.3, color = 'black')
                plt.gca().add_patch(temp_square)

        plt.xlim(-len(self.grid_[0])/2, len(self.grid_[0])/2)
        plt.ylim(0, len(self.grid_))
        if animation == False:
            plt.show()
        else:
            plt.pause(0.1)

    def tunnel_and_normal_visualizer_v2(self, tunnel_list = None, object_in_collision = None, true_color = False, animation = False):
        region_x, region_y = [], []
        for i in range(len(self.grid_)):
            for j in range(len(self.grid_[0])):
                region_x.append(i)
                region_y.append(j)
        if animation == False:
            plt.figure(figsize = (8.5, 18))
        plt.scatter(region_x, region_y)
        for cx, cy, radius, color in self.curr_config_:
            temp_circle = mpatches.Circle((cx, cy), radius, color = color)
            plt.gca().add_patch(temp_circle)
        for cx, cy, radius, color in self.static_config_:
            if not true_color:
                temp_circle = mpatches.Circle((cx, cy), radius, color = 'black')
                temp_circle_inner = mpatches.Circle((cx, cy), radius*0.7, color =  color)
                plt.gca().add_patch(temp_circle)
                plt.gca().add_patch(temp_circle_inner)
            else:
                temp_circle = mpatches.Circle((cx, cy), radius, color = color)
                plt.gca().add_patch(temp_circle)

        robot = mpatches.Rectangle((self.robot_[0] - 0.5, self.robot_[1] - 0.5), 1, 1)
        plt.gca().add_patch(robot)

        tunnel_counter = 0
        tunnel_color = ['b', 'r']

        if tunnel_list:

            for start_corner, width, height, angle, v2_start, v2_end, v3_start, v3_end in tunnel_list:
                tunnel_shape = mpatches.Rectangle(start_corner, width, height, angle, alpha = 0.5, color = 'r')
                plt.gca().add_patch(tunnel_shape)
                plt.plot([v2_start[0], v2_end[0]], [v2_start[1], v2_end[1]], color = 'r')
                plt.plot([v3_start[0], v3_end[0]], [v3_start[1], v3_end[1]], color = 'r')
                tunnel_counter += 1

        if object_in_collision:
            for index in object_in_collision:
                if index != 0:
                    cx, cy, radius, color = self.curr_config_[index]
                    temp_square = mpatches.Rectangle((cx - radius, cy - radius), radius*2, radius*2, alpha = 0.5, color = 'm')
                    plt.gca().add_patch(temp_square)

        plt.xlim(-2, 20)
        plt.ylim(-4, 20)
        if animation == False:
            plt.show()
        else:
            plt.pause(0.01)

    def check_collision_w_swept(self):
        collision_obj_list = []
        for obj_idx in range(len(self.obj_mesh)):
            # read collision mesh
            temp_verts, temp_tris = self.obj_mesh[obj_idx] # need mesh
            temp_m = fcl.BVHModel()
            temp_m.beginModel(len(temp_verts), len(temp_tris))
            temp_m.addSubModel(temp_verts, temp_tris)
            temp_m.endModel()
            temp_t = fcl.Transform()

            # check collision
            req = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request = req)
            self.swept_manager1.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
            is_collision1 = rdata.result.is_collision
            self.swept_manager2.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
            is_collision2 = rdata.result.is_collision

            if is_collision1 or is_collision2:
                collision_obj_list.append(obj_idx)

        # print("Swept check:", collision_obj_list)
        return collision_obj_list
    
    def get_blocking_object_swept(self):
        self.object_in_collision_ = self.check_collision_w_swept()

    def get_blocking_object(self):
        start_pose = self.curr_config_[0]
        goal_pose = self.goal_config_[0]

        grasp_tunnel = self.get_tunnel(self.robot_, start_pose)
        relocate_tunnel = self.get_tunnel(self.robot_, goal_pose)

        object_in_collision = self.collision_tunnel_object(grasp_tunnel)

        object_in_collision += self.collision_tunnel_object(relocate_tunnel)
        
        #another kind of collision is that placing the object blocks reaching of some other objects
        for i in range(len(self.curr_config_)):
            object_current_pose = self.curr_config_[i]
            object_current_grasp_tunnel = self.get_tunnel(self.robot_, object_current_pose)
            flag = self.collision_tunnel_object_spec_item(object_current_grasp_tunnel, 0)
            if flag:
                object_in_collision.append(i)

        self.object_in_collision_ = list(set(object_in_collision))

        self.object_in_collision_ = [x for x in self.object_in_collision_ if x != 0]

        # self.tunnel_and_normal_visualizer([grasp_tunnel, relocate_tunnel], self.object_in_collision_)

    def test_new_region_blocking(self, new_region, obs = None):
        new_region_relocate_tunnel = self.get_tunnel(self.robot_, new_region)
        flag4 = self.collision_tunnel_static(new_region_relocate_tunnel)
        flag3 = False

        if obs:
            for ob in obs:
                obs_start_pose = self.curr_config_[ob]
                obs_grasp_tunnel = self.get_tunnel(self.robot_, obs_start_pose)
                flag3 = self.collision_tunnel_region(obs_grasp_tunnel, new_region)
                if flag3: return True

        if flag3 or flag4:
            return True
        else:
            return False
        
    def convert2global(self, pos):
        new_pos = [pos[1] * self.scale, -pos[0] * self.scale, 0]
        return new_pos

    def get_feasible(self, idx, new_pos):
        curr_config = deepcopy(self.curr_config_)
        curr_feasible = set(self.random_object_selection())
        self.curr_config_[idx][0] = new_pos[0]
        self.curr_config_[idx][1] = new_pos[1]
        new_feasible = set(self.random_object_selection())
        self.curr_config_ = deepcopy(curr_config)

        new_element = []
        for i in new_feasible:
            if i not in curr_feasible:
                new_element.append(i)

        # return new_element, len(new_feasible)
        return new_element, len(new_element)

    def get_new_collision_swept(self, idx, transform, skip_flag):
        if skip_flag: return False

        g_transform = self.convert2global(transform)

        # read collision mesh
        temp_verts, temp_tris = self.obj_mesh[idx] # need mesh
        temp_m = fcl.BVHModel()
        temp_m.beginModel(len(temp_verts), len(temp_tris))
        temp_m.addSubModel(temp_verts, temp_tris)
        temp_m.endModel()
        temp_t = fcl.Transform(g_transform)
        # check collision
        req = fcl.CollisionRequest()
        rdata = fcl.CollisionData(request = req)
        self.swept_manager1.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
        is_collision1 = rdata.result.is_collision
        self.swept_manager2.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
        is_collision2 = rdata.result.is_collision

        return is_collision1 or is_collision2
    
    def propose_new_region(self, index, obs, random_obj_flag=False):
        swept_off_flag = False
        if random_obj_flag:
            swept_off_flag = random.choice([True, False])
            # swept_off_flag = True
            num_feasible_list = []

        gx, gy, radius, color = self.curr_config_[index]
        res = []
        for distance, offset_list in self.distance_lookup_:
            for ox, oy in offset_list:
                #change for IROS 2024, add a 2D gaussian offset to change the discrete region proposal
                #to continuous. The covariance matrix is [[R, 0], [0, R]]
                temp_x = gx + ox + round(random.gauss(0, 1),2)
                temp_y = gy + oy + round(random.gauss(0, 1),2)

                #may delete
                if self.x_min_ <= gx + ox <= self.x_max_ and self.y_min_ <= gy + oy <= self.y_max_:
                   while (temp_x < self.x_min_ or temp_x > self.x_max_ or temp_y < self.y_min_ or temp_y > self.y_max_):
                       temp_x = gx + ox + round(random.gauss(0, 1),2)
                       temp_y = gy + oy + round(random.gauss(0, 1),2)

                ox_rand = temp_x - gx
                oy_rand = temp_y - gy

                if (self.x_min_ <= temp_x <= self.x_max_) and \
                   (self.y_min_ <= temp_y <= self.y_max_) and \
                    self.dst_region_collision_free(index, [temp_x, temp_y]) and \
                    not self.test_new_region_blocking([temp_x, temp_y], obs) and \
                    not self.get_new_collision_swept(index, [ox_rand, oy_rand], swept_off_flag):

                    relocate_tunnel = self.get_tunnel(self.robot_, [temp_x, temp_y])
                    collision_object = self.collision_tunnel_object(relocate_tunnel)

                    collision_object = [x for x in collision_object if x != index]
                    if not collision_object and not self.collision_tunnel_static(relocate_tunnel):
                        if swept_off_flag:
                            new_element, num_feasible = self.get_feasible(index, [temp_x, temp_y])

                            if new_element:
                                res.append([temp_x, temp_y])
                                num_feasible_list.append(num_feasible)

                            if len(res) == 4:
                                # print(num_feasible_list)
                                # print(res)
                                max_val = max(num_feasible_list)
                                max_idx = num_feasible_list.index(max_val)
                                res1 = res.pop(max_idx)
                                num_feasible_list.pop(max_idx)
                                max_val = max(num_feasible_list)
                                max_idx = num_feasible_list.index(max_val)
                                return[res1, res[max_idx]]
                        else:
                            res.append([temp_x, temp_y])
                        
                        if len(res) == 4:
                            random.shuffle(res)
                            if random_obj_flag:
                                return res[0:2]
                            return res
                        
        if swept_off_flag and len(num_feasible_list) > 2:
            max_val = max(num_feasible_list)
            max_idx = num_feasible_list.index(max_val)
            res1 = res.pop(max_idx)
            num_feasible_list.pop(max_idx)
            max_val = max(num_feasible_list)
            max_idx = num_feasible_list.index(max_val)
            return[res1, res[max_idx]]
        
        if random_obj_flag:
            random.shuffle(res)
            return res[0:2]
        return res

    def random_object_selection(self):
        res = []
        for i in range(len(self.curr_config_)):
            temp_x, temp_y, _, _ = self.curr_config_[i]
            grasp_tunnel = self.get_tunnel(self.robot_, [temp_x, temp_y])
            collision_object = self.collision_tunnel_object(grasp_tunnel)
            static_collision = self.collision_tunnel_static(grasp_tunnel)
            collision_object = [x for x in collision_object if x != i]
            if not collision_object and not static_collision:
                res.append(i)
        random.shuffle(res)
        return res
                
    def add_child(self, child):
        self.children_.append(child)

    def set_parent(self, parent):
        self.parent_ = parent
    
    def get_UCB(self):
        if self.parent_:
            if self.visited_ != 0:
                return (self.reward_/self.visited_) + 2*math.sqrt(math.log(self.parent_.visited_)/self.visited_)
            else:
                return sys.maxsize
        else:
            print ('Something wrong here')
            sys.exit(1)
    
    def is_goal_config_swept(self):
        collision = self.check_collision_w_swept()
        if collision:
            self.object_in_collision_ = collision
            return False
        
        print("True!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return True

    def is_goal_config(self):
        return self.curr_config_[0] == self.goal_config_[0]

    def traverse_tree(self):
        res = []
        while self:
            res.append(self)
            self = self.parent_
        return res[::-1]

    def dst_region_collision_free(self, index, proposed_region):
        obj_x, obj_y, obj_radius, obj_color = self.curr_config_[index]
        if self.static_config_:
            for cx, cy, radius, color in self.static_config_:
                distance = np.sqrt((proposed_region[0] - cx)**2 + (proposed_region[1] - cy)**2)
                if distance < obj_radius + radius + 2:
                    return False
        for i in range(len(self.curr_config_)):
            if i != index:
                cx, cy, radius, color = self.curr_config_[i]
                distance = np.sqrt((proposed_region[0] - cx)**2 + (proposed_region[1] - cy)**2)
                if distance < obj_radius + radius + 2:
                    # print("dst_region_collision failed", i)
                    return False
        return True


def write_result(method, test_index, object_count, time, steps, length, displacement, res_plan):
    file_name = 'test_results_bound/' + method + '/test_result_' + str(test_index) + '.txt'
    plan_name = 'test_results_bound/' + method + '/test_result_' + str(test_index) + '.npy'
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as f:
        f.write('number of objects : ' + str(object_count) + '\n')
        f.write('time comsumption : ' + str(time) + '\n')
        f.write('number of steps : ' + str(steps) + '\n')
        f.write('total length travelled : ' + str(length) + '\n')
        f.write('total length displacement : ' + str(displacement) + '\n')
    f.close()

    with open(plan_name, 'wb') as f:
        np.save(f, res_plan)
    




def regression_test(case_index):
    #any newly implemeted version need to pass these regression test
    #case_index: 1 - four object rotation counterclockwise case
    #            2 - nested rearrangement that plRS fails to solve
    #            3 - tower of Hanoi
    #            4 - variant of tower of Hanoi
    #            5 - NeRP-like case
    #            6 - variant of NeRP-like case
  #            7 - monotone case for sanity check
    #            8 - slightly harder monotone case


    curr_config, goal_config = None, None
    if case_index == 1:
        curr_config = [[5, 3, 1, 'r'], [15, 7, 1, 'g'], [15, 17, 1, 'b'], [5, 13, 1, 'y']]
        goal_config = [[15, 7, 1, 'r'], [15, 17, 1,  'g'], [5, 13, 1, 'b'], [5, 3, 1, 'y']]
    elif case_index == 2:
        curr_config = [[10, 5, 1, 'r'], [10, 15, 1, 'g']]
        goal_config = [[10, 15, 1, 'r'], [10, 5, 1, 'g']]
    elif case_index == 3:
        curr_config = [[4, 4, 1, 'r'], [4, 10, 1, 'g'], [4, 16, 1, 'b']]
        goal_config = [[16, 4, 1, 'r'], [16, 10, 1, 'g'], [16, 16, 1, 'b']]
    elif case_index == 4:
        curr_config = [[4, 4, 1, 'r'], [4, 10, 1, 'g'], [4, 16, 1, 'b']]
        goal_config = [[16, 10, 1, 'r'], [16, 16, 1, 'g'], [16, 4, 1, 'b']]
    elif case_index == 5:
        curr_config = [[5, 15, 1, 'r'], [15, 15, 1, 'b'], [10, 10, 1, 'g'], [5, 5, 1, 'y'], [15, 5, 1, 'c']]
        goal_config = [[6, 17, 1, 'r'], [12, 12, 1, 'b'], [16, 4, 1, 'g'], [14, 8, 1, 'y'], [4, 14, 1, 'c']]
    elif case_index == 6:
        curr_config = [[5, 15, 1, 'r'], [15, 15, 1, 'b'], [10, 10, 1, 'g'], [5, 5, 1, 'y'], [15, 5, 1, 'c']]
        goal_config = [[14, 17, 1, 'r'], [12, 12, 1, 'b'], [16, 4, 1, 'g'], [14, 8, 1, 'y'], [4, 14, 1, 'c']]
    elif case_index == 7:
        curr_config = [[5, 15, 1, 'r'], [15, 15, 1, 'g']]
        goal_config = [[5, 5, 1, 'r'], [15, 5, 1, 'g']]
    elif case_index == 8:
        curr_config = [[5, 15, 1, 'r'], [15, 5, 1, 'g'], [10, 15, 1, 'b']]
        goal_config = [[5, 5, 1, 'r'], [15, 15, 1, 'g'], [10, 5, 1, 'b']]
    elif case_index == 9:
        curr_config = [[10, 16, 1, 'r'], [10, 12, 1, 'g'], [10, 8, 1, 'b'], [10, 4, 1, 'y']]
        goal_config = [[10, 4, 1, 'r'], [10, 8, 1, 'g'], [10, 12, 1, 'b'], [10, 16, 1, 'y']]
    else:
        #default case: four object rotate counterclosewise case
        curr_config = [[5, 3, 1, 'r'], [15, 7, 1, 'g'], [15, 17, 1, 'b'], [5, 13, 1, 'y']]
        goal_config = [[15, 7, 1, 'r'], [15, 17, 1,  'g'], [5, 13, 1, 'b'], [5, 3, 1, 'y']]

    return curr_config, goal_config

if __name__ == '__main__':
    smart_LMP_test()

