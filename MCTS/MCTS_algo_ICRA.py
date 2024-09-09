#
# file:   MCTS_algo.py
# Brief:  Implementation of ML-MCTS algorithm
# Author: Hanwen Ren --- Jun
# Date:   2023-03-02
#

import os
import sys
import math
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rearrangement_planning_util_ICRA import Tree_Node
from rearrangement_planning_util_ICRA import smart_LMP_motion
import fcl
import pdb


#main 2D ML-MCTS class that internally calls MCTS class
class multi_level_MCTS_algo():
    def __init__(self, curr_config, goal_config, scene_info=None, swept_volume1=None, swept_volume2=None, obj_mesh=None, target_pos=None, unknown_area=[], valid_area=[], potential_centers=[]):
        # init update
        self.curr_config_ = curr_config
        self.goal_config_ = goal_config
        self.target_pos = target_pos
        self.scene_info = scene_info

        self.unknown_area = unknown_area
        self.valid_area = valid_area
        self.potential_centers=potential_centers

        self.obj_mesh = obj_mesh
        self.swept_volume1 = swept_volume1
        self.swept_volume2 = swept_volume2

        # after run update
        self.MCTS_ins = None
        self.grid_ = None
        self.scale = None
        self.is_scenario_checked = False

        self.total_length_travelled_ = 0.0
        self.total_length_displacement_ = 0.0
        self.track_level_steps_ = None

    def init_MCTS(self):
        MCTS_ins = MCTS_algo(deepcopy(self.curr_config_), deepcopy(self.goal_config_), self.scene_info, swept_volume1=self.swept_volume1, swept_volume2=self.swept_volume2, obj_mesh=self.obj_mesh, target_pos=self.target_pos, unknown_area=self.unknown_area, valid_area=self.valid_area, potential_centers=self.potential_centers)
        self.MCTS_ins = MCTS_ins
        # MCTS_ins.MCTS_tree_.tunnel_and_normal_visualizer(unknown_show=True)

    def run_mcts(self, time_limit=None):
        start_time = time.time()
        track_level_steps = []
        total_steps = 0
        MCTS_ins = self.MCTS_ins
        MCTS_ins.start_time = start_time
        MCTS_ins.time_limit = time_limit

        level_steps = MCTS_ins.exec_algo()

        if level_steps is None:
            curr_time = time.time()
            self.time_consumption_ = curr_time - start_time
            print("planning failed. Time consumed", self.time_consumption_)
            children_list = MCTS_ins.get_children_nodes(MCTS_ins.root_)
            return False, children_list

        track_level_steps.append(level_steps)
        total_steps += len(level_steps)
        end_time = time.time()

        self.curr_config_ = MCTS_ins.curr_config_
        self.goal_config_ = MCTS_ins.goal_config_
        self.grid_ = MCTS_ins.grid_
        self.scale = MCTS_ins.scale
        self.track_level_steps_ = track_level_steps

        self.total_steps_ = len(self.track_level_steps_[0]) - 1
        print("total steps", self.total_steps_)

        self.calculate_total_length_displacement()
        self.calculate_total_length_travelled()

        self.time_consumption_ = end_time - start_time
        print("Time consumption:", self.time_consumption_)
        return True, self.time_consumption_

    def scenario_check(self):
        if self.is_scenario_checked:
            return

        print("checking scenario")
        swept_collision_obj = self.MCTS_ins.MCTS_tree_.check_collision_w_swept()
        for i in swept_collision_obj:
            tunnel = self.MCTS_ins.MCTS_tree_.get_tunnel(self.MCTS_ins.MCTS_tree_.robot_, self.MCTS_ins.MCTS_tree_.curr_config_[i][:2])
            is_collision = self.MCTS_ins.MCTS_tree_.collision_tunnel_static(tunnel, no_unknown=True)
            # self.MCTS_ins.MCTS_tree_.tunnel_and_normal_visualizer([tunnel])
            assert not is_collision, "Scenario is not solvable due to collision on grasp tunnel with static object: " + str(i)

        # Dependency relation checker
        dependency_graph = self.MCTS_ins.MCTS_tree_.get_dependency_relation()
        print(dependency_graph)
        for obj1 in swept_collision_obj:
            for obj2 in dependency_graph[obj1]:
                assert not obj1 in dependency_graph[obj2], "Scenario is not solvable due to dependency with object: " + str(obj1) + " & " + str(obj2)
        self.is_scenario_checked = True

    def unknown_tunnel_check(self):
        swept_collision_obj = self.MCTS_ins.MCTS_tree_.check_collision_w_swept()
        tunnel_collision_obj_idx = []
        tunnel_collision_area = []
        for i in swept_collision_obj:
            tunnel = self.MCTS_ins.MCTS_tree_.get_tunnel(self.MCTS_ins.MCTS_tree_.robot_, self.MCTS_ins.MCTS_tree_.curr_config_[i][:2])
            collision_area = self.MCTS_ins.MCTS_tree_.get_collision_tunnel_unknown(tunnel)
            # self.MCTS_ins.MCTS_tree_.tunnel_and_normal_visualizer([tunnel])
            if collision_area:
                tunnel_collision_obj_idx.append(i)
                tunnel_collision_area += collision_area
    
        # return tunnel_collision_area
        return tunnel_collision_obj_idx, tunnel_collision_area

    def resolve_length_issue(self, start_config, end_config, static_config):
        new_start_config = deepcopy(start_config)
        new_end_config = deepcopy(end_config)
        new_static_config = deepcopy(static_config)
        
        new_end_config = [new_static_config[-1]] + new_end_config
        return new_start_config, new_end_config

        
    def config_sanity_check(self, config1, config2):
        if len(config1) != len(config2):
            return False
        else:
            for i in range(len(config1)):
                if config1[i][-1] != config2[i][-1]:
                    return False
            return True

    def delete_swept(self, root):
        if root.children_ is not None:
            for child in root.children_:
                self.delete_swept(child)

        root.swept_volume1 = None
        root.swept_volume2 = None
        
    def insert_swept(self, root):
        if root.children_ is not None:
            for child in root.children_:
                self.insert_swept(child)

        root.swept_volume1 = self.swept_volume1
        root.swept_volume2 = self.swept_volume2

    def insert_radius(self, root):
        if root.children_ is not None:
            for child in root.children_:
                self.insert_radius(child)

        root.radius = (0.0515 / root.scale)
        root.radius = (0.0515 / root.scale)

    def global_optimization(self):
        #optimize the result in the globel level
        if self.obj_mesh is not None:
            for tree in self.track_level_steps_:
                for root in tree:
                    self.delete_swept(root)

        print('Start global optimization')
        global_plan = []
        for i in range(len(self.track_level_steps_)):
            if i != len(self.track_level_steps_)-1:
                global_plan += deepcopy(self.track_level_steps_[i][:-1])
            else:
                global_plan += deepcopy(self.track_level_steps_[i])
        
        for i in range(len(global_plan)):
            # global_plan[i].curr_config_ = global_plan[i].static_config_ + global_plan[i].curr_config_
            global_plan[i].curr_config_ = global_plan[i].curr_config_
        
        recover_plan = [[[-1, -1, -1], global_plan[0], None]]

        for i in range(len(global_plan)-1):
            start_config = global_plan[i].curr_config_
            end_config = global_plan[i+1].curr_config_

            move_index = None
            for t in range(len(start_config)):
                if start_config[t] != end_config[t]:
                    move_index = t
                    break
            if move_index == None:
                print("Tree Traversal Error!")

            recover_plan.append([[start_config[move_index][3], 
                                                        start_config[move_index][:2],
                                                        end_config[move_index][:2]], global_plan[i+1], move_index])

        
        #filter out same object move multiple times
        start_index = 0
        new_recover_plan = []
        while start_index < len(recover_plan):
            color1, pos1, _ = recover_plan[start_index][0]
            reach_index = start_index + 1
            while reach_index < len(recover_plan):
                color2, _, pos2 = recover_plan[reach_index][0]
                if color2 != color1: break
                else: reach_index += 1

            if reach_index == start_index + 1:
                new_recover_plan.append(recover_plan[start_index])
            else:
                (color1, pos1, _), ori_node, move_index = recover_plan[start_index]
                (color2, _, pos2), _, _ = recover_plan[reach_index - 1]
                temp_node = deepcopy(ori_node)
                temp_node.curr_config_[move_index][0] = pos2[0]
                temp_node.curr_config_[move_index][1] = pos2[1]
                new_recover_plan.append([[color1, pos1, pos2], temp_node, move_index])
                start_index = reach_index - 1
                print('optimized adjacent index', color1, color2)

            start_index += 1

        self.track_level_steps_ = [[x[1] for x in new_recover_plan]]
        
        #optimize sequence that are not adjacent
        while True:
            change_flag = False
            track_color_index = defaultdict(list)
            for i in range(len(new_recover_plan)):
                color, _, _ = new_recover_plan[i][0]
                track_color_index[color].append(i)

            all_pairs = []
            for key, value in track_color_index.items():
                if len(value) > 1:
                    for k in range(len(value)-1):
                        all_pairs.append([[value[k], value[k+1]], value[k+1] - value[k]])

            all_pairs.sort(key = lambda x : x[1])
            
            for [ind1, ind2], _ in all_pairs:
                color1, pos1, pos2 = new_recover_plan[ind1][0]
                color2, pos3, pos4 = new_recover_plan[ind2][0]

                if color1 != color2 or pos2 != pos3:
                    print('something wrong on global optimization')
                    sys.exit(1)
                else:
                    temp_treenode = deepcopy(new_recover_plan[ind1][1])
                    temp_index = new_recover_plan[ind1][2]
                    relocate_tunnel = temp_treenode.get_tunnel(temp_treenode.robot_, pos4)
                    col_objects = temp_treenode.collision_tunnel_object(relocate_tunnel)
                    col_objects = [x for x in col_objects if x != temp_index]
                    if not col_objects:
                        temp_treenode.curr_config_[temp_index][0] = pos4[0]
                        temp_treenode.curr_config_[temp_index][1] = pos4[1]
                        collision_free_flag = True
                        for middle_index in range(ind1+1, ind2):
                            _, grasp_loc, relocate_loc = new_recover_plan[middle_index][0]
                            grasp_tunnel = temp_treenode.get_tunnel(temp_treenode.robot_, grasp_loc)
                            relocate_tunnel = temp_treenode.get_tunnel(temp_treenode.robot_, relocate_loc)
                            col_objects = temp_treenode.collision_tunnel_object(grasp_tunnel)
                            col_objects += temp_treenode.collision_tunnel_object(relocate_tunnel)
                            if len(set(col_objects)) > 1:
                                collision_free_flag = False
                                break

                        if collision_free_flag:
        
                            modified_recover_plan = new_recover_plan[:ind1]
                            modified_recover_plan.append([[color1, pos1, pos4], temp_treenode, temp_index])
                            for ii in range(ind1+1,ind2):
                                for jj in range(len(new_recover_plan[ii][1].curr_config_)):
                                    if new_recover_plan[ii][1].curr_config_[jj][3] == color1:
                                        new_recover_plan[ii][1].curr_config_[jj][0] = pos4[0]
                                        new_recover_plan[ii][1].curr_config_[jj][1] = pos4[1]
                                        break
                            modified_recover_plan += new_recover_plan[ind1+1:ind2]
                            modified_recover_plan += new_recover_plan[ind2+1:]
                            new_recover_plan = modified_recover_plan
                            change_flag = True
                            print('optimized distance index')
                            break
                    
            if not change_flag: break


        #print([x[0] for x in new_recover_plan])
        if self.obj_mesh is not None:
            for tree in new_recover_plan:
                # pdb.set_trace()
                for root in tree:
                    if type(root) == type(self.track_level_steps_[0][0]):
                        self.insert_swept(root)
        self.track_level_steps_ = [[x[1] for x in new_recover_plan]]
        self.final_optimized_plan_ = new_recover_plan
    
    def vis_whole_sequence(self):
        for level_steps in self.track_level_steps_:
            self.vis_steps(level_steps)
        self.track_level_steps_[-1][-1].tunnel_and_normal_visualizer(true_color = True)

    def vis_steps(self, tree_nodes):
        tree_nodes[0].tunnel_and_normal_visualizer()
        for i in range(len(tree_nodes)-1):
            start_config = tree_nodes[i].curr_config_
            end_config = tree_nodes[i+1].curr_config_
            move_index = None
            for t in range(len(start_config)):
                if start_config[t] != end_config[t]:
                    move_index = t
                    break
            if move_index == None: 
                print("Tree Traversal Error!")
                sys.exit(1)
            grasp_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, start_config[move_index])
            relocate_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, end_config[move_index])
            tree_nodes[i].tunnel_and_normal_visualizer([grasp_tunnel, relocate_tunnel])

        tree_nodes[-1].tunnel_and_normal_visualizer()

    def calculate_total_length_travelled_ICRA(self):
        self.total_length_travelled_ = 0.0

        for i in range(1, len(self.final_optimized_plan_)):
            color, start_pos, end_pos = self.final_optimized_plan_[i][0]
            tree_node = self.final_optimized_plan_[i][1]
            
            rx, ry = tree_node.robot_

            grasp_x, grasp_y = start_pos
            relocate_x, relocate_y = end_pos

            self.total_length_travelled_ += math.sqrt((rx - grasp_x)**2 + (ry - grasp_y)**2)
            self.total_length_travelled_ += math.sqrt((rx - relocate_x)**2 + (ry - relocate_y)**2)


        return self.total_length_travelled_

    def calculate_total_length_displacement_ICRA(self):
        self.total_length_displacement_ = 0.0

        for i in range(1, len(self.final_optimized_plan_)):
            color, start_pos, end_pos = self.final_optimized_plan_[i][0]
            
            grasp_x, grasp_y = start_pos
            relocate_x, relocate_y = end_pos

            self.total_length_displacement_ += math.sqrt((grasp_x - relocate_x)**2 + (grasp_y - relocate_y)**2)

        return self.total_length_displacement_


    def save_planning_results(self):
        plan = []
        for tree_nodes in self.track_level_steps_:
            for i in range(len(tree_nodes)):
                start_config = tree_nodes[i].curr_config_
                plan.append(start_config)
        plan = np.array(plan)
        return plan


    def calculate_total_length_travelled(self):
        self.total_length_travelled_ = 0.0
        for tree_nodes in self.track_level_steps_:
            for i in range(len(tree_nodes)-1):
                start_config = tree_nodes[i].curr_config_
                end_config = tree_nodes[i+1].curr_config_
                move_index = None
                for t in range(len(start_config)):
                    if start_config[t] != end_config[t]:
                        move_index = t
                        break
                if move_index == None:
                    print('Tree Traversal Error!')
                    sys.exit(1)
                rx, ry = tree_nodes[i].robot_

                self.total_length_travelled_ += math.sqrt((start_config[move_index][0] - rx)**2 + (start_config[move_index][1] - ry)**2)
                self.total_length_travelled_ += math.sqrt((end_config[move_index][0] - rx)**2 + (end_config[move_index][1] - ry)**2)

        return self.total_length_travelled_
    
    def calculate_total_length_displacement(self):
        self.total_length_displacement_ = 0.0
        for tree_nodes in self.track_level_steps_:
            for i in range(len(tree_nodes)-1):
                start_config = tree_nodes[i].curr_config_
                end_config = tree_nodes[i+1].curr_config_
                move_index = None
                for t in range(len(start_config)):
                    if start_config[t] != end_config[t]:
                        move_index = t
                        break
                if move_index == None:
                    print('Tree Traversal Error!')
                    sys.exit(1)
                rx, ry = tree_nodes[i].robot_

                self.total_length_displacement_ += math.sqrt((start_config[move_index][0] - end_config[move_index][0])**2 + (start_config[move_index][1] - end_config[move_index][1])**2)

        return self.total_length_displacement_

    def animate_whole_sequence_IROS(self):
        #create a global canvas
        #plt.figure(figsize = (8.5, 18))
        plt.figure(figsize = (10, 10))
        step_size = 1.0

        for tree_nodes in self.track_level_steps_:
            plt.clf()
            tree_nodes[0].tunnel_and_normal_visualizer(animation = True)
            for i in range(len(tree_nodes)-1):
                start_config = deepcopy(tree_nodes[i].curr_config_[:])
                end_config = deepcopy(tree_nodes[i+1].curr_config_[:])
                static_config = deepcopy(tree_nodes[i].static_config_[:])

                #if len(start_config) != len(end_config):
                #    end_config = [static_config[-1]] + end_config
                #    static_config = static_config[:-1]

                move_index = None
                for t in range(len(start_config)):
                    if start_config[t] != end_config[t]:
                        move_index = t
                        break
                if move_index == None:
                    print('Tree Traversal Error!')
                    sys.exit(1)
                start_x, start_y, _, _ = start_config[move_index]
                robot_x, robot_y = tree_nodes[i].robot_
                end_x, end_y, _, _ = end_config[move_index]

                #dummy variables
                dummy_curr_config = deepcopy(start_config)
                dummy_end_config = deepcopy(end_config)
                dummy_goal_config = end_config
                dummy_grid = tree_nodes[i].grid_
                dummy_static_config = static_config
                
                #grasp_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, start_config[move_index])
                #relocate_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, end_config[move_index])

                #grasp tunnel animation
                #distance_start = math.sqrt((start_x - robot_x)**2 + (start_y - robot_y)**2)
                #start_steps = math.ceil(distance_start / step_size)
                #delta_x, delta_y = (robot_x - start_x) / start_steps, (robot_y - start_y) / start_steps

                dummy_source_tree_node = Tree_Node(dummy_curr_config, dummy_goal_config, dummy_grid, dummy_static_config)
                dummy_target_tree_node = Tree_Node(dummy_end_config, dummy_goal_config, dummy_grid, dummy_static_config)

                smart_LMP_motion(dummy_source_tree_node, dummy_target_tree_node)
                #for t in range(start_steps + 1):
                #    plt.clf()
                #    dummy_tree_node.tunnel_and_normal_visualizer([grasp_tunnel], animation = True)
                #    dummy_tree_node.curr_config_[move_index][0] += delta_x
                #    dummy_tree_node.curr_config_[move_index][1] += delta_y


                #relocate tunnel animation
                #distance_end = math.sqrt((end_x - robot_x)**2 + (end_y - robot_y)**2)
                #end_steps = math.ceil(distance_end / step_size)
                #delta_x, delta_y = (end_x - robot_x) / end_steps, (end_y - robot_y) / end_steps

                #dummy_tree_node.curr_config_[move_index][0] = robot_x
                #dummy_tree_node.curr_config_[move_index][1] = robot_y

                #for t in range(end_steps + 1):
                #    plt.clf()
                #    dummy_tree_node.tunnel_and_normal_visualizer([relocate_tunnel], animation = True)
                #    dummy_tree_node.curr_config_[move_index][0] += delta_x
                #    dummy_tree_node.curr_config_[move_index][1] += delta_y
            plt.clf()
            tree_nodes[-1].tunnel_and_normal_visualizer(animation = True)

        plt.clf()
        self.track_level_steps_[-1][-1].tunnel_and_normal_visualizer(true_color = True, animation = True)
        plt.show()

    def animate_whole_sequence(self):
        #create a global canvas
        if self.scale == 0.01:
            plt.figure(figsize = (len(self.grid_[0])/5, len(self.grid_)/5))
            step_size = 5.0
        else:
            plt.figure(figsize = (len(self.grid_[0]), len(self.grid_)))
            step_size = 1.0

        # for tree_nodes in self.track_level_steps_:
        #     for node in tree_nodes:
        #         node.unknown_area = self.unknown_area
        #         node.valid_points = self.valid_area
        
        for tree_nodes in self.track_level_steps_:
            # pdb.set_trace()
            plt.clf()

            tree_nodes[0].tunnel_and_normal_visualizer(animation = True)
            for i in range(len(tree_nodes)-1):
                start_config = deepcopy(tree_nodes[i].curr_config_[:])
                end_config = deepcopy(tree_nodes[i+1].curr_config_[:])
                static_config = deepcopy(tree_nodes[i].static_config_[:])

                #if len(start_config) != len(end_config):
                #    end_config = [static_config[-1]] + end_config
                #    static_config = static_config[:-1]

                move_index = None
                for t in range(len(start_config)):
                    if start_config[t] != end_config[t]:
                        move_index = t
                        break
                if move_index == None:
                    print('Tree Traversal Error!')
                    sys.exit(1)
                start_x, start_y, _, _ = start_config[move_index]
                robot_x, robot_y = tree_nodes[i].robot_
                end_x, end_y, _, _ = end_config[move_index]

                #dummy variables
                dummy_curr_config = deepcopy(start_config)
                dummy_goal_config = end_config
                dummy_grid = tree_nodes[i].grid_
                dummy_static_config = static_config
                
                grasp_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, start_config[move_index])
                temp = tree_nodes[i].collision_tunnel_static(grasp_tunnel)
                print(start_config[move_index])
                relocate_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, end_config[move_index])

                #grasp tunnel animation
                distance_start = math.sqrt((start_x - robot_x)**2 + (start_y - robot_y)**2)
                start_steps = math.ceil(distance_start / step_size)
                delta_x, delta_y = (robot_x - start_x) / start_steps, (robot_y - start_y) / start_steps

                dummy_tree_node = Tree_Node(dummy_curr_config, dummy_goal_config, dummy_grid, dummy_static_config, swept_volume1=self.swept_volume1, swept_volume2=self.swept_volume2, obj_mesh=self.obj_mesh, scale=self.scale, unknown_area=self.unknown_area, valid_area=self.valid_area, potential_centers=self.potential_centers)
                for t in range(start_steps + 1):
                    plt.clf()
                    dummy_tree_node.tunnel_and_normal_visualizer([grasp_tunnel], animation = True)
                    dummy_tree_node.curr_config_[move_index][0] += delta_x
                    dummy_tree_node.curr_config_[move_index][1] += delta_y


                #relocate tunnel animation
                distance_end = math.sqrt((end_x - robot_x)**2 + (end_y - robot_y)**2)
                end_steps = math.ceil(distance_end / step_size)
                delta_x, delta_y = (end_x - robot_x) / end_steps, (end_y - robot_y) / end_steps

                dummy_tree_node.curr_config_[move_index][0] = robot_x
                dummy_tree_node.curr_config_[move_index][1] = robot_y

                for t in range(end_steps + 1):
                    plt.clf()
                    dummy_tree_node.tunnel_and_normal_visualizer([relocate_tunnel], animation = True)
                    dummy_tree_node.curr_config_[move_index][0] += delta_x
                    dummy_tree_node.curr_config_[move_index][1] += delta_y

        
            plt.clf()
            tree_nodes[-1].tunnel_and_normal_visualizer(animation = True)

        plt.clf()
        self.track_level_steps_[-1][-1].tunnel_and_normal_visualizer(true_color = True, animation = True)
        plt.show()




class MCTS_algo():
    def __init__(self, curr_config, goal_config, scene_info, grid=None, index=0, swept_volume1=None, swept_volume2=None, obj_mesh=None, target_pos=[], unknown_area=[], valid_area=[], potential_centers=[]):
        self.grid_ = grid
        self.scene_info = scene_info
        self.swept_volume1 = swept_volume1
        self.swept_volume2 = swept_volume2
        self.obj_mesh = obj_mesh

        self.time_limit = None
        self.start_time = None

        self.scale = self.scale_grid(scene_info, curr_config, unknown_area != [])
        curr_config = self.scale_config(curr_config)
        goal_config = self.scale_config(goal_config)

        if target_pos:
            target_pos = self.scale_config([target_pos])
        
        self.static_config_ = goal_config[:index]

        self.curr_config_ = curr_config[index:]
        self.goal_config_ = goal_config[index:]

        self.MCTS_tree_ = Tree_Node(self.curr_config_, self.goal_config_, self.grid_, self.static_config_, swept_volume1=swept_volume1, swept_volume2=swept_volume2, obj_mesh=obj_mesh, scale=self.scale, target_pos=target_pos, unknown_area=unknown_area, valid_area=valid_area, potential_centers=potential_centers)
        self.leaf_ = []
        self.root_ = self.MCTS_tree_

        # self.distance_lookup_ = defaultdict(list)
        # for t in range(-len(self.grid_) + 1, len(self.grid_)):
        #     for k in range(-len(self.grid_[0]) + 1, len(self.grid_[0])):
        #         distance = round((t)**2 + (k)**2, 3)
        #         self.distance_lookup_[distance].append([t, k])
        # self.distance_lookup_ = [list(x) for x in self.distance_lookup_.items()]
        # self.distance_lookup_.sort(key = lambda x: x[0])

    def scale_grid(self, scene_info, curr_config, cm_scale=False):
        # find max radius
        max_radius = -sys.maxsize
        for config in curr_config:
            if max_radius < config[2]:
                max_radius = config[2]
    
        if cm_scale:
            scale = 0.01
            print("scale")
        else:
            scale = max_radius

        # grid_x = int(scene_info[1] / scale) - 2
        # grid_y = int(scene_info[0] / scale + 0.3 / scale)
        # pdb.set_trace()
        grid_x = int(scene_info[1] / scale)
        grid_y = int(scene_info[0] / scale + 0.3 / scale)
        print("x axis:", grid_y)
        print("y axis:", grid_x)

        self.grid_ = []
        self.grid_ = np.zeros((grid_y, grid_x))

        return scale

    def scale_config(self, config):
        for pos in config:
            for i in range(3):
                pos[i] = pos[i] / self.scale
            temp = pos[0]
            pos[0] = -pos[1]
            pos[1] = temp

            # print("--------------------------------------------------", config)
        return config

    def selection(self):
        selected_node = None
        start = self.root_
        while start:
            if not start.children_: return start
            else:
                new_start = None
                max_ucb = -sys.maxsize
                for child in start.children_:
                    temp_ucb = child.get_UCB()
                    if temp_ucb > max_ucb:
                        max_ucb = temp_ucb
                        new_start = child
                start = new_start
        print('should not reach here\n')

    def expansion(self, selected_leaf_node, rollout_flag = False):
        curr_config = deepcopy(selected_leaf_node.curr_config_)
        goal_config = deepcopy(selected_leaf_node.goal_config_)
        grid = selected_leaf_node.grid_
        static_config = selected_leaf_node.static_config_
        search_depth = 0

        #print(curr_config)
        #print(goal_config)

        # print("--------------------------------------------------------------------")
        # new_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.curr_config_[1])
        # print("collision", selected_leaf_node.object_in_collision_) 
        # print("feasible", selected_leaf_node.random_object_selection())
        # selected_leaf_node.tunnel_and_normal_visualizer([new_tunnel])

        if selected_leaf_node.object_in_collision_:
            #there are still objects that need to be moved
            #first try to grasp it
            random_obj_flag = False
            current_list = selected_leaf_node.object_in_collision_
            while True:
                new_current_list = set()
                search_depth += 1
                for index in current_list:
                    grasp_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.curr_config_[index])
                    relocate_object = selected_leaf_node.collision_tunnel_object(grasp_tunnel)
                    relocate_object = [x for x in relocate_object if x != index]

                    grasp_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.goal_config_[index])
                    if selected_leaf_node.collision_tunnel_static(grasp_tunnel): continue
                    
                    #relocated_object is None, grasp it to another region
                    if not relocate_object:
                        #first try to place it at goal region
                        relocate_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.goal_config_[index])
                        collision_object = selected_leaf_node.collision_tunnel_object(relocate_tunnel)
                        collision_object = [x for x in collision_object if x != index]

                        #this tries to directly place it at goal region
                        if not collision_object:
                            new_regions = []
                            for upper_index in range(index, -1, -1):
                                is_overrun = self.check_time()
                                if is_overrun:
                                    return None
                                new_regions = selected_leaf_node.propose_new_region(index, [x for x in range(1, upper_index)], random_obj_flag)
                                if new_regions: break

                            if new_regions:
                                for gx, gy in new_regions:
                                    new_curr_config = deepcopy(curr_config)
                                    new_grid = deepcopy(grid)
                                    new_static_config = deepcopy(static_config)
                                    move_distance = math.sqrt((curr_config[index][0] - gx)**2 + \
                                                                (curr_config[index][1] - gy)**2)
                                    new_curr_config[index][0] = gx
                                    new_curr_config[index][1] = gy

                                    new_obj_mesh = self.update_mesh_pos(selected_leaf_node, [gx - curr_config[index][0], gy - curr_config[index][1]], index)

                                    new_node = Tree_Node(new_curr_config, new_curr_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance, swept_volume1=self.swept_volume1, swept_volume2=self.swept_volume2, obj_mesh=new_obj_mesh, scale=self.scale, unknown_area=selected_leaf_node.unknown_area, valid_area=selected_leaf_node.valid_area, potential_centers=selected_leaf_node.potential_centers)
                                    selected_leaf_node.add_child(new_node)
                                    new_node.set_parent(selected_leaf_node)
                                    if rollout_flag: 
                                        return new_node

                        else:
                            # no need in my case
                            new_regions = []
                            for upper_index in range(index, -1, -1):
                                is_overrun = self.check_time()
                                if is_overrun:
                                    return None
                                new_regions = selected_leaf_node.propose_new_region(index, [x for x in range(1, upper_index)], random_obj_flag)
                                if new_regions: break
    
                            if new_regions:
                                for gx, gy in new_regions:
                                    #print (gx, gy)
                                    new_curr_config = deepcopy(curr_config)
                                    new_grid = deepcopy(grid)
                                    new_static_config = deepcopy(static_config)
                                    move_distance = math.sqrt((curr_config[index][0] - gx)**2 + \
                                                              (curr_config[index][1] - gy)**2)
                                    new_curr_config[index][0] = gx
                                    new_curr_config[index][1] = gy

                                    new_obj_mesh = self.update_mesh_pos(selected_leaf_node, [gx - curr_config[index][0], gy - curr_config[index][1]], index)
                                    
                                    new_node = Tree_Node(new_curr_config, new_curr_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance, swept_volume1=self.swept_volume1, swept_volume2=self.swept_volume2, obj_mesh=new_obj_mesh, scale=self.scale, unknown_area=selected_leaf_node.unknown_area, valid_area=selected_leaf_node.valid_area, potential_centers=selected_leaf_node.potential_centers)
                                    selected_leaf_node.add_child(new_node)
                                    new_node.set_parent(selected_leaf_node)
                                    if rollout_flag: 
                                        return new_node
                            
                    else:
                        #some object block the path to grasp the object
                        for new_index in relocate_object:
                            temp_grasp_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.curr_config_[new_index])
                            temp_relocate_object = selected_leaf_node.collision_tunnel_object(temp_grasp_tunnel)
                            temp_relocate_object = [x for x in temp_relocate_object if x != new_index]
                            if not temp_relocate_object:
                                new_regions = []
                                for upper_index in range(new_index, -1, -1):
                                    is_overrun = self.check_time()
                                    if is_overrun:
                                        return None
                                    new_regions = selected_leaf_node.propose_new_region(new_index, [index] + [x for x in range(1, upper_index)], random_obj_flag)
                                    if new_regions: break
                                
                                if new_regions:
                                    for gx, gy in new_regions:
                                        new_curr_config = deepcopy(curr_config)
                                        new_grid = deepcopy(grid)
                                        new_static_config = deepcopy(static_config)
                                        move_distance = math.sqrt((curr_config[new_index][0] - gx)**2 + \
                                                              (curr_config[new_index][1] - gy)**2)
                                        new_curr_config[new_index][0] = gx
                                        new_curr_config[new_index][1] = gy

                                        new_obj_mesh = self.update_mesh_pos(selected_leaf_node, [gx - curr_config[new_index][0], gy - curr_config[new_index][1]], new_index)
                                        
                                        new_node = Tree_Node(new_curr_config, new_curr_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance, swept_volume1=self.swept_volume1, swept_volume2=self.swept_volume2, obj_mesh=new_obj_mesh, scale=self.scale, unknown_area=selected_leaf_node.unknown_area, valid_area=selected_leaf_node.valid_area, potential_centers=selected_leaf_node.potential_centers)
                                        selected_leaf_node.add_child(new_node)
                                        new_node.set_parent(selected_leaf_node)
                                        if rollout_flag:
                                            return new_node
                                
                            else:
                                new_current_list.add(new_index)
                if len(selected_leaf_node.children_):
                    break
                else:
                    current_list = list(new_current_list)
                    if search_depth == 4:
                        break
                    if search_depth >= 2:
                        current_list = selected_leaf_node.random_object_selection()
                        random_obj_flag = True

        else:
            new_curr_config = deepcopy(curr_config)
            new_goal_config = deepcopy(goal_config)
            new_grid = deepcopy(grid)
            new_static_config = deepcopy(static_config)
            move_distance = math.sqrt((curr_config[0][0] - goal_config[0][0])**2 + \
                                      (curr_config[0][1] - goal_config[0][1])**2)
            new_curr_config[0] = new_goal_config[0]
            new_node = Tree_Node(new_curr_config, new_curr_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance, swept_volume1=self.swept_volume1, swept_volume2=self.swept_volume2, obj_mesh=self.obj_mesh, scale=self.scale, unknown_area=selected_leaf_node.unknown_area, valid_area=selected_leaf_node.valid_area, potential_centers=selected_leaf_node.potential_centers)
            selected_leaf_node.add_child(new_node)
            new_node.set_parent(selected_leaf_node)
            if rollout_flag:
                print("return rollout5")
                return new_node

        if selected_leaf_node.children_:
            return selected_leaf_node.children_[0]
        else:
            print("return None")
            return None
        
    def check_collision_w_swept(self, obj_mesh):
        collision_obj_list = []
        for obj_idx in range(0, len(self.curr_config_)):
            # read collision mesh
            temp_verts, temp_tris = obj_mesh[obj_idx] # need mesh
            temp_m = fcl.BVHModel()
            temp_m.beginModel(len(temp_verts), len(temp_tris))
            temp_m.addSubModel(temp_verts, temp_tris)
            temp_m.endModel()
            temp_t = fcl.Transform()

            # check collision
            req = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request = req)
            self.swept_volume1.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
            is_collision1 = rdata.result.is_collision
            self.swept_volume2.collide(fcl.CollisionObject(temp_m, temp_t), rdata, fcl.defaultCollisionCallback)
            is_collision2 = rdata.result.is_collision

            if is_collision1 or is_collision2:
                collision_obj_list.append(obj_idx)

        # print("Swept check:", collision_obj_list)
        return collision_obj_list
    
    def convert2global(self, pos):
        new_pos = [pos[1] * self.scale, -pos[0] * self.scale, 0]
        return new_pos
        
    def update_mesh_pos(self, node, offset, idx):
        trans = self.convert2global(offset)
        new_obj_mesh = deepcopy(node.obj_mesh)
        verts, tries = new_obj_mesh[idx]
        for i in range(len(verts)):
                verts[i] += trans

        old_collision = self.check_collision_w_swept(node.obj_mesh)
        new_collision = self.check_collision_w_swept(new_obj_mesh)
        # print("OLD:", old_collision, "idx", idx)
        # print("NEW:", new_collision, "idx", idx)
        # problem = False
        # if idx in new_collision:
        #     problem = True
        # if idx in old_collision and set([idx] + new_collision) != set(old_collision):
        #     problem = True

        return new_obj_mesh

    def rollout(self, rollout_leaf_node):
        if not rollout_leaf_node:
            return 10000
        #little bit smarter here instead of random
        if rollout_leaf_node.is_goal_config():
            #print ('rollout finished\n')
            return rollout_leaf_node.total_distance_
        else:
            new_node = self.expansion(rollout_leaf_node, True)
            if new_node:
                reward = self.rollout(new_node)
                return reward
            else:
                return 10000

    def backup(self, reward, rollout_leaf_node):
        start = rollout_leaf_node
        while start:
            start.reward_ += -1*reward
            start.visited_ += 1.0
            start = start.parent_

    def check_time(self):
        if self.time_limit is None:
            return False
        curr_time = time.time()
        if self.time_limit <= curr_time - self.start_time:
            return True
        return False
    
    def get_children_nodes(self, root):
        if not root.children_:
            return [root]

        total_child = []
        for child in root.children_:
            total_child += self.get_children_nodes(child)

        return total_child

    def exec_algo(self):
        final_leaf = None
        while True:
            is_overrun = self.check_time()
            if is_overrun:
                return None

            selected_leaf_node = self.selection()
            if selected_leaf_node.is_goal_config_swept() and not selected_leaf_node.object_in_collision_:
                final_leaf = selected_leaf_node
                break
            rollout_leaf_node = None
            if selected_leaf_node.visited_ != 0.0 or selected_leaf_node == self.root_:
                rollout_leaf_node = self.expansion(selected_leaf_node)
            else:
                rollout_leaf_node = selected_leaf_node
            if rollout_leaf_node:
                reward = self.rollout(rollout_leaf_node)
                self.backup(reward, rollout_leaf_node)
                rollout_leaf_node.children_ = []
            else:
                reward = 10000
                self.backup(reward, selected_leaf_node)

        #traverse the tree
        tree_nodes = final_leaf.traverse_tree()
        return tree_nodes



if __name__ == '__main__':
    
    mode = int(sys.argv[1])
    case_index = int(sys.argv[2])
    animation_flag = int(sys.argv[3])
    LMP_motion = sys.argv[4]

    # curr_config = [[0.753384612398586   , -0.026747099577607514 , 0.05153550000000001, 'orange'],
    #                 [0.86                , 0.86                  , 0.05153550000000001, 'b'],
    #             #    [0.86                , 0.81                  , 0.05153550000000001, 'r'],
    #                [0.31                , 0.01                  , 0.05153550000000001, 'g'],
    #                [0.38488997462182534 , -0.1468114942372023   , 0.05153550000000001, 'r']]
    
    # goal_config = [[0.4   , -0.026747099577607514 , 0.05153550000000001, 'orange'],
    #                 [0.86                , 0.86                  , 0.05153550000000001, 'b'],
    #             #    [0.86                , 0.81                  , 0.05153550000000001, 'r'],
    #                [0.31                , 0.01                  , 0.05153550000000001, 'g'],
    #                [0.38488997462182534 , -0.1468114942372023   , 0.05153550000000001, 'r']]
    
    # curr_config = [[4, 4, 1, 'r'], [4, 10, 1, 'g'], [4, 16, 1, 'b']]
    
    # goal_config = [[16, 4, 1, 'r'], [16, 10, 1, 'g'], [16, 16, 1, 'b']]


    curr_config = [[0.86                , 0.86                  , 0.05153550000000001, 'b'],
                   [0.31                , 0.01                  , 0.05153550000000001, 'g'],
                   [0.38488997462182534 , -0.1468114942372023   , 0.05153550000000001, 'r']]
    

    goal_config = [[0.86                , 0.86                  , 0.05153550000000001, 'b'],
                   [0.31                , 0.01                  , 0.05153550000000001, 'g'],
                   [0.38488997462182534 , -0.1468114942372023   , 0.05153550000000001, 'r']]
    
    scene_info = [0.56, 0.86000001, 0.1, 0.5]
    # obj_mesh = 

    # collision_objs = [0, 1, 2, 3, 4]
    # pdb.set_trace()
    ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config, scene_info=scene_info)
    ML_MCTS_ins.animate_whole_sequence()
    

    # if mode == 1:
    #     curr_config, goal_config = regression_test(case_index)
    #     # pdb.set_trace()
    #     ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
    #     #ML_MCTS_ins.vis_whole_sequence()
    #     #ML_MCTS_ins.global_optimization()
    #     print(animation_flag)
    #     if animation_flag == 1:
    #         if LMP_motion == 'smart':
    #             ML_MCTS_ins.animate_whole_sequence_IROS()
    #         else:
    #             ML_MCTS_ins.animate_whole_sequence()
    #     print("Total length travelled: {0}".format(ML_MCTS_ins.calculate_total_length_travelled())) 
    # # elif mode == 2:
    #     curr_config, goal_config = test_case_reader(case_index)

    #     ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
    #     #ML_MCTS_ins.vis_whole_sequence()
    #     #ML_MCTS_ins.global_optimization()
    #     #ML_MCTS_ins.animate_whole_sequence()
    #     print("Time consumption: {0}".format(ML_MCTS_ins.time_consumption_)) 
    #     print("Total steps: {0}".format(ML_MCTS_ins.total_steps_)) 
    #     print("Total length travelled: {0}".format(ML_MCTS_ins.calculate_total_length_travelled_ICRA())) 
    #     print("Total length displacement: {0}".format(ML_MCTS_ins.calculate_total_length_displacement_ICRA()))
    #     res_plan = ML_MCTS_ins.save_planning_results()

    #     write_result('MS_MCTS_DIS_FINAL_2', case_index, len(curr_config), ML_MCTS_ins.time_consumption_, ML_MCTS_ins.total_steps_, ML_MCTS_ins.calculate_total_length_travelled_ICRA(), ML_MCTS_ins.calculate_total_length_displacement_ICRA(), res_plan)
    # else:
    #     for t in range(1, 10):
    #         curr_config, goal_config = regression_test(t)
    #         ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
    #         print("Total length travelled: {0}".format(ML_MCTS_ins.calculate_total_length_travelled())) 
