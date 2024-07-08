#
# file:   MCTS_algo.py
# Brief:  Implementation of ML-MCTS algorithm
# Author: Hanwen Ren
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
from rearrangement_planning_util_ICRA import regression_test
from rearrangement_planning_util_ICRA import write_result
#from rearrangement_planning_util_ICRA import smart_LMP_motion
from test_case_generator import RP_test_case_generator
from test_case_generator import test_case_reader


#main 2D ML-MCTS class that internally calls MCTS class
class multi_level_MCTS_algo():
	def __init__(self, curr_config, goal_config):
		self.curr_config_ = curr_config
		self.goal_config_ = goal_config
		self.total_length_travelled_ = 0.0
		self.total_length_displacement_ = 0.0

		#create object topo order here
		goal_config_data = [[self.goal_config_[i][1], i] for i in range(len(self.goal_config_))]
		goal_config_data.sort(key = lambda x : x[0], reverse = True)

		sorted_curr_config = []
		sorted_goal_config = []

		for _, index in goal_config_data:
			sorted_curr_config.append(self.curr_config_[index])
			sorted_goal_config.append(self.goal_config_[index])

		self.curr_config_ = sorted_curr_config
		self.goal_config_ = sorted_goal_config

		self.grid_ = []
		for i in range(40):
			temp_arr = []
			for j in range(13):
				temp_arr.append(0)
			self.grid_.append(temp_arr)


		topo_revising_node = Tree_Node(self.curr_config_, self.goal_config_, self.grid_)
		dependency_graph = topo_revising_node.get_dependency_relation()

		new_curr_config, new_goal_config = [], []

		max_iteration = len(self.curr_config_)

		visited = set()

		for t in range(max_iteration):
			for i in range(len(self.curr_config_)):
				if i not in visited:
					if not dependency_graph[i]:
						new_curr_config.append(self.curr_config_[i])
						new_goal_config.append(self.goal_config_[i])
						visited.add(i)
						for key,value in dependency_graph.items():
							if i in value:
								dependency_graph[key].remove(i)

		if len(new_curr_config) == len(self.curr_config_):
			print('Topological order updated!')
			self.curr_config_ = new_curr_config
			self.goal_config_ = new_goal_config

		#object topo order done

		print ('Start configuration')
		print (self.curr_config_)
		print ('Goal configuration')
		print (self.goal_config_)

		
		start_config = self.curr_config_

		track_level_steps = []

		total_steps = 0

		start_time = time.time()

		#debug purpose
		#colors = ['r', 'g', 'b', 'm', 'k', 'y']
		#print (start_config)
		#for i in range(6):
		#	start_config[i][3] = colors[i]
		#	self.goal_config_[i][3] = colors[i]
		#check_start_node = Tree_Node(self.goal_config_, self.goal_config_, self.grid_)
		#check_start_node.tunnel_and_normal_visualizer()
		#delete later

		for i in range(len(self.curr_config_)):
			#print(start_config, i)
			MCTS_ins = MCTS_algo(start_config, self.goal_config_, self.grid_, i)
			level_steps = MCTS_ins.exec_algo()
			start_config = level_steps[-1].static_config_ + level_steps[-1].curr_config_
			track_level_steps.append(level_steps)
			total_steps += len(level_steps)

		self.track_level_steps_ = track_level_steps

		self.global_optimization()

		self.total_steps_ = len(self.track_level_steps_[0])
		print(self.total_steps_)

		end_time = time.time()
		self.time_consumption_ = end_time - start_time
		#self.total_steps_ = total_steps - len(self.curr_config_) + 1

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


	def global_optimization(self):
		#optimize the result in the globel level
		print('Start global optimization')
		global_plan = []
		for i in range(len(self.track_level_steps_)):
			if i != len(self.track_level_steps_)-1:
				global_plan += deepcopy(self.track_level_steps_[i][:-1])
			else:
				global_plan += deepcopy(self.track_level_steps_[i])
		
		for i in range(len(global_plan)):
			global_plan[i].curr_config_ = global_plan[i].static_config_ + global_plan[i].curr_config_
		
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
				#	end_config = [static_config[-1]] + end_config
				#	static_config = static_config[:-1]

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
				#	plt.clf()
				#	dummy_tree_node.tunnel_and_normal_visualizer([grasp_tunnel], animation = True)
				#	dummy_tree_node.curr_config_[move_index][0] += delta_x
				#	dummy_tree_node.curr_config_[move_index][1] += delta_y


				#relocate tunnel animation
				#distance_end = math.sqrt((end_x - robot_x)**2 + (end_y - robot_y)**2)
				#end_steps = math.ceil(distance_end / step_size)
				#delta_x, delta_y = (end_x - robot_x) / end_steps, (end_y - robot_y) / end_steps

				#dummy_tree_node.curr_config_[move_index][0] = robot_x
				#dummy_tree_node.curr_config_[move_index][1] = robot_y

				#for t in range(end_steps + 1):
				#	plt.clf()
				#	dummy_tree_node.tunnel_and_normal_visualizer([relocate_tunnel], animation = True)
				#	dummy_tree_node.curr_config_[move_index][0] += delta_x
				#	dummy_tree_node.curr_config_[move_index][1] += delta_y

		
			plt.clf()
			tree_nodes[-1].tunnel_and_normal_visualizer(animation = True)

		plt.clf()
		self.track_level_steps_[-1][-1].tunnel_and_normal_visualizer(true_color = True, animation = True)
		plt.show()



	def animate_whole_sequence(self):
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
				#	end_config = [static_config[-1]] + end_config
				#	static_config = static_config[:-1]

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
				relocate_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, end_config[move_index])

				#grasp tunnel animation
				distance_start = math.sqrt((start_x - robot_x)**2 + (start_y - robot_y)**2)
				start_steps = math.ceil(distance_start / step_size)
				delta_x, delta_y = (robot_x - start_x) / start_steps, (robot_y - start_y) / start_steps

				dummy_tree_node = Tree_Node(dummy_curr_config, dummy_goal_config, dummy_grid, dummy_static_config)
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
	def __init__(self, curr_config, goal_config, grid, index):
		self.grid_ = grid
		self.static_config_ = goal_config[:index]

		self.curr_config_ = curr_config[index:]
		self.goal_config_ = goal_config[index:]

		self.MCTS_tree_ = Tree_Node(self.curr_config_, self.goal_config_, self.grid_, self.static_config_)
		self.leaf_ = []
		self.root_ = self.MCTS_tree_

		self.distance_lookup_ = defaultdict(list)
		for t in range(-19, 20):
			for k in range(-19, 20):
				distance = round((t)**2 + (k)**2, 3)
				self.distance_lookup_[distance].append([t, k])
		self.distance_lookup_ = [list(x) for x in self.distance_lookup_.items()]
		self.distance_lookup_.sort(key = lambda x: x[0])

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
		print ('should not reach here\n')

	def expansion(self, selected_leaf_node, rollout_flag = False):
		curr_config = selected_leaf_node.curr_config_
		goal_config = selected_leaf_node.goal_config_
		grid = selected_leaf_node.grid_
		static_config = selected_leaf_node.static_config_
		search_depth = 0
		#print(curr_config)
		#print(goal_config)

		#print(selected_leaf_node.object_in_collision_)
		#print(curr_config)
		#print(goal_config)
		#selected_leaf_node.tunnel_and_normal_visualizer()

		#print(selected_leaf_node.object_in_collision_, curr_config)
		if selected_leaf_node.object_in_collision_:
			#there are still objects that need to be moved
			#first try to grasp it
			current_list = selected_leaf_node.object_in_collision_
			while True:
				new_current_list = set()
				search_depth += 1
				#print(search_depth, current_list, rollout_flag)
				for index in current_list:
					grasp_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.curr_config_[index])
					relocate_object = selected_leaf_node.collision_tunnel_object(grasp_tunnel)
					relocate_object = [x for x in relocate_object if x != index]
					
					#relocated_object is None, grasp it to another region
					if not relocate_object:
						#first try to place it at goal region
						relocate_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.goal_config_[index])
						collision_object = selected_leaf_node.collision_tunnel_object(relocate_tunnel)
						collision_object = [x for x in collision_object if x != index]
						#this tries to directly place it at goal region
						if not collision_object:
							#print ('enters no collision')
							#create a new tree node
							flag = selected_leaf_node.test_new_region_blocking(selected_leaf_node.goal_config_[index])
							flag = True
							if not flag:
								new_curr_config = deepcopy(curr_config)
								new_goal_config = deepcopy(goal_config)
								new_grid = deepcopy(grid)
								new_static_config = deepcopy(static_config)
								new_curr_config[index] = new_goal_config[index]
								move_distance = math.sqrt((curr_config[index][0] - goal_config[index][0])**2 + 
																					(curr_config[index][1] - goal_config[index][1])**2)
								new_node = Tree_Node(new_curr_config, goal_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance)
								selected_leaf_node.add_child(new_node)
								new_node.set_parent(selected_leaf_node)
								#print (new_curr_config)
								#anew_node.tunnel_and_normal_visualizer([relocate_tunnel])
								if rollout_flag: return new_node
							else:
								#propose two target regions
								new_regions = []
								for upper_index in range(index, 0, -1):
									new_regions = selected_leaf_node.propose_new_region(index, [x for x in range(1, upper_index)])
									if new_regions: break
								if new_regions:
									for gx, gy in new_regions:
										#print (gx, gy)
										new_curr_config = deepcopy(curr_config)
										new_goal_config = deepcopy(goal_config)
										new_grid = deepcopy(grid)
										new_static_config = deepcopy(static_config)
										move_distance = math.sqrt((curr_config[index][0] - gx)**2 + \
									  	                        (curr_config[index][1] - gy)**2)
										new_curr_config[index][0] = gx
										new_curr_config[index][1] = gy
										new_node = Tree_Node(new_curr_config, goal_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance)
										selected_leaf_node.add_child(new_node)
										new_node.set_parent(selected_leaf_node)
										new_tunnel = new_node.get_tunnel(new_node.robot_, new_node.curr_config_[index])
										#new_node.tunnel_and_normal_visualizer([new_tunnel])
										if rollout_flag: return new_node
						else:
							#propose two target regions
							new_regions = []
							for upper_index in range(index, 0, -1):
								new_regions = selected_leaf_node.propose_new_region(index, [x for x in range(1, upper_index)])
								if new_regions: break
							if new_regions:
								for gx, gy in new_regions:
									#print (gx, gy)
									new_curr_config = deepcopy(curr_config)
									new_goal_config = deepcopy(goal_config)
									new_grid = deepcopy(grid)
									new_static_config = deepcopy(static_config)
									move_distance = math.sqrt((curr_config[index][0] - gx)**2 + \
									                          (curr_config[index][1] - gy)**2)
									new_curr_config[index][0] = gx
									new_curr_config[index][1] = gy
									new_node = Tree_Node(new_curr_config, goal_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance)
									selected_leaf_node.add_child(new_node)
									new_node.set_parent(selected_leaf_node)
									new_tunnel = new_node.get_tunnel(new_node.robot_, new_node.curr_config_[index])
									#new_node.tunnel_and_normal_visualizer([new_tunnel])
									if rollout_flag: return new_node
					else:
						#some object block the path to grasp the object
						for new_index in relocate_object:
							temp_grasp_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.curr_config_[new_index])
							temp_relocate_object = selected_leaf_node.collision_tunnel_object(temp_grasp_tunnel)
							temp_relocate_object = [x for x in temp_relocate_object if x != new_index]
							#need to check this - bug fix
							if not temp_relocate_object:
								new_regions = []
								for upper_index in range(new_index, -1, -1):
									new_regions = selected_leaf_node.propose_new_region(new_index, [index] + [x for x in range(1, upper_index)])
									if new_regions: break
								if new_regions:
									for gx, gy in new_regions:
										new_curr_config = deepcopy(curr_config)
										new_goal_config = deepcopy(goal_config)
										new_grid = deepcopy(grid)
										new_static_config = deepcopy(static_config)
										move_distance = math.sqrt((curr_config[new_index][0] - gx)**2 + \
									                          (curr_config[new_index][1] - gy)**2)
										new_curr_config[new_index][0] = gx
										new_curr_config[new_index][1] = gy
										new_node = Tree_Node(new_curr_config, goal_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance)
										selected_leaf_node.add_child(new_node)
										new_node.set_parent(selected_leaf_node)
									#new_node.tunnel_and_normal_visualizer()
										if rollout_flag: return new_node
							else:
								new_current_list.add(new_index)
				if len(selected_leaf_node.children_):
					break
				else:
					current_list = list(new_current_list)
					if search_depth == 4: 
						#print('reach here'); 
						#print(selected_leaf_node.curr_config_[0])
						#print(selected_leaf_node.goal_config_[0])
						#print(new_current_list)
						#grasp_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.curr_config_[0])
						#relocate_tunnel = selected_leaf_node.get_tunnel(selected_leaf_node.robot_, selected_leaf_node.goal_config_[0])
						#selected_leaf_node.tunnel_and_normal_visualizer([grasp_tunnel, relocate_tunnel]); 
						break
					if search_depth >= 2:
						#pass
						current_list = [x for x in selected_leaf_node.random_object_selection() if x != 0]
						#if len(current_list) > 2:
						#	current_list = current_list[:2]
						#print(current_list, search_depth, rollout_flag)
		else:
			new_curr_config = deepcopy(curr_config)
			new_goal_config = deepcopy(goal_config)
			new_grid = deepcopy(grid)
			new_static_config = deepcopy(static_config)
			move_distance = math.sqrt((curr_config[0][0] - goal_config[0][0])**2 + \
			                          (curr_config[0][1] - goal_config[0][1])**2)
			new_curr_config[0] = new_goal_config[0]
			new_node = Tree_Node(new_curr_config, new_goal_config, new_grid, new_static_config, selected_leaf_node.total_distance_ + move_distance)
			selected_leaf_node.add_child(new_node)
			new_node.set_parent(selected_leaf_node)
			if rollout_flag: return new_node

		if selected_leaf_node.children_:
			return selected_leaf_node.children_[0]
		else:
			return None

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

			

	def exec_algo(self):
		final_leaf = None
		while True:
			selected_leaf_node = self.selection()
			if selected_leaf_node.is_goal_config() and not selected_leaf_node.object_in_collision_:
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
		#for i in range(len(tree_nodes)):
		#	#debugging propose
		#	#tree_nodes[i].tunnel_and_normal_visualizer()
		#	#end debugging
		#	if i == 0: print('start config: {0}'.format(tree_nodes[i].curr_config_))
		#	elif i == len(tree_nodes)-1:
		#		print('goal_config: {0}'.format(tree_nodes[i].curr_config_))
		#	else:
		#		print('step {0}: {1}'.format(i, tree_nodes[i].curr_config_))
		#self.vis_steps(tree_nodes)
		return tree_nodes



if __name__ == '__main__':
	
	test_index = int(sys.argv[1])

	#paper case 1
	#four objects flip
	curr_config = [[20, 3, 1, 'blue'], [20, 6, 1, 'orange'], [20, 9, 1, 'red'], [20, 12, 1, 'yellow']]
	goal_config = [[20, 12, 1, 'blue'], [20, 9, 1, 'orange'], [20, 6, 1, 'red'], [20, 3, 1, 'yellow']]

	if test_index == 1:
		curr_config = [[20, 3, 1, 'blue'], [20, 6, 1, 'orange'], [20, 9, 1, 'red'], [20, 12, 1, 'yellow']]
		goal_config = [[20, 12, 1, 'blue'], [20, 9, 1, 'orange'], [20, 6, 1, 'red'], [20, 3, 1, 'yellow']]

	#six objects bowling
	if test_index == 2:
		curr_config = [[4, 2, 1, 'blue'], [32, 12, 1, 'orange'], [4, 7, 1, 'red'], [12, 8, 1, 'yellow'], [18, 5, 1, 'green'], [30, 3, 1, 'purple']]

		goal_config = [[20, 4, 1, 'blue'], [16, 7, 1, 'orange'], [24, 7, 1, 'red'], [20, 10, 1, 'yellow'], [12, 10, 1, 'green'], [28, 10, 1, 'purple']]

	#six objects bowling for demo usage
	#curr_config = [[4, 1, 1, 'blue'], [26, 10, 1, 'orange'], [5, 7, 1, 'red'], [12, 8, 1, 'yellow'], [18, 5, 1, 'green'], [24, 3, 1, 'purple']]


	#goal_config = [[14, 4, 1, 'blue'], [10, 7, 1, 'orange'], [18, 7, 1, 'red'], [14, 10, 1, 'yellow'], [6, 10, 1, 'green'], [22, 10, 1, 'purple']]



	#eight objects parallel
	if test_index == 3:
		curr_config = [[4, 8, 1, 'red_1'], [36, 8, 1, 'red_2'], 
								 [10, 7, 1, 'orange_1'], [22, 12, 1, 'orange_2'], 
								 [18, 8, 1, 'green_1'], [24, 2, 1, 'green_2'], 
								 [36, 3, 1, 'blue_1'], [30, 7, 1, 'blue_2']]
		goal_config = [[8, 5, 1, 'red_1'], [8, 10, 1, 'red_2'], 
								 [16, 5, 1, 'orange_1'], [16, 10, 1, 'orange_2'], 
								 [24, 5, 1, 'green_1'], [24, 10, 1, 'green_2'], 
								 [32, 5, 1, 'blue_1'], [32, 10, 1, 'blue_2']]


	#night objects 3x3
	if test_index == 4:
		curr_config = [[4, 8, 1, 'orange_1'], [36, 8, 1, 'red_1'], [10, 7, 1, 'green_1'],
								 [22, 12, 1, 'blue_1'], [18, 8, 1, 'purple'], [24, 2, 1, 'blue_2'],
								 [36, 3, 1, 'green_2'], [30, 7, 1, 'red_2'], [4, 1, 1, 'orange_2']]

		goal_config = [[14, 4, 1, 'orange_1'], [20, 4, 1, 'red_1'], [26, 4, 1, 'green_1'],
								 [14, 7, 1, 'blue_1'], [20, 7, 1, 'purple'], [26, 7, 1, 'blue_2'],
								 [14, 10, 1, 'green_2'], [20, 10, 1, 'red_2'], [26, 10, 1, 'orange_2']]

	start_time = time.time()

	ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
	#ML_MCTS_ins.animate_whole_sequence()
	for element in ML_MCTS_ins.track_level_steps_[0]:
		print(element.curr_config_)


	print('Planning time: {0}'.format(time.time() - start_time))
	sys.exit(1)


	mode = int(sys.argv[1])
	case_index = int(sys.argv[2])
	animation_flag = int(sys.argv[3])
	LMP_motion = sys.argv[4]

	if mode == 1:
		curr_config, goal_config = regression_test(case_index)
		ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
		#ML_MCTS_ins.vis_whole_sequence()
		#ML_MCTS_ins.global_optimization()
		print(animation_flag)
		if animation_flag == 1:
			if LMP_motion == 'smart':
				ML_MCTS_ins.animate_whole_sequence_IROS()
			else:
				ML_MCTS_ins.animate_whole_sequence()
		print("Total length travelled: {0}".format(ML_MCTS_ins.calculate_total_length_travelled())) 
	elif mode == 2:
		curr_config, goal_config = test_case_reader(case_index)

		ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
		#ML_MCTS_ins.vis_whole_sequence()
		#ML_MCTS_ins.global_optimization()
		#ML_MCTS_ins.animate_whole_sequence()
		print("Time consumption: {0}".format(ML_MCTS_ins.time_consumption_)) 
		print("Total steps: {0}".format(ML_MCTS_ins.total_steps_)) 
		print("Total length travelled: {0}".format(ML_MCTS_ins.calculate_total_length_travelled_ICRA())) 
		print("Total length displacement: {0}".format(ML_MCTS_ins.calculate_total_length_displacement_ICRA()))
		res_plan = ML_MCTS_ins.save_planning_results()

		write_result('MS_MCTS_CONT_2', case_index, len(curr_config), ML_MCTS_ins.time_consumption_, ML_MCTS_ins.total_steps_, ML_MCTS_ins.calculate_total_length_travelled_ICRA(), ML_MCTS_ins.calculate_total_length_displacement_ICRA(), res_plan)
	else:
		for t in range(1, 10):
			curr_config, goal_config = regression_test(t)
			ML_MCTS_ins = multi_level_MCTS_algo(curr_config, goal_config)
			print("Total length travelled: {0}".format(ML_MCTS_ins.calculate_total_length_travelled())) 
