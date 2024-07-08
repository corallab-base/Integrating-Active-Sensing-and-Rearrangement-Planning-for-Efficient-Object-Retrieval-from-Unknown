#
# file   mRS.py
# brief  mRS(monotone rearrangement solver) implementation
# author Hanwen Ren
# date   2023-03-07
#

#stat_paper_info*********************************************
#Manipulation Planning Among Movable Obstacles
#Under Section III.B Simplifying Assumptions
#Quote: We first assume the problem is monotone or that if
#a solution exists, if can be found by moving each obstacle
#once. Consequently we need not consider plans longer than 
#the number of movable obstacles.
#end_paper_info*********************************************


import os
import sys
import math
import time
import random
from mRS_tree_node import mRS_Tree_Node
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rearrangement_planning_util_ICRA import regression_test
import numpy as np
from copy import deepcopy
from test_case_generator import test_case_reader
from rearrangement_planning_util_ICRA import write_result


class mRS_RSC():
	def __init__(self, curr_config, goal_config):
		#they are stored in reverse order
		#because this is a backtracking search method 
		#starting fromt the goal configuration
		self.curr_config_ = goal_config
		self.goal_config_ = curr_config
		self.track_level_steps_ = []


	def search(self):
		#implement based on the pseudocode presented in the 
		#original paper

		#try each object as the goal
		plan = []
		for i in range(len(self.curr_config_)):
			#create new root specifing this is the last to move
			#only this object is not in place
			temp_curr_config = deepcopy(self.curr_config_)
			temp_goal_config = deepcopy(self.goal_config_)

			temp_root = mRS_Tree_Node(temp_curr_config, temp_goal_config, i, set(), set(), [])

			res = self.step(temp_root)
			if res:
				whole_sequence = res.traverse_tree()
				plan = whole_sequence
				break
		if plan:
			self.track_level_steps_.append(plan)
			return True
		else:
			#print('No solution found!')
			return False

			
	#this is from the RSC algorithm in the original paper
	def step(self, node):
		cx, cy, radius, color = node.curr_config_[node.obj_c_]
		gx, gy, radius, color = node.goal_config_[node.obj_c_]

		grasp_tunnel = node.get_tunnel(node.robot_, [cx, cy])
		relocate_tunnel = node.get_tunnel(node.robot_, [gx, gy])

		#first criteria
		obj_col_grasp = node.collision_tunnel_object(grasp_tunnel, node.obj_fut_)
		obj_col_relocate = node.collision_tunnel_object(relocate_tunnel, node.obj_fut_)

		#backward checking
		backward_fail_flag = node.backward_checking(node.obj_c_)

		if obj_col_grasp or obj_col_relocate or backward_fail_flag:
			return None
		else:
			#all objects can be potential candidates
			candidates = set([i for i in range(len(node.curr_config_))])
			obj_col_grasp = node.collision_tunnel_object(grasp_tunnel, candidates)
			obj_col_relocate = node.collision_tunnel_object(relocate_tunnel, candidates)

			new_obj_past = deepcopy(node.obj_past_)
			new_obj_fut = deepcopy(node.obj_fut_)
			new_curr_config = deepcopy(node.curr_config_)
			new_goal_config = deepcopy(node.goal_config_)
			new_curr_config[node.obj_c_] = new_goal_config[node.obj_c_]
			new_occupied_region = deepcopy(node.occupied_region_)
			new_occupied_region += [grasp_tunnel, relocate_tunnel]
			new_obj_fut.add(node.obj_c_)

			for element in obj_col_grasp:
				if element != node.obj_c_:
					new_obj_past.add(element)
			for element in obj_col_relocate:
				if element != node.obj_c_:
					new_obj_past.add(element)

			if not new_obj_past:
				obj_not_in_place = []
				for i in range(len(new_curr_config)):
					if new_curr_config[i] != new_goal_config[i]:
						obj_not_in_place.append(i)
				if not obj_not_in_place:
					#need to add a dummy node to show the final configuration(initial configration in the problem setting)
					new_node = mRS_Tree_Node(new_curr_config, new_goal_config, -1, new_obj_past, new_obj_fut, new_occupied_region)
					new_node.parent_ = node
					node.children_.append(new_node)
					return new_node
				else:
					for element in obj_not_in_place:
						new_node = mRS_Tree_Node(new_curr_config, new_goal_config, element, new_obj_past, new_obj_fut, new_occupied_region)
						new_node.parent_ = node
						node.children_.append(new_node)
						temp_res = self.step(new_node)
						if temp_res:
							return temp_res
					return None
			else:
				for element in new_obj_past:
					second_obj_past = deepcopy(new_obj_past)
					second_obj_past.remove(element)
					new_node = mRS_Tree_Node(new_curr_config, new_goal_config, element, second_obj_past, new_obj_fut, new_occupied_region)
					new_node.parent_ = node
					node.children_.append(new_node)
					temp_res = self.step(new_node)
					if temp_res:
						return temp_res
				return None
			return None
			
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

	def get_stats(self):
		steps = 0
		distance_travel = 0
		distance_displace = 0
		path = []
		for tree_nodes in self.track_level_steps_:
			steps += len(tree_nodes) - 1
			for i in range(len(tree_nodes)):
				path.append(tree_nodes[i].curr_config_)
			for i in range(len(tree_nodes)-1):
				start_config = tree_nodes[i].curr_config_
				end_config = tree_nodes[i+1].curr_config_
				move_index = None
				for t in range(len(start_config)):
					if start_config[t] != end_config[t]:
						move_index = t
						break
				if move_index == None:
					move_index = 0
				start_x, start_y, _, _ = start_config[move_index]
				robot_x, robot_y = tree_nodes[i].robot_
				end_x, end_y, _, _ = end_config[move_index]

				distance_travel += math.sqrt((start_x - robot_x)**2 + (start_y - robot_y)**2) + math.sqrt((end_x - robot_x)**2 + (end_y - robot_y)**2)
				distance_displace += math.sqrt((start_x - end_x)**2 + (start_y - end_y)**2)
		return steps, distance_travel, distance_displace, path

				

	def animate_whole_sequence(self):
		#create a global canvas
		plt.figure(figsize = (8.5, 18))
		step_size = 1.0

		for tree_nodes in self.track_level_steps_:
			plt.clf()
			tree_nodes[0].tunnel_and_normal_visualizer(animation = True)
			for i in range(len(tree_nodes)-1):
				start_config = tree_nodes[i].curr_config_
				end_config = tree_nodes[i+1].curr_config_
				print(start_config, end_config)
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
				#dummy_grid = tree_nodes[i].grid_
				#dummy_static_config = tree_nodes[i].static_config_
				
				grasp_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, start_config[move_index])
				relocate_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, end_config[move_index])

				#grasp tunnel animation
				distance_start = math.sqrt((start_x - robot_x)**2 + (start_y - robot_y)**2)
				start_steps = math.ceil(distance_start / step_size)
				delta_x, delta_y = (robot_x - start_x) / start_steps, (robot_y - start_y) / start_steps

				dummy_tree_node = mRS_Tree_Node(dummy_curr_config, dummy_goal_config, 0, set(), set(), [])
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

class final_tree_node():
	def __init__(self, config, parent, steps, travel, displace, path):
		self.config_ = config
		self.parent_ = parent
		self.steps_ = steps
		self.travel_ = travel
		self.displace_ = displace
		self.path_ = path


class RRT_extension():
	def __init__(self, curr_config, goal_config):
		self.curr_config_ = curr_config
		self.goal_config_ = goal_config
		self.root_ = final_tree_node(self.curr_config_, None, 0, 0, 0, [])
		self.goal_ = final_tree_node(self.goal_config_, None, 0, 0, 0, [])
		self.all_configs_ = [self.root_]

	def get_distance(self, config1, config2):
		res = 0
		for i in range(len(config1)):
			x1, y1, _, _ = config1[i]
			x2, y2, _, _ = config2[i]
			res += (x1 - x2)**2 + (y1 - y2)**2
		return res

	def sample(self):
		new_config = []
		for i in range(len(self.curr_config_)):
			_, _, radius, color = self.curr_config_[i]
			new_x = np.random.randint(20) + round(random.gauss(0, 1), 2)
			new_y = np.random.randint(20) + round(random.gauss(0, 1), 2)

			if (new_x < 0 or new_x > 19 or new_y < 0 or new_y > 19):
				new_x = np.random.randint(20) + round(random.gauss(0, 1), 2)
				new_y = np.random.randint(20) + round(random.gauss(0, 1), 2)

			new_config.append([new_x, new_y, radius, color])
		if self.get_distance(new_config, self.goal_config_) < 50:
			new_config = self.goal_config_
		return new_config

	def steer_to(self, config1, config2):
		mRS_RSC_ins = mRS_RSC(config1, config2)
		flag = mRS_RSC_ins.search()
		steps, travel, displace, path = sys.maxsize, sys.maxsize, sys.maxsize, None
		if flag:
			steps, travel, displace, path = mRS_RSC_ins.get_stats()
		return flag, steps, travel, displace, path

	def get_nearest(self, config):
		min_distance = sys.maxsize
		res = None
		for element in self.all_configs_:
			exist_config = element.config_
			temp_distance = self.get_distance(config, exist_config)
			if temp_distance < min_distance:
				min_distance = temp_distance
				res = element
		return res

	def get_nearest_2(self, config, tree_list):
		min_distance = sys.maxsize
		res = None
		for element in tree_list:
			exist_config = element.config_
			temp_distance = self.get_distance(config, exist_config)
			if temp_distance < min_distance:
				min_distance = temp_distance
				res = element
		return res

	def non_monotone_search(self):
		#root = self.curr_config_
		res = None
		while True:
			q_rand_config = self.sample()
			q_near = self.get_nearest(q_rand_config)
			flag, steps, travel, displace, path = self.steer_to(q_rand_config, q_near.config_)
			if flag:
				q_rand = final_tree_node(q_rand_config, q_near, steps, travel, displace, path)
				self.all_configs_.append(q_rand)
				if q_rand.config_ == self.goal_config_:
					res = q_rand
					break

		total_steps, total_travel, total_displace = 0.0, 0.0, 0.0
		while res:
			total_steps += res.steps_
			total_travel += res.travel_
			total_displace += res.displace_
			res = res.parent_

		return total_steps, total_travel, total_displace

	def non_monotone_bi_search(self):
		res1, res2 = None, None
		t1 = self.root_
		t2 = self.goal_
		t1_list = [t1]
		t2_list = [t2]

		while True:
			q_rand_config = self.sample()
			q_near = self.get_nearest_2(q_rand_config, t1_list)
			flag, steps, travel, displace, path = self.steer_to(q_rand_config, q_near.config_)
			if flag:
				q_rand = final_tree_node(q_rand_config, q_near, steps, travel, displace, path)
				t1_list.append(q_rand)
				q_near2 = self.get_nearest_2(q_rand_config, t2_list)
				flag2, steps2, travel2, displace2, path2 = self.steer_to(q_rand_config, q_near2.config_)
				if flag2:
					q_rand2 = final_tree_node(q_rand_config, q_near2, steps2, travel2, displace2, path2)
					res1, res2 = q_rand, q_rand2
					break
				if len(t1_list) < len(t2_list):
					t1_list, t2_list = t2_list, t1_list

		total_steps, total_travel, total_displace = 0.0, 0.0, 0.0
		total_path1 = []
		total_path2 = []

		while res1:
			total_steps += res1.steps_
			total_travel += res1.travel_
			total_displace += res1.displace_
			total_path1 += res1.path_
			res1 = res1.parent_
		while res2:
			total_steps += res2.steps_
			total_travel += res2.travel_
			total_displace += res2.displace_
			total_path2 += res2.path_
			res2 = res2.parent_

		return total_steps, total_travel, total_displace, np.array(total_path1[::-1] + total_path2)
		
	


if __name__ == '__main__':
	mode = int(sys.argv[1])
	case_index = int(sys.argv[2])

	curr_config, goal_config = test_case_reader(case_index)

	#mRS_RSC_ins = mRS_RSC(curr_config, goal_config)

	#mRS_RSC_ins.search()

	#if mRS_RSC_ins.track_level_steps_:
	#	mRS_RSC_ins.animate_whole_sequence()
	start_time = time.time()

	rrt_mrs_ins = RRT_extension(curr_config, goal_config)

	print(curr_config, goal_config)

	#total_steps, total_travel, total_displace = rrt_mrs_ins.non_monotone_search()
	total_steps, total_travel, total_displace, total_path = rrt_mrs_ins.non_monotone_bi_search()

	end_time = time.time()

	print('Time comsuption: {0}'.format(end_time - start_time))
	print('Total steps: {0}'.format(total_steps))
	print('Total length travelled: {0}'.format(total_travel))
	print('Total length displacement: {0}'.format(total_displace))

	#print(total_path)

	write_result('BIRRT_MRS', case_index, len(curr_config), end_time - start_time, total_steps, total_travel, total_displace, total_path)

