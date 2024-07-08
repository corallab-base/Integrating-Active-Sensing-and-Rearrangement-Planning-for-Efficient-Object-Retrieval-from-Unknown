#
# file
# brief
# author
# date
#

#start_paper_info**************************
#end_paper_info****************************

import os
import sys
import math
import numpy as np
import time
from PERTS_tree_node import PERTS_Tree_Node
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rearrangement_planning_util_ICRA import regression_test
from rearrangement_planning_util_ICRA import write_result
from copy import deepcopy
from test_case_generator import test_case_reader

class CIRS():
	#monotone assumption
	def __init__(self, curr_config, goal_config):
		self.curr_config_ = curr_config
		self.goal_config_ = goal_config
		self.grid_ = []
		for i in range(20):
			temp_arr = []
			for j in range(20):
				temp_arr.append(0)
			self.grid_.append(temp_arr)
		self.root_ = PERTS_Tree_Node(self.curr_config_, self.goal_config_, self.grid_)
		self.leaf_nodes_ = set()
		self.total_steps_ = 0
		self.invalid_tracker_ = self.root_.detect_invalidity()
		self.goal_invalid_tracker_ = self.root_.detect_goal_invalidity()

	def perform_search(self):
		flag, end_node = self.search(self.root_)
		return flag, end_node

	def get_root(self):
		return self.root_

	#can be viewed as a DFs like approach
	def search(self, start_node):
		#get all invalid states
		#invalid_tracker = start_node.detect_invalidity()

		object_list = start_node.get_object_not_at_goal()

		object_moved = [i for i in range(len(self.curr_config_)) if i not in object_list]

		for index in object_list:
			flag1 = False
			flag2 = False

			if index in self.goal_invalid_tracker_:
				for temp_object in object_moved:
					if temp_object in self.goal_invalid_tracker_[index]:
						flag1 = True
						break

			if index in self.invalid_tracker_:
				for temp_object in object_list:
					if temp_object in self.invalid_tracker_[index]:
						flag2 = True
						break

			if not flag1 and not flag2:
				new_curr_config = deepcopy(start_node.curr_config_)
				new_goal_config = deepcopy(start_node.goal_config_)
				new_curr_config[index] = new_goal_config[index]
				new_node = PERTS_Tree_Node(new_curr_config, new_goal_config, self.grid_)
				start_node.add_children(new_node)
				new_node.set_parent(start_node)
				if not new_node.goal_reached():
					flag, end_node = self.search(new_node)
					if flag: 
						return flag, end_node
				else:
					self.leaf_nodes_.add(new_node)
					return True, new_node

		self.leaf_nodes_.add(start_node)
		return False, start_node


class PERTS_CIRS():
	def __init__(self, curr_config, goal_config):
		self.curr_config_ = curr_config
		self.goal_config_ = goal_config
		self.root_ = None
		self.leaf_nodes_ = set()

		CIRS_ins = CIRS(self.curr_config_, self.goal_config_)
		flag, end_node = CIRS_ins.perform_search()

		self.root_ = CIRS_ins.get_root()
		self.end_node_ = end_node
		self.track_level_steps_  = []
		self.total_length_travelled_ = 0.0
		self.total_length_displacement_ = 0.0


	def get_random_leaf_nodes(self):
		leaf_nodes = []
		start = [self.root_]
		while start:
			new_start = []
			for element in start:
				if not element.children_:
					leaf_nodes.append(element)
				else:
					new_start += element.children_
			start = new_start
		rand_index = np.random.randint(len(leaf_nodes))
		return leaf_nodes[rand_index]

	def concat_tree(self, start_node, new_root):
		start_node.add_children(new_root)
		new_root.set_parent(start_node)
		#print(start_node.curr_config_, new_root.curr_config_)

	def find_end_node(self):
		counter = 0
		start = [self.root_]
		while start:
			new_start = []
			for element in start:
				counter += 1
				if element.goal_reached():
					self.end_node_ = element
					return
				else:
					new_start += element.children_
			start = new_start

	def search(self):
		#check check if search is needed
		if self.end_node_.goal_reached():
			plan = self.end_node_.traverse_tree()[::-1]
			self.track_level_steps_.append(plan)
			print('Total steps: {0}'.format(len(plan)))
		else:
			#perform search
			flag = False
			while not flag:
				new_start_node = self.get_random_leaf_nodes()
				
				new_next_curr_config = new_start_node.pertube_node()

				CIRS_ins = CIRS(new_next_curr_config, self.goal_config_)
				flag, end_node = CIRS_ins.perform_search()

				if flag: 
					new_root = CIRS_ins.get_root()
					self.concat_tree(new_start_node, new_root)

					self.find_end_node()
					break
				else:

					new_root = CIRS_ins.get_root()
					self.concat_tree(new_start_node, new_root)



		plan = self.end_node_.traverse_tree()[::-1]
		self.track_level_steps_ = [plan]
		#self.track_level_steps_.append(plan)
		self.total_steps_ = len(plan)

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
				rx, ry = tree_nodes[i].robot_
				move_index = None
				for t in range(len(start_config)):
					if start_config[t] != end_config[t]:
						move_index = t
						break
				#remove this logic for PERTS because it is possible to sample the same config
				#if move_index == None:
				#	print('Tree Traversal Error!')
				#	sys.exit(1)
				
				if move_index != None:
					self.total_length_travelled_ += math.sqrt((start_config[move_index][0] - rx)**2 + (start_config[move_index][1] - ry)**2)
					self.total_length_travelled_ += math.sqrt((end_config[move_index][0] - rx)**2 + (end_config[move_index][1] - ry)**2)

		return self.total_length_travelled_

	def calculate_total_length_displacement(self):
		self.total_length_displacement_ = 0.0
		for tree_nodes in self.track_level_steps_:
			for i in range(len(tree_nodes)-1):
				start_config = tree_nodes[i].curr_config_
				end_config = tree_nodes[i+1].curr_config_
				rx, ry = tree_nodes[i].robot_
				move_index = None
				for t in range(len(start_config)):
					if start_config[t] != end_config[t]:
						move_index = t
						break
				#remove this logic for PERTS because it is possible to sample the same config
				#if move_index == None:
				#	print('Tree Traversal Error!')
				#	sys.exit(1)
				
				if move_index != None:
					self.total_length_displacement_ += math.sqrt((start_config[move_index][0] - end_config[move_index][0])**2 + (start_config[move_index][1] - end_config[move_index][1])**2)

		return self.total_length_displacement_


	




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
				#dummy_static_config = tree_nodes[i].static_config_
				
				grasp_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, start_config[move_index])
				relocate_tunnel = tree_nodes[i].get_tunnel(tree_nodes[i].robot_, end_config[move_index])

				self.total_length_travelled_ = math.sqrt((start_config[move_index][0] - end_config[move_index][0])**2 + (start_config[move_index][1] - end_config[move_index][1])**2)

				#grasp tunnel animation
				distance_start = math.sqrt((start_x - robot_x)**2 + (start_y - robot_y)**2)
				start_steps = math.ceil(distance_start / step_size)
				delta_x, delta_y = (robot_x - start_x) / start_steps, (robot_y - start_y) / start_steps

				dummy_tree_node = PERTS_Tree_Node(dummy_curr_config, dummy_goal_config, dummy_grid)
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



if __name__ == '__main__':
	mode = int(sys.argv[1])
	case_index = int(sys.argv[2])

	#remove the old result
	old_file_name = './test_results/PERTS_NEW_2/test_result_' + str(case_index) + '.txt'
	if os.path.exists(old_file_name):
		print('Remove old result')
		os.remove(old_file_name)

	curr_config, goal_config = test_case_reader(case_index)
	print(curr_config, goal_config)
	#curr_config, goal_config = regression_test(case_index)

	start_time = time.time()

	PERTS_CIRS_ins = PERTS_CIRS(curr_config, goal_config)

	PERTS_CIRS_ins.search()
	
	end_time = time.time()

	res_plan = PERTS_CIRS_ins.save_planning_results()

	print ('Time comsuption: {0}'.format(end_time - start_time))
	print ('Total steps: {0}'.format(PERTS_CIRS_ins.total_steps_))
	print ('Total length travelled: {0}'.format(PERTS_CIRS_ins.calculate_total_length_travelled()))
	print ('Total length displacement: {0}'.format(PERTS_CIRS_ins.calculate_total_length_displacement()))

	write_result('PERTS', case_index, len(curr_config), end_time - start_time, PERTS_CIRS_ins.total_steps_, PERTS_CIRS_ins.calculate_total_length_travelled(), PERTS_CIRS_ins.calculate_total_length_displacement(), res_plan)


	#PERTS_CIRS_ins.animate_whole_sequence()

