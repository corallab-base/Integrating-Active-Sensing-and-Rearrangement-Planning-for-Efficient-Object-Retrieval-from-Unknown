#
#
#
#
#
#

import os
import sys
import math
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy

class PERTS_Tree_Node():
	def __init__(self, curr_config, goal_config, occupied_region):
		self.curr_config_ = curr_config
		self.goal_config_ = goal_config
		self.invalid_tracker_ = {}
		self.goal_invalid_tracker_ = {}
		self.occupied_region_ = occupied_region
		self.robot_ = [10.0, -2.0]
		self.robot_width_ = 2
		self.parent_ = None
		self.children_ = []
		self.grid_ = occupied_region

	def add_children(self, child):
		self.children_.append(child)

	def set_parent(self, parent):
		self.parent_ = parent

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

	def collision_tunnel_object(self, tunnel, object_list):
		#object_fut is a set
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
		
		for index in object_list:
			cx, cy, radius, color = self.curr_config_[index]
			test_vector = np.array([cx, cy]) - v2_start
			proj_v2 = np.dot(v2, test_vector)
			proj_v3 = np.dot(v3, test_vector)

			if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
			   -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
				collision_items.add(index)
			
		return collision_items

	def collision_tunnel_object_goal(self, tunnel, object_list):
		#object_fut is a set
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
		
		for index in object_list:
			cx, cy, radius, color = self.goal_config_[index]
			test_vector = np.array([cx, cy]) - v2_start
			proj_v2 = np.dot(v2, test_vector)
			proj_v3 = np.dot(v3, test_vector)

			if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
			   -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
				collision_items.add(index)
			
		return collision_items


	def detect_invalidity(self):
		object_list = [i for i in range(len(self.curr_config_))]
		for i in range(len(self.curr_config_)):
			cx, cy, radius, color = self.curr_config_[i]
			gx, gy, radius, color = self.goal_config_[i]

			grasp_tunnel = self.get_tunnel(self.robot_, self.curr_config_[i])	
			relocate_tunnel = self.get_tunnel(self.robot_, self.goal_config_[i])

			invalid_object_grasp = self.collision_tunnel_object(grasp_tunnel, object_list)
			invalid_object_relocate = self.collision_tunnel_object(relocate_tunnel, object_list)

			invalid_object = set()
			for element in invalid_object_grasp:
				if element != i:
					invalid_object.add(element)
			for element in invalid_object_relocate:
				if element != i:
					invalid_object.add(element)

			self.invalid_tracker_[i] = invalid_object

		return self.invalid_tracker_

	def detect_goal_invalidity(self):
		object_list = [i for i in range(len(self.goal_config_))]
		for i in range(len(self.goal_config_)):
			gx, gy, radius, color = self.goal_config_[i]

			grasp_tunnel = self.get_tunnel(self.robot_, self.curr_config_[i])	
			relocate_tunnel = self.get_tunnel(self.robot_, self.goal_config_[i])

			invalid_object_grasp = self.collision_tunnel_object_goal(grasp_tunnel, object_list)
			invalid_object_relocate = self.collision_tunnel_object_goal(relocate_tunnel, object_list)

			invalid_object = set()
			for element in invalid_object_grasp:
				if element != i:
					invalid_object.add(element)
			for element in invalid_object_relocate:
				if element != i:
					invalid_object.add(element)

			self.goal_invalid_tracker_[i] = invalid_object

		return self.goal_invalid_tracker_


	def pertube_node(self):
		object_list = [i for i in range(len(self.curr_config_))]
		
		while True:
			randomly_selected_object = np.random.randint(len(self.curr_config_))
			cx, cy, radius, color = self.curr_config_[randomly_selected_object]
			
			grasp_tunnel = self.get_tunnel(self.robot_, self.curr_config_[randomly_selected_object])
			invalid_object_grasp  = self.collision_tunnel_object(grasp_tunnel, object_list)

			if len(invalid_object_grasp) == 1:
				random_loc_x = np.random.randint(20) + round(random.gauss(0, 1), 2)
				random_loc_y = np.random.randint(20) + round(random.gauss(0, 1), 2)

				while (random_loc_x < 0 or random_loc_x > 19 or random_loc_y < 0 or random_loc_y > 19):
					random_loc_x = np.random.randint(20) + round(random.gauss(0, 1), 2)
					random_loc_y = np.random.randint(20) + round(random.gauss(0, 1), 2)
				
				relocate_tunnel = self.get_tunnel(self.robot_, [random_loc_x, random_loc_y])
				invalid_object_relocate = self.collision_tunnel_object(relocate_tunnel, object_list)

				other_object = 0

				for element in invalid_object_relocate:
					if element != randomly_selected_object:
						other_object += 1
						break

				if other_object == 0:
					new_curr_config = deepcopy(self.curr_config_)
					new_curr_config[randomly_selected_object][0] = random_loc_x
					new_curr_config[randomly_selected_object][1] = random_loc_y
					return new_curr_config
				else:
					continue
			else:
				continue

	def get_object_not_at_goal(self):
		res = []
		for i in range(len(self.curr_config_)):
			if self.curr_config_[i] != self.goal_config_[i]:
				res.append(i)
		return res

	def goal_reached(self):
		return self.curr_config_ == self.goal_config_


	def traverse_tree(self):
		res = []
		start = self
		while start:
			res.append(start)
			start = start.parent_
		return res

	def tunnel_and_normal_visualizer(self, tunnel_list = None, object_in_collision = None, true_color = False, animation = False):
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
		#for cx, cy, radius, color in self.static_config_:
		#	if not true_color:
		#		temp_circle = mpatches.Circle((cx, cy), radius, color = 'black')
		#		temp_circle_inner = mpatches.Circle((cx, cy), radius*0.7, color =  color)
		#		plt.gca().add_patch(temp_circle)
		#		plt.gca().add_patch(temp_circle_inner)
		#	else:
		#		temp_circle = mpatches.Circle((cx, cy), radius, color = color)
		#		plt.gca().add_patch(temp_circle)

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
				if index != 0:
					cx, cy, radius, color = self.curr_config_[index]
					temp_square = mpatches.Rectangle((cx - radius, cy - radius), radius*2, radius*2, alpha = 0.5, color = 'm')
					plt.gca().add_patch(temp_square)

		plt.xlim(-1, 20)
		plt.ylim(-4, 20)
		if animation == False:
			plt.show()
		else:
			plt.pause(0.06)



if __name__ == '__main__':
	curr_config = [[10, 16, 1, 'r'], [10, 12, 1, 'g'], [10, 8, 1, 'b'], [10, 4, 1, 'y']]
	goal_config = [[10, 4, 1, 'r'], [10, 8, 1, 'g'], [10, 12, 1, 'b'], [10, 16, 1, 'y']]
	PERTS_tree_node_ins = PERTS_Tree_Node(curr_config, goal_config, None)
	PERTS_tree_node_ins.detect_invalidity()
	print(PERTS_tree_node_ins.invalid_tracker_)

		
