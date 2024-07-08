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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class mRS_Tree_Node():
	def __init__(self, curr_config, goal_config, obj_c, obj_past, obj_fut, occupied_region):
		self.curr_config_ = curr_config
		self.goal_config_ = goal_config
		self.occupied_region_ = occupied_region
		self.obj_c_ = obj_c
		self.obj_past_ = obj_past
		self.obj_fut_ = obj_fut
		self.parent_ = None
		self.children_ = []
		self.robot_ = [10.0, -2.0]
		self.robot_width_ = 2

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


	def traverse_tree(self):
		res = []
		start = self
		while start:
			res.append(start)
			start = start.parent_
		return res

	def collision_tunnel_object(self, tunnel, object_fut):
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
		
		for index in object_fut:
			cx, cy, radius, color = self.curr_config_[index]
			test_vector = np.array([cx, cy]) - v2_start
			proj_v2 = np.dot(v2, test_vector)
			proj_v3 = np.dot(v3, test_vector)

			if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
			   -height/2.0 - 1.0 < proj_v3 < height/2.0 + 1.0: 
				collision_items.add(index)
			
		return collision_items

	def tunnel_and_normal_visualizer(self, tunnel_list = None, object_in_collision = None, true_color = False, animation = False):
		region_x, region_y = [], []
		#for i in range(len(self.grid_)):
		#	for j in range(len(self.grid_[0])):
		#		region_x.append(i)
		#		region_y.append(j)
		if animation == False:
			plt.figure(figsize = (8.5, 18))
		#plt.scatter(region_x, region_y)
		for cx, cy, radius, color in self.curr_config_:
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



	def backward_checking(self, index):
		flag = False
		dummy_set = set()
		dummy_set.add(index)
		for tunnel in self.occupied_region_:
			if self.collision_tunnel_object(tunnel, dummy_set):
				flag = True
				break
		return flag
		
		
