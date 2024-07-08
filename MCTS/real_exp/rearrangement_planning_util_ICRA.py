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

class Tree_Node():
	def __init__(self, current_config, goal_config, current_grid, static_config = [], total_distance = 0):
		self.reward_ = 0.0
		self.visited_ = 0.0
		self.parent_ = None
		self.children_ = []
		self.curr_config_ = current_config
		self.goal_config_ = goal_config
		self.static_config_ = static_config
		self.grid_ = current_grid
		self.robot_ = [20.0, -12.37]
		self.robot_width_right_ = 5
		self.robot_width_left_ = 3
		self.object_in_collision_ = None
		self.total_distance_ = total_distance
		self.x_min_ = 1
		self.x_max_ = 39
		self.y_min_ = 0
		self.y_max_ = 10

		if not self.is_goal_config():
			self.get_blocking_object()
			#print('check collision_object')
			#print(self.object_in_collision_, self.curr_config_, self.goal_config_)
			#print ('current blocking')
			#print (self.object_in_collision_)
			#print ('end current blocking')

		self.distance_lookup_ = defaultdict(list)
		for t in range(-19, 20):
			for k in range(-19, 20):
				distance = round((t)**2 + (k)**2, 3)
				self.distance_lookup_[distance].append([t, k])
		self.distance_lookup_ = [list(x) for x in self.distance_lookup_.items()]
		self.distance_lookup_.sort(key = lambda x: x[0])

	def add_reward(self, reward):
		self.reward_ += reward
		self.visited += 1.0

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
		v3_end = np.array([center_point[0] - np.sin(angle)*self.robot_width_left_,
											 center_point[1] + np.cos(angle)*self.robot_width_left_])

		v4_start = np.array(v2_start)
		v4_end = np.array([center_point[0] + np.sin(angle)*self.robot_width_right_,
											 center_point[1] - np.cos(angle)*self.robot_width_right_])

		#bottom left point, width, height, rotation degree, normal1_start, normal1_end, normal2_start, normal2_end
		return [(self.robot_[0] + self.robot_width_right_*np.sin(angle), self.robot_[1] - self.robot_width_right_*np.cos(angle)), np.sqrt(dy**2 + dx**2)+1, self.robot_width_left_ + self.robot_width_right_, np.degrees(angle), v2_start, v2_end, v3_start, v3_end, v4_start, v4_end]


	def collision_tunnel_static(self, tunnel):
		_, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end, v4_start, v4_end = tunnel
		angle = np.radians(angle_degree)
		
		v2 = v2_end - v2_start
		v2 = v2 / np.linalg.norm(v2)
		v3 = v3_end - v3_start
		v3 = v3 / np.linalg.norm(v3)
		v4 = v4_end - v4_start
		v4 = v4 / np.linalg.norm(v4)
		
		collision_items = set()
		
		if np.dot(v2, v3) > 1e-6 or np.dot(v2, v4) > 1e-6:
			print('tunnel axes calculation fails\n')
			sys.exit(1)
		
		for i in range(len(self.static_config_)):
			cx, cy, radius, color = self.static_config_[i]
			test_vector = np.array([cx, cy]) - v2_start
			proj_v2 = np.dot(v2, test_vector)
			proj_v3 = np.dot(v3, test_vector)
			proj_v4 = np.dot(v4, test_vector)
		
			if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
				 (0 <= proj_v3 < self.robot_width_left_  + 1.0 or
				  0 <= proj_v4 < self.robot_width_right_ + 1.0): 
				return True

		#the goal of current MCTS objective
		cx, cy, radius, color = self.goal_config_[0]
		test_vector = np.array([cx, cy]) - v2_start
		proj_v2 = np.dot(v2, test_vector)
		proj_v3 = np.dot(v3, test_vector)
		proj_v4 = np.dot(v4, test_vector)

		if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
			 (0 <= proj_v3 < self.robot_width_left_  + 1.0 or
			  0 <= proj_v4 < self.robot_width_right_ + 1.0): 
			return True
			
		return False

	def collision_tunnel_object_spec_item(self, tunnel, item_index):
		_, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end, v4_start, v4_end= tunnel
		angle = np.radians(angle_degree)

		v2 = v2_end - v2_start
		v2 = v2 / np.linalg.norm(v2)
		v3 = v3_end - v3_start
		v3 = v3 / np.linalg.norm(v3)
		v4 = v4_end - v4_start
		v4 = v4 / np.linalg.norm(v4)

		collision_items = set()

		if np.dot(v2, v3) > 1e-6 or np.dot(v2, v4) > 1e-6:
			print('tunnel axes calculation fails\n')
			sys.exit(1)
		
		cx, cy, radius, color = self.goal_config_[item_index]
		test_vector = np.array([cx, cy]) - v2_start
		proj_v2 = np.dot(v2, test_vector)
		proj_v3 = np.dot(v3, test_vector)
		proj_v4 = np.dot(v4, test_vector)

		if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
				 (0 <= proj_v3 < self.robot_width_left_  + 1.0 or
				  0 <= proj_v4 < self.robot_width_right_ + 1.0): 
				return True

		return False


	def get_dependency_relation(self):
		res = defaultdict(set)
		
		for i in range(len(self.goal_config_)):
			relocate_tunnel = self.get_tunnel(self.robot_, self.goal_config_[i])
			collision_objs = self.collision_tunnel_object_goal(relocate_tunnel)
			collision_objs = [x for x in collision_objs if x != i]
			for obj in collision_objs:
				res[obj].add(i)

		return res


	def collision_tunnel_object_goal(self, tunnel):

		_, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end, v4_start, v4_end = tunnel
		angle = np.radians(angle_degree)

		v2 = v2_end - v2_start
		v2 = v2 / np.linalg.norm(v2)
		v3 = v3_end - v3_start
		v3 = v3 / np.linalg.norm(v3)
		v4 = v4_end - v4_start
		v4 = v4 / np.linalg.norm(v4)

		collision_items = set()

		if np.dot(v2, v3) > 1e-6 or np.dot(v2, v4) > 1e-6:
			print('tunnel axes calculation fails\n')
			sys.exit(1)
		
		for i in range(len(self.curr_config_)):
			cx, cy, radius, color = self.curr_config_[i]
			test_vector = np.array([cx, cy]) - v2_start
			proj_v2 = np.dot(v2, test_vector)
			proj_v3 = np.dot(v3, test_vector)
			proj_v4 = np.dot(v4, test_vector)
			
			if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
				 (0 <= proj_v3 < self.robot_width_left_  + 1.0 or
				  0 <= proj_v4 < self.robot_width_right_ + 1.0): 
				collision_items.add(i)
			
		return list(collision_items)


	def collision_tunnel_object(self, tunnel):
		_, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end, v4_start, v4_end = tunnel
		angle = np.radians(angle_degree)

		v2 = v2_end - v2_start
		v2 = v2 / np.linalg.norm(v2)
		v3 = v3_end - v3_start
		v3 = v3 / np.linalg.norm(v3)
		v4 = v4_end - v4_start
		v4 = v4 / np.linalg.norm(v4)

		collision_items = set()

		if np.dot(v2, v3) > 1e-6 or np.dot(v2, v4) > 1e-6:
			print('tunnel axes calculation fails\n')
			sys.exit(1)
		
		for i in range(len(self.curr_config_)):
			cx, cy, radius, color = self.curr_config_[i]
			test_vector = np.array([cx, cy]) - v2_start
			proj_v2 = np.dot(v2, test_vector)
			proj_v3 = np.dot(v3, test_vector)
			proj_v4 = np.dot(v4, test_vector)
			
			if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
				 (0 <= proj_v3 < self.robot_width_left_  + 1.0 or
				  0 <= proj_v4 < self.robot_width_right_ + 1.0): 
				collision_items.add(i)
			
		return list(collision_items)

	def collision_tunnel_region(self, tunnel, new_region):
		_, width, height, angle_degree, v2_start, v2_end, v3_start, v3_end, v4_start, v4_end = tunnel
		angle = np.radians(angle_degree)
	
		v2 = v2_end - v2_start
		v2 = v2 / np.linalg.norm(v2)
		v3 = v3_end - v3_start
		v3 = v3 / np.linalg.norm(v3)
		v4 = v4_end - v4_start
		v4 = v4 / np.linalg.norm(v4)

		if np.dot(v2, v3) > 1e-6 or np.dot(v2, v4) > 1e-6:
			print('tunnel axes calculation fails\n')
			sys.exit(1)

		cx, cy = new_region[0], new_region[1]
		test_vector = np.array([cx, cy]) - v2_start
		proj_v2 = np.dot(v2, test_vector)
		proj_v3 = np.dot(v3, test_vector)
		proj_v4 = np.dot(v4, test_vector)
	
		if -width/2.0 - 1.0 < proj_v2 < width/2.0 + 1.0 and \
				 (0 <= proj_v3 < self.robot_width_left_  + 1.0 or
				  0 <= proj_v4 < self.robot_width_right_ + 1.0): 
			return True
		else:
			return False


	def tunnel_and_normal_visualizer(self, tunnel_list = None, object_in_collision = None, true_color = False, animation = False):
		region_x, region_y = [], []
		for i in range(self.x_min_, self.x_max_ + 1):
			for j in range(self.y_min_, self.y_max_ + 1):
				region_x.append(i)
				region_y.append(j)
		if animation == False:
			plt.figure(figsize = (20, 18))
		plt.scatter(region_x, region_y)
		for cx, cy, radius, color in self.curr_config_:
			if '_' in color: color = color[:color.index('_')]
			temp_circle = mpatches.Circle((cx, cy), radius, color = color)
			plt.gca().add_patch(temp_circle)
		for cx, cy, radius, color in self.static_config_:
			if '_' in color: color = color[:color.index('_')]
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

			for start_corner, width, height, angle, v2_start, v2_end, v3_start, v3_end, v4_start, v4_end in tunnel_list:
				tunnel_shape = mpatches.Rectangle(start_corner, width, height, angle, alpha = 0.5, color = tunnel_color[tunnel_counter])
				plt.gca().add_patch(tunnel_shape)
				plt.plot([v2_start[0], v2_end[0]], [v2_start[1], v2_end[1]], color = 'r')
				plt.plot([v3_start[0], v3_end[0]], [v3_start[1], v3_end[1]], color = 'g')
				plt.plot([v4_start[0], v4_end[0]], [v4_start[1], v4_end[1]], color = 'b')
				tunnel_counter += 1

		if object_in_collision:
			for index in object_in_collision:
				if index != 0:
					cx, cy, radius, color = self.curr_config_[index]
					temp_square = mpatches.Rectangle((cx - radius, cy - radius), radius*2, radius*2, alpha = 0.5, color = 'm')
					plt.gca().add_patch(temp_square)

		plt.xlim(-1, 40)
		plt.ylim(-8, 20)
		if animation == False:
			plt.show()
		else:
			plt.pause(0.01)


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

		#self.tunnel_and_normal_visualizer([grasp_tunnel, relocate_tunnel], self.object_in_collision_)

	def test_new_region_blocking(self, new_region, obs = None):
		start_pose = self.curr_config_[0]
		goal_pose = self.goal_config_[0]

		grasp_tunnel = self.get_tunnel(self.robot_, start_pose)
		relocate_tunnel = self.get_tunnel(self.robot_, goal_pose)

		flag1 = self.collision_tunnel_region(grasp_tunnel, new_region)
		flag2 = self.collision_tunnel_region(relocate_tunnel, new_region)
		new_region_relocate_tunnel = self.get_tunnel(self.robot_, new_region)

		flag4 = self.collision_tunnel_static(new_region_relocate_tunnel)
		flag3 = False

		if obs:
			for ob in obs:
				obs_start_pose = self.curr_config_[ob]
				obs_grasp_tunnel = self.get_tunnel(self.robot_, obs_start_pose)
				flag3 = self.collision_tunnel_region(obs_grasp_tunnel, new_region)
				if flag3: return True

		if flag1 or flag2 or flag3 or flag4:
			return True
		else:
			return False

	def propose_new_region(self, index, obs):
		#current set to return two regions
    #need some good guidence here
		gx, gy, radius, color = self.curr_config_[index]
		res = []
		for distance, offset_list in self.distance_lookup_:
			for ox, oy in offset_list:
				temp_x = gx + ox #+ round(random.gauss(0, 0.1), 2)
				temp_y = gy + oy #+ round(random.gauss(0, 0.1), 2)
				#if (self.x_min_ <= gx + ox <= self.x_max_ and self.y_min_ < gy + oy <= self.y_max_):
				#	while (temp_x < self.x_min_ or temp_x > self.x_max_ or temp_y < self.y_min_ or temp_y > self.y_max_):
				#		temp_x = gx + ox + round(random.gauss(0, 0.1), 2)
				#		temp_y = gy + oy + round(random.gauss(0, 0.1), 2)

				if (self.x_min_ <= temp_x <= self.x_max_) and \
				   (self.y_min_ <= temp_y <= self.y_max_) and \
					 self.dst_region_collision_free(index, [temp_x, temp_y]) and \
				   not self.test_new_region_blocking([temp_x, temp_y], obs):
					relocate_tunnel = self.get_tunnel(self.robot_, [temp_x, temp_y])
					collision_object = self.collision_tunnel_object(relocate_tunnel)
					collision_object = [x for x in collision_object if x != index]
					if not collision_object and not self.collision_tunnel_static(relocate_tunnel):
						res.append([temp_x, temp_y])
						if len(res) == 4:
							random.shuffle(res)
							return res
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
	
	def random_object_selection(self):
		res = []
		for i in range(len(self.curr_config_)):
			temp_x, temp_y, _, _ = self.curr_config_[i]
			grasp_tunnel = self.get_tunnel(self.robot_, [temp_x, temp_y])
			collision_object = self.collision_tunnel_object(grasp_tunnel)
			collision_object = [x for x in collision_object if x != i]
			if not collision_object:
				res.append(i)
		random.shuffle(res)
		return res


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
					return False
		return True


def write_result(method, test_index, object_count, time, steps, length):
	file_name = 'test_results/' + method + '/test_result_' + str(test_index) + '.txt'
	if os.path.exists(file_name):
		os.remove(file_name)
	with open(file_name, 'w') as f:
		f.write('number of objects : ' + str(object_count) + '\n')
		f.write('time comsumption : ' + str(time) + '\n')
		f.write('number of steps : ' + str(steps) + '\n')
		f.write('total length travelled : ' + str(length) + '\n')
	f.close()



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



