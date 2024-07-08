#
#
#
#
# maximum objects 25
#


import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt


class RP_test_case_generator():
	def __init__(self, min_width, max_width, min_length, max_length, min_object, max_object):
		self.min_width_ = min_width
		self.max_width_ = max_width
		self.min_length_ = min_length
		self.max_length_ = max_length
		self.min_object_ = min_object
		self.max_object_ = max_object
		self.curr_config_ = []
		self.goal_config_ = []

		self.colors_ = ['dimgray', 'lightcoral', 'firebrick', 'red', 'sienna', 'saddlebrown', 'darkorange', 'tan', 'gold', 'yellow', 'beige', 'yellowgreen', 'lawngreen', 'palegreen', 'turquoise', 'teal', 'cyan', 'dodgerblue', 'blue', 'navy', 'violet', 'fuchsia', 'crimson', 'pink', 'orchid']



	def region_valid(self, cx, cy, flag):
		#flag = 0 checks start region
		#flag = 1 checks goal region
		valid= True
		if flag == 0:
			for tx, ty, _, _ in self.curr_config_:
				distance = math.sqrt((cx - tx)**2 + (cy - ty)**2)
				if distance <= 5: 
					valid = False
					break
			return valid
		else:
			for tx, ty, _, _ in self.goal_config_:
				distance = math.sqrt((cx - tx)**2 + (cy - ty)**2)
				if distance <= 5: 
					valid = False
					break
			return valid
	

	def generate_case(self):
		new_width = np.random.randint(self.min_width_, self.max_width_+1)

		new_length = np.random.randint(self.min_length_, self.max_length_+1)

		new_object = np.random.randint(self.min_object_, self.max_object_ + 1)

		curr_config = []
		goal_config = []

		random.shuffle(self.colors_)

		for i in range(new_object):

			#solve the overlap case
			cx = np.random.randint(new_width)
			cy = np.random.randint(new_length)
			while not self.region_valid(cx, cy, 0):
				cx = np.random.randint(new_width)
				cy = np.random.randint(new_length)

			radius = 1
			rgb = self.colors_[i]

			gx = np.random.randint(new_width)
			gy = np.random.randint(new_length)
			while not self.region_valid(gx, gy, 1):
				gx = np.random.randint(new_width)
				gy = np.random.randint(new_length)

			self.curr_config_.append([cx, cy, radius, rgb])
			self.goal_config_.append([gx, gy, radius, rgb])

		return self.curr_config_, self.goal_config_

def test_case_reader(case_index):
	file_name = 'test_cases/test_' + str(case_index) + '.txt'
	curr_config, goal_config = [], []
	with open(file_name, 'r') as f:
		data = f.readlines()
		object_count = int((len(data)-2)/2)
		for t in range(1, 1 + object_count):
			line = data[t][:-1].split()
			curr_config.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
			
		for t in range(2 + object_count, 2 + object_count*2):
			line = data[t][:-1].split()
			goal_config.append([int(line[0]), int(line[1]), int(line[2]), line[3]])
	f.close()
	return curr_config, goal_config

	

if __name__ == '__main__':

	for k in range(8, 9):
		for t in [338]:
		#for t in range((k-5)*80, ((k-4)*80)):
		#for t in range(40, 50):
			file_name = 'test_cases/test_' + str(t) + '.txt'
			RP_test_case_generator_ins = RP_test_case_generator(19, 19, 19, 19, 8, 8)
			curr_config, goal_config = RP_test_case_generator_ins.generate_case()
			with open(file_name, 'w') as f:
				f.write('curr config\n')
				for cx, cy, radius, color in curr_config:
					arr = [str(cx), str(cy), str(radius), color]
					f.write(' '.join(arr) + '\n')
				f.write('goal config\n')
				for cx, cy, radius, color in goal_config:
					arr = [str(cx), str(cy), str(radius), color]
					f.write(' '.join(arr) + '\n')
			f.close()


	
