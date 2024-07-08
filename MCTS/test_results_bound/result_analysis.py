import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.getcwd()
sys.path.append(current_dir + '/../')

from rearrangement_planning_util_ICRA import Tree_Node
from rearrangement_planning_util_ICRA import find_tunnel_side_swipe


def read_result(file_name):
	with open(file_name, 'r') as f:
		data = f.readlines()
		number_of_objects = 0
		time_consumption = 0
		number_of_steps = 0
		total_length_travelled = 0
		total_length_displacement = 0
		for t in range(5):
			str_value = data[t][data[t].index(':')+2 : -1]
			if t == 0:
				number_of_objects = int(str_value)
			elif t == 1:
				time_consumption = float(str_value)
			elif t == 2:
				number_of_steps = int(float(str_value))
			elif t == 3:
				total_length_travelled = float(str_value)
			elif t == 4:
				total_length_displacement = float(str_value)
	f.close()
	return number_of_objects, time_consumption, number_of_steps, total_length_travelled, total_length_displacement


def cast_config(config):
	res = []
	for element in config:
		x, y, r, color = element
		res.append([float(x), float(y), float(r), color])
	return res


def data_analysis(exp_start, exp_end, section):
	total_test_cases = 400

	#methods = ['BIRRT_MRS', 'PERTS_new', 'MS_MCTS_IROS_Gauss']
	methods = None
	if section != 'AB':
		#methods = ['BIRRT_MRS', 'PERTS', 'MS_MCTS_CONT_2']
		methods = ['MS_MCTS_CONT_FINAL', 'PERTS', 'BIRRT_MRS']
	else:
		methods = ['MS_MCTS_CONT_FINAL', 'MS_MCTS_CONT_AB_NONE_FINAL', 'MS_MCTS_CONT_AB_NOPT_FINAL', 'MS_MCTS_DIS_FINAL_2']
	#methods = ['MS_MCTS_ICRA']

	#methods = ['MS_MCTS_CONT_FINAL', 'MS_MCTS_CONT_AB_NONE_FINAL', 'MS_MCTS_DIS_FINAL', 'MS_MCTS_CONT_AB_NOPT_FINAL']

	data_tracker = {}

	object_set = set()

	dummy_grid = []
	for i in range(20):
		temp_arr = []
		for j in range(20):
			temp_arr.append(0)
		dummy_grid.append(temp_arr)

	for method in methods:
		method_data_tracker = {}
		ma_steps = []
		ma_time = []
		ma_travel = []
		ma_gripper = []
		ma_displace = []
		total_success = 0

		#for smart LMP travel length calculation
		for t in range(exp_start, exp_end):
			file_name = method + '/test_result_' + str(t) + '.npy'
			if os.path.exists(file_name):
				f = open(file_name, 'rb')
				plan = np.load(f)
				plan = plan.tolist()

				total_travel = 0.0

				gripper_loc = [10, -2]

				total_gripper = 0.0

				for k in range(len(plan)-1):
					source_config = plan[k]
					target_config = plan[k+1]

					source_config = cast_config(source_config)
					target_config = cast_config(target_config)

					move_index = None
					for q in range(len(source_config)):
						if source_config[q] != target_config[q]:
							move_index = q
							break

					if move_index != None:
						dummy_source = Tree_Node(source_config, target_config, dummy_grid)
						dummy_target = Tree_Node(target_config, target_config, dummy_grid)

						rx, ry = dummy_source.robot_
						robot_width = dummy_source.robot_width_

						grasp_tunnel = dummy_source.get_tunnel(dummy_source.robot_, source_config[move_index])
						relocate_tunnel = dummy_target.get_tunnel(dummy_target.robot_, target_config[move_index])
						
						grasp_angle_degree = grasp_tunnel[3]
						relocate_angle_degree = relocate_tunnel[3]
						grasp_depth = grasp_tunnel[1]
						relocate_depth = relocate_tunnel[1]

						smaller_angle = min(grasp_angle_degree, relocate_angle_degree)
						bigger_angle = max(grasp_angle_degree, relocate_angle_degree)

						sweep_depth = grasp_depth
						if relocate_depth < grasp_depth: sweep_depth = relocate_depth

						for ind in range(len(source_config)):
							if ind != move_index:
								cx, cy, radius, _ = source_config[ind]
								left_theta, right_theta, max_length = find_tunnel_side_swipe(cx, cy, radius, rx, ry, robot_width)
								left_theta_degree = left_theta / math.pi * 180
								right_theta_degree = right_theta / math.pi * 180

								if right_theta_degree <= smaller_angle + 3 or left_theta_degree >= bigger_angle - 3:
									pass
								else:
									sweep_depth = min(sweep_depth, max_length)

						total_travel += (grasp_depth - sweep_depth)
						total_travel += (relocate_depth - sweep_depth)
						total_travel += (abs(grasp_angle_degree - relocate_angle_degree) / 180 * math.pi * sweep_depth)

						total_gripper += (grasp_depth - sweep_depth)
						total_gripper += (relocate_depth - sweep_depth)
						total_gripper += (abs(grasp_angle_degree - relocate_angle_degree) / 180 * math.pi * sweep_depth)


					if move_index != None:
						new_source_config = source_config
						new_target_config = source_config

						new_source_config[move_index][0] = gripper_loc[0]
						new_source_config[move_index][1] = gripper_loc[1]

						dummy_source = Tree_Node(new_source_config, new_target_config, dummy_grid)
						dummy_target = Tree_Node(new_target_config, new_target_config, dummy_grid)

						rx, ry = dummy_source.robot_
						robot_width = dummy_source.robot_width_

						grasp_tunnel = dummy_source.get_tunnel(dummy_source.robot_, source_config[move_index])
						relocate_tunnel = dummy_target.get_tunnel(dummy_target.robot_, target_config[move_index])
						
						grasp_angle_degree = grasp_tunnel[3]
						relocate_angle_degree = relocate_tunnel[3]
						grasp_depth = grasp_tunnel[1]
						relocate_depth = relocate_tunnel[1]

						smaller_angle = min(grasp_angle_degree, relocate_angle_degree)
						bigger_angle = max(grasp_angle_degree, relocate_angle_degree)

						sweep_depth = grasp_depth
						if relocate_depth < grasp_depth: sweep_depth = relocate_depth

						for ind in range(len(source_config)):
							if ind != move_index:
								cx, cy, radius, _ = source_config[ind]
								left_theta, right_theta, max_length = find_tunnel_side_swipe(cx, cy, radius, rx, ry, robot_width)
								left_theta_degree = left_theta / math.pi * 180
								right_theta_degree = right_theta / math.pi * 180

								if right_theta_degree <= smaller_angle + 3 or left_theta_degree >= bigger_angle - 3:
									pass
								else:
									sweep_depth = min(sweep_depth, max_length)

						total_gripper += (grasp_depth - sweep_depth)
						total_gripper += (relocate_depth - sweep_depth)
						total_gripper += (abs(grasp_angle_degree - relocate_angle_degree) / 180 * math.pi * sweep_depth)

						gripper_loc = target_config[move_index][:2]
				
				ma_travel.append(total_travel)
				ma_gripper.append(total_gripper)
			
		for t in range(exp_start, exp_end):
			file_name = method + '/test_result_' + str(t) + '.txt'
			if os.path.exists(file_name):
				number_of_objects, time_consumption, number_of_steps, total_length_travelled, total_length_displacement = read_result(file_name)
				if method != 'BIRRT_MRS':
					number_of_steps -= 1
				if time_consumption > 60:
					continue
				object_set.add(number_of_objects)
				temp_counter = 0
				if number_of_objects not in method_data_tracker:
					method_data_tracker[number_of_objects] = [time_consumption, number_of_steps, total_length_travelled, total_length_displacement, 1]
				else:
					method_data_tracker[number_of_objects][0] += time_consumption
					method_data_tracker[number_of_objects][1] += number_of_steps
					method_data_tracker[number_of_objects][2] += ma_travel[temp_counter]
					method_data_tracker[number_of_objects][3] += total_length_displacement
					method_data_tracker[number_of_objects][4] += 1
					temp_counter += 1

				ma_steps.append(number_of_steps)
				ma_time.append(time_consumption)
				#ma_travel.append(total_length_travelled)
				ma_displace.append(total_length_displacement)

				total_success += 1
			else:
				#print(t)
				pass

		#sys.exit(1)

		print('*******************************************************************')
		print('stat start ' + method)
		print('Success rate {0}'.format(total_success * 1.0/(exp_end - exp_start)))
		print('time {0} {1}'.format(round(np.mean(np.array(ma_time)),2), round(np.std(np.array(ma_time)),2)))
		print('steps {0} {1}'.format(round(np.mean(np.array(ma_steps)),2), round(np.std(np.array(ma_steps)),2)))
		print('travel {0} {1}'.format(round(np.mean(np.array(ma_travel)),2), round(np.std(np.array(ma_travel)),2)))
		#print('displace {0} {1}'.format(round(np.mean(np.array(ma_displace)),2), round(np.std(np.array(ma_displace)),2)))
		print('gripper {0} {1}'.format(round(np.mean(np.array(ma_gripper)),2), round(np.std(np.array(ma_gripper)),2)))
		print('stat end ' + method)
		print('*******************************************************************')

		data_tracker[method] = method_data_tracker

	success_rate_list = []
	time_consumption_list = []
	number_of_steps_list = []
	total_length_travelled_list = []
	total_length_displacement_list = []

	object_set = sorted(list(object_set))

	sys.exit(1)

	for method in methods:
		method_success_rate_list = []
		method_time_consumption_list = []
		method_number_of_steps_list = []
		method_total_length_travelled_list = []
		method_total_length_displacement_list = []
		print(data_tracker[method])
		for number_of_object in object_set:
			if number_of_object in data_tracker[method]:
				time_consumption,number_of_steps, total_length_travelled, total_length_displacement, success_instance = data_tracker[method][number_of_object]
			else:
				time_comsuption, number_of_steps, total_length_travelled, total_length_displacement, success_instance = 0, 0, 0, 0, 0
			method_success_rate_list.append(success_instance / 80.0)
			if success_instance != 0:
				method_time_consumption_list.append(time_consumption / success_instance)
				method_number_of_steps_list.append(number_of_steps / success_instance)
				method_total_length_travelled_list.append(total_length_travelled / success_instance)
				method_total_length_displacement_list.append(total_length_displacement / success_instance)
			else:
				method_time_consumption_list.append(0)
				method_number_of_steps_list.append(0)
				method_total_length_travelled_list.append(0)
				method_total_length_displacement_list.append(0)

		success_rate_list.append(method_success_rate_list)
		time_consumption_list.append(method_time_consumption_list)
		number_of_steps_list.append(method_number_of_steps_list)
		total_length_travelled_list.append(method_total_length_travelled_list)
		total_length_displacement_list.append(method_total_length_displacement_list)


	plot_two_in_one(object_set, [success_rate_list, number_of_steps_list], methods, ['success rate', 'average steps'])

	print('here')
	sys.exit(1)

	print(success_rate_list)
	#print(time_consumption_list)
	print(number_of_steps_list)
	#print(total_length_travelled_list)
	print(total_length_displacement_list)

	#sys.exit(1)
	plot_bar_chart(object_set, success_rate_list, methods, 'success rate')
	plot_bar_chart(object_set, time_consumption_list, methods, 'average time consumption')
	plot_bar_chart(object_set, number_of_steps_list, methods, 'average number of steps')
	plot_bar_chart(object_set, total_length_travelled_list, methods, 'average travelled distance')
	plot_bar_chart(object_set, total_length_displacement_list, methods, 'average displacement distance')

			

def plot_bar_chart(x_data, y_data, legend_list, y_label):
	
	barwidth = 0.25
	fig = plt.subplots(figsize = (12, 8))

	br = np.arange(len(x_data))

	colors = ['#f4d7de', '#a4d4dc', '#cccaf0']

	legend_list = ['MC-MCTS-OP', 'PERTS(CIRS)', 'BiRRT(mRS)']

	for i in range(len(y_data)):
		temp_br = [x + i * barwidth for x in br]
		plt.bar(temp_br, y_data[i], color = colors[i], width = barwidth, edgecolor = 'grey', label = legend_list[i])

	plt.xlabel('number of objects', fontweight = 'bold', fontsize = 20)
	plt.ylabel(y_label, fontweight = 'bold', fontsize = 20)
	#plt.title(y_label + ' vs number of objects', fontweight = 'bold', fontsize = 25)
	plt.xticks([x + barwidth for x in br], ['4', '5', '6', '7', '8'], fontsize = 20)
	plt.yticks(fontsize = 20)

	if y_label == 'success rate':
		plt.ylim(0, 1.19)

	plt.grid()
	plt.legend(fontsize = 20)
	plt.show()


def plot_two_in_one(x_data, y_data_list, legend_list, y_label_list):
	barwidth = 0.25

	fig, axs = plt.subplots(2, figsize = (10, 20))

	br = np.arange(len(x_data))

	colors = ['#f4d7de', '#a4d4dc', '#cccaf0']

	legend_list = ['MS-MCTS', 'PERTS(CIRS)', 'BiRRT(mRS)']

	for t in range(2):
		y_data = y_data_list[t]
		y_label = y_label_list[t]
		for i in range(len(y_data)):
			temp_br = [x + i * barwidth for x in br]
			axs[t].bar(temp_br, y_data[i], color = colors[i], width = barwidth, edgecolor = 'grey', label = legend_list[i])

		if t == 1:
			axs[t].set_xlabel('number of objects', fontsize = 15)
		axs[t].set_ylabel(y_label, fontsize = 15)
		#plt.title(y_label + ' vs number of objects', fontweight = 'bold', fontsize = 25)
		#axs[t].yticks(fontsize = 20)

		axs[t].grid()
		#axs[t].show()

	plt.setp(axs, xticks=[x + barwidth for x in br], xticklabels = ['4', '5', '6', '7', '8'])
	plt.legend(bbox_to_anchor = (0.5, -0.32), loc = 'lower center', ncol = 3, fontsize = 15)
	plt.show()
		





if __name__ == '__main__':
	exp_start = int(sys.argv[1])
	exp_end = int(sys.argv[2])
	section = sys.argv[3]
	data_analysis(exp_start, exp_end, section)


