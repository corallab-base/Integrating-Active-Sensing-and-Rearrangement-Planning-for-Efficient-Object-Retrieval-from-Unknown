import os
from copy import deepcopy
import numpy as np
import pdb

def get_data(dir, data_dict):
    file_list = []
    for x in os.listdir(dir):
        if x.endswith(".txt"):
            file_list.append(x)
    
    file_list.sort()
    f = open(dir+file_list[-1], "r")
    data_lines = f.read().split("\n")
            
    for line in data_lines[:-1]:
        split_idx = line.find(":")
        name = line[:split_idx-1]

        try:
            number = float(line[split_idx+1:])
        except:
            number = line[split_idx+2:]
        
        data_dict[name].append(number)
    
    return data_dict

def get_data_from_folders(root):
    folder_list = [f.path for f in os.scandir(root) if f.is_dir()]

    # empty_dict = {'number of objects': [],
    #               'time comsumption': [],
    #               'number of steps': [],
    #               'total length travelled': [],
    #               'total length displacement': [],
    #               'number of view': [],
    #               'number of initial collision objects': []}
        
    empty_dict = {'number of objects' : [],
                  'rearrangment time comsumption' : [],
                  'number of steps' : [],
                  'rearrangment length travelled' : [],
                  'rearrangment length displacement' : [],
                  'number of view' : [],
                  'number of initial collision objects' : [],
                  'number of rearrangement attempts' : [],
                  'view time comsumption' : [],
                  'calculation for cam loc time comsumption' : [],
                  'total time comsumptio' : []}

    complete_sensing_MCTS_dict = deepcopy(empty_dict)
    complete_sensing_MCTS_OG_dict = deepcopy(empty_dict)
    complete_sensing_BASE1_dict = deepcopy(empty_dict)
    complete_sensing_BASE2_dict = deepcopy(empty_dict)

    dense_sensing_dict = deepcopy(empty_dict)
    init_sensing_dict = deepcopy(empty_dict)
    init_w_feed_back_dict = deepcopy(empty_dict)
    init_w_swept_dict = deepcopy(empty_dict)

    folder_list.sort()
    for folder in folder_list:
        print("collecting data from ", folder)
        complete_sensing_MCTS = folder + '/test_results/complete_sensing/MCTS*/'
        complete_sensing_MCTS_OG = folder + '/test_results/complete_sensing/MCTS_OG/'
        complete_sensing_BASE1 = folder + '/test_results/complete_sensing/BASE1/'
        complete_sensing_BASE2 = folder + '/test_results/complete_sensing/BASE2/'

        dense_sensing = folder + '/test_results/dense_sensing/'
        init_sensing = folder + '/test_results/init_sensing/'
        init_w_feed_back = folder + '/test_results/init_w_feed_back/'
        init_w_swept = folder + '/test_results/init_w_swept/'

        complete_sensing_MCTS_dict = get_data(complete_sensing_MCTS, complete_sensing_MCTS_dict)
        complete_sensing_MCTS_OG_dict = get_data(complete_sensing_MCTS_OG, complete_sensing_MCTS_OG_dict)
        complete_sensing_BASE1_dict = get_data(complete_sensing_BASE1, complete_sensing_BASE1_dict)
        complete_sensing_BASE2_dict = get_data(complete_sensing_BASE2, complete_sensing_BASE2_dict)

        dense_sensing_dict = get_data(dense_sensing, dense_sensing_dict)
        init_sensing_dict = get_data(init_sensing, init_sensing_dict)
        init_w_feed_back_dict = get_data(init_w_feed_back, init_w_feed_back_dict)
        init_w_swept_dict = get_data(init_w_swept, init_w_swept_dict)

    data_dicts = [complete_sensing_MCTS_dict, complete_sensing_MCTS_OG_dict, complete_sensing_BASE1_dict, complete_sensing_BASE2_dict, dense_sensing_dict, init_sensing_dict, init_w_feed_back_dict, init_w_swept_dict]
    names = ["complete_sensing_MCTS", "complete_sensing_MCTS_OG", "complete_sensing_BASE1", "complete_sensing_BASE2", "dense_sensing", "init_sensing", "init_w_feed_back", "init_w_swept"]

    return data_dicts, names, len(folder_list)


def write_result(data_dicts, names, root, test_num):
    for idx, data_dict in enumerate(data_dicts):
        file_name = names[idx]

        with open(root + file_name, 'w') as f:
            f.write("Result of " + str(test_num) + " test cases\n\n")

            success_rate = 100
            for key, val in data_dict.items():
                new_val = [i for i in val if not isinstance(i, str)]        
                if len(new_val) != len(val):
                    success_rate = len(new_val) / len(val) * 100

                print(new_val)
                if len(new_val) == 0:
                    f.write(key + " : ALL FAILED")
                else:
                    average = np.mean(new_val)
                    std = np.std(new_val)
                    f.write(key + " : " + str(average) + " ± " + str(std) + '\n')
            

            f.write("success rate : " + str(success_rate))


if __name__ == '__main__':
    root = 'test_data/collected_data/'
    data_dicts, names, test_num = get_data_from_folders(root)
    write_result(data_dicts, names, root, test_num)

