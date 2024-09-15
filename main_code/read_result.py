import os
from copy import deepcopy
import numpy as np
import pdb

def get_data(dir, data_dict, check_fail=False):
    file_list = []
    for x in os.listdir(dir):
        if x.endswith(".txt"):
            file_list.append(x)
    
    file_list.sort()
    try: f = open(dir+file_list[-1], "r")
    except: pdb.set_trace()
    data_lines = f.read().split("\n")
    
    is_failed = False
    for line in data_lines[:-1]:
        split_idx = line.find(":")
        name = line[:split_idx-1]

        try:
            number = float(line[split_idx+1:])
        except:
            if check_fail:
                is_failed = True
            number = line[split_idx+2:]
        
        data_dict[name].append(number)
    
    return data_dict, is_failed

def get_data_w_keyword(dir, data_dict, key_ward):
    file_list = []
    for x in os.listdir(dir):
        if x.endswith(".txt") and key_ward in x:
            file_list.append(x)
    
    file_list.sort()
    if not file_list:
        for i in data_dict.keys():
            data_dict[i].append('Failed')

        return data_dict, 'Failed'

    f = open(dir+file_list[-1], "r")
    data_lines = f.read().split("\n")
            
    for line in data_lines[:-1]:
        split_idx = line.find(":")
        name = line[:split_idx-1]

        try:
            number = float(line[split_idx+1:])
        except:
            number = line[split_idx+2:]
        
        try:data_dict[name].append(number)
        except:
            pdb.set_trace()
    
    return data_dict, data_dict['number of view'][-1]

def get_data_wo_keyword(dir, data_dict, key_ward):
    file_list = []
    for x in os.listdir(dir):
        if x.endswith(".txt") and key_ward[0] not in x and key_ward[1] not in x:
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
        
        try:
            data_dict[name].append(number)
        except:
            pdb.set_trace()
    
    return data_dict, data_dict['number of view'][-1]

def get_data_from_folders(root):
    folder_list = [f.path for f in os.scandir(root) if f.is_dir()]

    # empty_dict = {'number of objects': [],
    #               'time consumption': [],
    #               'number of steps': [],
    #               'total length travelled': [],
    #               'total length displacement': [],
    #               'number of view': [],
    #               'number of initial collision objects': []}
        
    empty_dict = {'number of objects' : [],
                  'rearrangement time consumption' : [],
                  'number of steps' : [],
                  'rearrangement length travelled' : [],
                  'rearrangement length displacement' : [],
                  'number of view' : [],
                  'number of initial collision objects' : [],
                  'number of rearrangement attempts' : [],
                  'view time consumption' : [],
                  'calculation for cam loc time consumption' : [],
                  'total time consumption' : []}

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
        # print("collecting data from ", folder)
        complete_sensing_MCTS = folder + '/complete_sensing/MCTS*/test_results/'
        complete_sensing_MCTS_OG = folder + '/complete_sensing/MCTS_OG/test_results/'
        complete_sensing_BASE1 = folder + '/complete_sensing/BASE1/test_results/'
        complete_sensing_BASE2 = folder + '/complete_sensing/BASE2/test_results/'

        dense_sensing = folder + '/dense_sensing/test_results/'
        init_sensing = folder + '/init_sensing/test_results/'
        init_w_feed_back = folder + '/init_w_feed_back/test_results/'
        init_w_swept = folder + '/init_w_swept/test_results/'

        complete_sensing_MCTS_dict, MCTS_failed = get_data(complete_sensing_MCTS, complete_sensing_MCTS_dict, check_fail=True)
        complete_sensing_MCTS_OG_dict, MCTS_OG_failed = get_data(complete_sensing_MCTS_OG, complete_sensing_MCTS_OG_dict, check_fail=True)
        complete_sensing_BASE1_dict, BASE1_failed = get_data(complete_sensing_BASE1, complete_sensing_BASE1_dict, check_fail=True)
        complete_sensing_BASE2_dict, BASE2_failed = get_data(complete_sensing_BASE2, complete_sensing_BASE2_dict, check_fail=True)

        if MCTS_failed and MCTS_OG_failed and BASE1_failed and BASE2_failed:
            print("ALL METHOD FAILED: ", folder)

        if MCTS_failed:
            print("MCTS Failed: ", folder)

        dense_sensing_dict, _ = get_data(dense_sensing, dense_sensing_dict)
        init_sensing_dict, _ = get_data(init_sensing, init_sensing_dict)
        init_w_feed_back_dict, _ = get_data(init_w_feed_back, init_w_feed_back_dict)
        init_w_swept_dict, _ = get_data(init_w_swept, init_w_swept_dict)

    data_dicts = [complete_sensing_MCTS_dict, complete_sensing_MCTS_OG_dict, complete_sensing_BASE1_dict, complete_sensing_BASE2_dict, dense_sensing_dict, init_sensing_dict, init_w_feed_back_dict, init_w_swept_dict]
    names = ["complete_sensing_MCTS", "complete_sensing_MCTS_OG","complete_sensing_BASE1", "complete_sensing_BASE2", "dense_sensing", "init_sensing", "init_w_feed_back", "init_w_swept"]

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
                    success_rate = np.round(len(new_val) / len(val) * 100, 2)

                if len(new_val) == 0:
                    f.write(key + " : ALL FAILED\n")
                else:
                    average = np.round(np.mean(new_val), 2)
                    std = np.round(np.std(new_val), 2)
                    f.write(key + " : " + str(average) + " ± " + str(std) + '\n')
            
            f.write("success rate : " + str(success_rate))

def write_success_only(data_dicts, names, root, test_num):
    success_list = [True] * test_num
    for idx, data_dict in enumerate(data_dicts):
        for val in data_dict.values():
            for idx, el in enumerate(val):
                if isinstance(el, str):
                    success_list[idx] = False

    for idx, data_dict in enumerate(data_dicts):
        file_name = names[idx] + "_success_only"

        with open(root + file_name, 'w') as f:
            f.write("Result of " + str(sum(success_list)) + " test cases\n\n")

            success_rate = 100
            for key, val in data_dict.items():
                new_val = [i for i in val if not isinstance(i, str)]        
                if len(new_val) != len(val):
                    success_rate = np.round(len(new_val) / len(val) * 100, 2)

                new_val = [el for idx, el in enumerate(val) if success_list[idx]]

                if len(new_val) == 0:
                    f.write(key + " : ALL FAILED\n")
                else:
                    average = np.round(np.mean(new_val), 2)
                    std = np.round(np.std(new_val), 2)
                    f.write(key + " : " + str(average) + " ± " + str(std) + '\n')
            

            f.write("success rate : " + str(success_rate))

def write_difference(data_dicts1, data_dicts2, root, num_win_1, num_win_2, name1, name2):
    diff_keys = ['rearrangement time consumption',
                 'number of steps',
                 'rearrangement length travelled',
                 'rearrangement length displacement',
                 'number of view']

    with open(root + "difference_" + name2 + "_" + name1, 'w') as f:
        f.write("Result of " + str(len(data_dicts1['number of view'])) + " test cases\n")
        f.write("difference = " + name2 + " - " + name1 + "\n\n")


        for key in diff_keys:
            diff_list = []
            for idx in range(len(data_dicts1['number of view'])):
                data1 = data_dicts1[key][idx]
                data2 = data_dicts2[key][idx]
                if isinstance(data1, str) or isinstance(data2, str):
                    continue
                diff_list.append(data2 - data1)

            diff = np.round(np.average(diff_list), 2)
            std = np.round(np.std(diff_list), 2)
            f.write(key + " difference : " + str(diff) + " ± " + str(std) + '\n')
        
        f.write(name2 + " outperform in number of view : " + str(num_win_2) + " times" + '\n')
        f.write(name1 + " outperform in number of view : " + str(num_win_1) + " times" + '\n')


def fix_data_(root):
    folder_list = [f.path for f in os.scandir(root) if f.is_dir()]

    folder_list.sort()
    for folder in folder_list:
        complete_sensing_MCTS = folder + '/complete_sensing/MCTS*/test_results/'
        complete_sensing_MCTS_OG = folder + '/complete_sensing/MCTS_OG/test_results/'
        complete_sensing_BASE1 = folder + '/complete_sensing/BASE1/test_results/'
        complete_sensing_BASE2 = folder + '/complete_sensing/BASE2/test_results/'

        dense_sensing = folder + '/dense_sensing/test_results/'
        init_sensing = folder + '/init_sensing/test_results/'
        init_w_feed_back = folder + '/init_w_feed_back/test_results/'
        init_w_swept = folder + '/init_w_swept/test_results/'

        fix_typeo(complete_sensing_MCTS)
        fix_typeo(complete_sensing_MCTS_OG)
        fix_typeo(complete_sensing_BASE1)
        fix_typeo(complete_sensing_BASE2)

        fix_typeo(dense_sensing)
        fix_typeo(init_sensing)
        fix_typeo(init_w_feed_back)
        fix_typeo(init_w_swept)

def fix_typeo(dir):
    file_list = []
    for x in os.listdir(dir):
        if x.endswith(".txt"):
            file_list.append(x)
    
    file_list.sort()
    for file in file_list:
        f = open(dir+file, "r")
        data_lines = f.read().split("\n")
                
        new_data = []
        for line in data_lines[:-1]:
            split_idx = line.find(":")
            name = line[:split_idx-1]

            if 'comsumption' in name:
                cut_idx = name.find('comsumption')
                name = name[:cut_idx] + 'consumption'
                line = name + line[split_idx-1:]

            if 'comsumptio' in name:
                cut_idx = name.find('comsumptio')
                name = name[:cut_idx] + 'consumption'
                line = name + ' ' + line[split_idx:]

            if 'rearrangment' in name:
                line = 'rearrangement ' + ' '.join(line.split(' ')[1:])

            new_data.append(line)

        with open(dir+file, 'w') as f:
            for line in new_data:
                f.write(line + '\n')

def find_scene_size(root, scene_range):
    folder_list = [f.path for f in os.scandir(root) if f.is_dir()]

    name_list = []
    for folder in folder_list:
        dir = folder + "/init_sensing/test_results/"
        for x in os.listdir(dir):
            if x.endswith("failed.npy") or x.endswith("success.npy"):
                data = np.load(dir + x, allow_pickle=True)[0]
                try:
                    scene_info = data['scene_info']
                except:
                    pdb.set_trace()

                if scene_info[0] >= scene_range[0][0] and scene_info[0] <= scene_range[0][1] and scene_info[1] >= scene_range[1][0] and scene_info[1] <= scene_range[1][1]:
                    name_list.append(folder)
                    break

    return name_list

if __name__ == '__main__':
    root = 'test_data/last_test/'
    root = 'test_data/collected_data/'
    # fix_data_(root)
    # scene_size = [[0.76, 0.8], [1.1, 1.2]]
    # folder_list = find_scene_size(root, scene_size)
    # pdb.set_trace()

    data_dicts, names, test_num = get_data_from_folders(root)
    write_result(data_dicts, names, root, test_num)
    write_success_only(data_dicts[:4], names[:4], root, test_num)



    # folder_list = [f.path for f in os.scandir(root) if f.is_dir()]
        
    # empty_dict = {'number of objects' : [],
    #               'rearrangement time consumption' : [],
    #               'number of steps' : [],
    #               'rearrangement length travelled' : [],
    #               'rearrangement length displacement' : [],
    #               'number of view' : [],
    #               'number of initial collision objects' : [],
    #               'number of rearrangement attempts' : [],
    #               'view time consumption' : [],
    #               'calculation for cam loc time consumption' : [],
    #               'total time consumption' : []}

    # MCTS_dict = deepcopy(empty_dict)
    # loop_dict = deepcopy(empty_dict)
    # obj_dict = deepcopy(empty_dict)

    # num_win_3 = 0
    # num_win_2 = 0
    # num_win_1_2 = 0
    # num_win_1_3 = 0

    # num_win_2_3 = 0
    # num_win_3_2 = 0
    
    # folder_list.sort()
    # for folder in folder_list:
    #     if folder == 'test_data/collected_data/1' or folder == 'test_data/collected_data/2':
    #         continue

    #     complete_sensing_MCTS = folder + '/test_results/complete_sensing/MCTS*/'

    #     MCTS_dict, num_view1 = get_data_wo_keyword(complete_sensing_MCTS, MCTS_dict, ["w_loop", "w_num_obj"])
    #     loop_dict, num_view2 = get_data_w_keyword(complete_sensing_MCTS, loop_dict, "w_loop")
    #     obj_dict, num_view3 = get_data_w_keyword(complete_sensing_MCTS, obj_dict, "w_num_obj")

    #     if not isinstance(num_view2, str) and num_view1 > num_view2:
    #         num_win_2 += 1

    #     if not isinstance(num_view3, str) and num_view1 > num_view3:
    #         num_win_3 += 1

    #     if (not isinstance(num_view2, str) and not isinstance(num_view3, str) and num_view2 > num_view3) or (not isinstance(num_view2, str) and isinstance(num_view3, str)):
    #         num_win_2_3 += 1

    #     if (not isinstance(num_view2, str) and not isinstance(num_view3, str) and num_view3 > num_view2) or (not isinstance(num_view3, str) and isinstance(num_view2, str)):
    #         num_win_2_3 += 1

    #     if isinstance(num_view2, str):
    #         num_win_1_2 += 1

    #     if isinstance(num_view3, str):
    #         num_win_1_2 += 1

    # write_difference(MCTS_dict, loop_dict, root, num_win_1_2, num_win_2, "WO_loop", "W_loop")
    # write_difference(MCTS_dict, obj_dict, root, num_win_1_2, num_win_2, "WO_loop", "obj_loop")
    # write_difference(loop_dict, obj_dict, root, num_win_1_2, num_win_2, "W_loop", "obj_loop")
    # write_result([loop_dict], ['complete_sensing_MCTS_W_loop'], root, len(loop_dict['number of view']))
    # write_result([obj_dict], ['complete_sensing_MCTS_W_num_obj'], root, len(loop_dict['number of view']))

    # write_success_only([MCTS_dict, loop_dict, obj_dict], ['MCTS', 'MCTS_W_loop', 'MCTS_W_num_obj'], root, test_num)
    # pdb.set_trace()

