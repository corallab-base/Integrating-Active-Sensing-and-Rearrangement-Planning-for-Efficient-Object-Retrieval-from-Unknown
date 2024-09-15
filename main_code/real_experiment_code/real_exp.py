import os
import sys
import math
import time
import re
from copy import deepcopy
import numpy as np
import rtde_control
import rtde_receive
import robotiq_gripper
from rtde_control import RTDEControlInterface as RTDEControl
import pdb
# from rearrangement_planning_util_ICRA_backup import smart_LMP_motion_real
sys.path.append('/home/j0k/Project/Imsa/main_code')
import robot_arm_configuration as RC


def power_off_pose(rtde_c):
    print('Reset to initial upstraight pose')
    rtde_c.moveJ([0, -math.pi/2, 0, -math.pi/2, 0, 0])


def get_plan_npy(plan_file):
    new_plan = np.load(plan_file)

    plan = []
    for config in new_plan:
        new_config = []
        for i in config:
            # print(i)
            new_config.append([float(i[0])/100, float(i[1])/100, float(i[2])/100])
        plan.append(new_config)
        

    # plan = plan.tolist()
    print(plan)

    #sanity check
    #for element in plan:
    #    print(element)

    #sys.exit(1)
    
    move_plan = []

    distance = 0.0

    for t in range(len(plan)-1):
        source = plan[t]
        target = plan[t+1]
        if len(source) != len(target):
            continue
        else:
            counter = 0
            for k in range(len(source)):
                if source[k] != target[k]:
                    sx, sy = source[k][0], source[k][1]
                    tx, ty = target[k][0], target[k][1]

                    move_plan.append([sx, sy, tx, ty])
                    distance += math.sqrt((sx - sy)**2 + (tx - ty)**2)
                    counter += 1
            if counter != 1:
                print('something wrong\n')
                sys.exit(1)
    print(move_plan)
    print(distance)
    # sys.exit(1)

    return move_plan

def get_plan(plan_file):
    plan = []
    with open(plan_file, 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            new_str = ''
            for char in line:
                if char.isdigit() or char.isalpha() or char == ' ' or char == '.':
                    new_str += char
            sp = new_str.split()
            #print(sp)
            subplan = []
            for t in range(len(sp)//4):
                subplan.append(sp[t*4:(t+1)*4])
            plan.append(subplan)



    #sanity check
    #for element in plan:
    #    print(element)

    #sys.exit(1)
    
    move_plan = []

    distance = 0.0

    old_loc_x, old_loc_y = 20, 0

    for t in range(len(plan)-1):
        source = plan[t]
        target = plan[t+1]
        for i in range(len(source)): 
            source[i][0] = float(source[i][0])
            source[i][1] = float(source[i][1])
            source[i][2] = float(source[i][2])
        for i in range(len(target)): 
            target[i][0] = float(target[i][0])
            target[i][1] = float(target[i][1])
            target[i][2] = float(target[i][2])

        if len(source) != len(target):
            continue
        else:
            counter = 0
            for k in range(len(source)):
                if source[k] != target[k]:
                    dummy_source = deepcopy(source[:])
                    dummy_target = deepcopy(source[:])
                    dummy_source[k][0] = old_loc_x
                    dummy_source[k][1] = old_loc_y
                    print(dummy_source, dummy_target)
                    sx, sy = float(source[k][0]), float(source[k][1])
                    tx, ty = float(target[k][0]), float(target[k][1])
                    grasp_depth = smart_LMP_motion_real(dummy_source, dummy_target)
                    sweep_depth = smart_LMP_motion_real(source, target)
                    move_plan.append([sx, sy, tx, ty, grasp_depth,  sweep_depth])
                    distance += math.sqrt((sx*0.038 - sy*0.038)**2 + (tx*0.038 - ty*0.038)**2)
                    counter += 1
                    old_loc_x = target[k][0]
                    old_loc_y = target[k][1]
            if counter != 1:
                print('something wrong\n')
                sys.exit(1)
    print(move_plan)
    print(distance)
    return move_plan
                

def reset_pose(rtde_c, robot_move_map):
    #start_config_up_2 = [-0.7776, -1.5929, 2.7465, -1.1525, 0.8413, -3.1415]
    start_config_up_2 = robot_move_map[0]
    rtde_c.moveJ(start_config_up_2)

def apply_rotation(angle):
    rotation = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

    rotation2 = np.array([[math.cos(angle), -math.sin(angle), 0],
                                             [math.sin(angle), math.cos(angle), 0], 
                                             [0, 0, 1]])

    print(np.matmul(rotation, rotation2))


def linear_motion_planner_old(rtde_c, gripper, plan, move_map, place_map, drop_map, move = True):


    #this move indictates the planning starts
    gripper.activate()

    print('Start linear motion planner')
    
    #x = 0.2 up
    start_config_up_2 = move_map[0]
    start_config_down_2 = place_map[0]

    #x = 0.3 up
    start_config_up_3 = [-0.4604, -1.4206, 2.4393, -1.0178, 1.1104, -3.1412]

    #x = 0.5 up
    middle_config_up = [0, -1.0210, 1.8022, -0.7804, 1.5702, -3.1408]

  #x = 0.5 down
    middle_config_down = [0, -0.9312, 1.8214, -0.8894, 1.5701, -3.1408]

    #x = 0.7, up
    max_config_reach_7 = [-0.1906, -0.6579, 1.1165, -0.4578, 1.3792, -3.141]

    #x = 0.8, up
    max_config_reach_8 = [-0.1674, -0.3364, 0.4743, -0.1371, 1.4034, -3.1409]

    #place down
    #rtde_c.moveJ(start_config_up_2)

    linear_start_pos = [-0.3, 0, 0.148, 2.417, 2.426, -2.413]

    linear_max_pos = [-0.9, 0, 0.148, 2.417, 2.426, -2.413]

    rx = 0
    #rx = 0.038*10
    ry = -0.12


    start_config_up_2 = move_map[0]
    start_config_down_2 = place_map[0]

    def coord_cast(x, y, z, angle):
        #x = x + math.sin(angle)*0.25
        #y = y - math.cos(angle)*0.25
        print(x, y, z, angle)
        return x, y, z

    for element in plan:
        sx, sy, tx, ty = element
        print(element)
        sy += 0.07
        ty += 0.07
        # sx *= 0.038
        # sy *= 0.038
        # tx *= 0.038
        # ty *= 0.038

        angle1, angle2 = 0, 0

        #angle1 = -math.asin((sx - rx)/(sy - ry))
        angle1 = -math.atan2(sx - rx, sy - ry)
        #angle2 = -math.asin((tx - rx)/(ty - ry))
        angle2 = -math.atan2(tx - rx, ty - ry)

        grasp_x, grasp_y, grasp_z = coord_cast(sx, sy, 0.148, angle1)
        distance = math.sqrt((grasp_x - rx)**2 + (grasp_y - ry)**2) - 0.44
        grasp_index = round(distance/0.01)


        grasp_start_config = [angle1 - 0.7776] + start_config_down_2[1:]
        grasp_end_config = [angle1 - 0.7776] + start_config_up_2[1:]
        grasp_move_config = move_map[grasp_index][:]
        grasp_move_config[0] += angle1
        grasp_place_config = place_map[grasp_index][:]
        grasp_place_config[0] += angle1

        if move:
            rtde_c.moveJ(grasp_start_config)
            #move down grasp
            rtde_c.moveL_FK(grasp_place_config)
            #grasp
            gripper.move(200, 100, 0)
            time.sleep(1)
            #move up
            rtde_c.moveL_FK(grasp_move_config)
            #retrieve
            rtde_c.moveL_FK(grasp_end_config)
            #rtde_c.moveJ(start_config_up_2)

        #if element == [10, 12, 10, 11]:
        #    break

        
        relocate_x, relocate_y, relocate_z = coord_cast(tx, ty, 0.148, angle2)
        distance = math.sqrt((relocate_x - rx)**2 + (relocate_y - ry)**2) - 0.44
        relocate_index = round(distance/0.01)
        
        relocate_start_config = [angle2 - 0.7776] + start_config_up_2[1:]
        relocate_end_config = [angle2 - 0.7776] + start_config_down_2[1:]
        relocate_move_config = move_map[relocate_index][:]
        relocate_move_config[0] += angle2
        relocate_place_config = drop_map[relocate_index][:]
        relocate_place_config[0] += angle2


        if move:
            rtde_c.moveJ(relocate_start_config)
            #move grasp
            rtde_c.moveL_FK(relocate_move_config)
            #move down grasp
            rtde_c.moveL_FK(relocate_place_config)
            #place
            gripper.move(0, 100, 0)
            time.sleep(1)
            #move up
            #rtde_c.moveL_FK(relocate_move_config)
            #retrieve
            rtde_c.moveL_FK(relocate_end_config)
            #rtde_c.moveJ(start_config_down_2)

        
        print('move complete')

    

def linear_motion_planner(rtde_c, gripper, plan, move_map, place_map, drop_map, move = True):


    #this move indictates the planning starts
    gripper.activate()

    print('Start linear motion planner')
    
    #x = 0.2 up
    start_config_up_2 = [-0.7776, -1.5929, 2.7465, -1.1525, 0.8413, -3.1415]
    start_config_down_2 = [-0.7776, -1.4129, 2.7716, -1.3577, 0.8413, -3.1415]

    #x = 0.3 up
    start_config_up_3 = [-0.4604, -1.4206, 2.4393, -1.0178, 1.1104, -3.1412]

    #x = 0.5 up
    middle_config_up = [0, -1.0210, 1.8022, -0.7804, 1.5702, -3.1408]

  #x = 0.5 down
    middle_config_down = [0, -0.9312, 1.8214, -0.8894, 1.5701, -3.1408]

    #x = 0.7, up
    max_config_reach_7 = [-0.1906, -0.6579, 1.1165, -0.4578, 1.3792, -3.141]

    #x = 0.8, up
    max_config_reach_8 = [-0.1674, -0.3364, 0.4743, -0.1371, 1.4034, -3.1409]

    #place down
    #rtde_c.moveJ(start_config_up_2)

    linear_start_pos = [-0.3, 0, 0.148, 2.417, 2.426, -2.413]

    linear_max_pos = [-0.9, 0, 0.148, 2.417, 2.426, -2.413]

    rx = 0
    #rx = 0.038*10
    ry = -0.12

    start_config_up_2 = move_map[0]
    start_config_down_2 = place_map[0]

    def coord_cast(x, y, z, angle):
        #x = x + math.sin(angle)*0.25
        #y = y - math.cos(angle)*0.25
        #print(x, y, z, angle)
        return x, y, z

    old_loc_x, old_loc_y = 20, 0
    old_angle = 0


    for i in range(len(plan)):
        element = plan[i]
        sx, sy, tx, ty, grasp_depth, depth = element
        print(element)
        # sx *= 0.038
        # sy *= 0.038
        # tx *= 0.038
        # ty *= 0.038

        angle1, angle2 = 0, 0

        #angle1 = -math.asin((sx - rx)/(sy - ry))
        angle1 = -math.atan2(sx - rx, sy - ry)
        #angle2 = -math.asin((tx - rx)/(ty - ry))
        angle2 = -math.atan2(tx - rx, ty - ry)

        grasp_x, grasp_y, grasp_z = coord_cast(sx, sy, 0.148, angle1)
        distance = math.sqrt((grasp_x - rx)**2 + (grasp_y - ry)**2) - 0.44
        grasp_index = round(distance/0.01)


        grasp_start_config = [angle1 - 0.7776] + start_config_down_2[1:]
        grasp_end_config = [angle1 - 0.7776] + start_config_up_2[1:]
        grasp_move_config = move_map[grasp_index][:]
        grasp_move_config[0] += angle1
        grasp_place_config = place_map[grasp_index][:]
        grasp_place_config[0] += angle1

        retract_index = max(0, math.floor((grasp_depth*0.038 - 0.55)/0.01))
        #retract_index = min(retract_index, min(relocate_index, grasp_index))
        #if abs(relocate_index - retract_index <= 2): retract_index = relocate_index
        retract_move_config = place_map[retract_index][:]
        retract_move_config[0] += old_angle
        
        swipe_config = place_map[retract_index][:]
        swipe_config[0] += angle1


        if move:
            rtde_c.moveL_FK(retract_move_config)
            rtde_c.moveJ(swipe_config)
            #move down grasp
            rtde_c.moveL_FK(grasp_place_config)
            #grasp
            gripper.move(50, 100, 0)
            time.sleep(1)
            #move up
            rtde_c.moveL_FK(grasp_move_config)
            #retrieve
            #rtde_c.moveL_FK(grasp_end_config)
            #rtde_c.moveJ(start_config_up_2)

        #if element == [10, 12, 10, 11]:
        #    break

        
        relocate_x, relocate_y, relocate_z = coord_cast(tx, ty, 0.148, angle2)
        distance = math.sqrt((relocate_x - rx)**2 + (relocate_y - ry)**2) - 0.44
        relocate_index = round(distance/0.01)
        
        relocate_start_config = [angle2 - 0.7776] + start_config_up_2[1:]
        relocate_end_config = [angle2 - 0.7776] + start_config_down_2[1:]
        relocate_move_config = move_map[relocate_index][:]
        relocate_move_config[0] += angle2
        relocate_place_config = drop_map[relocate_index][:]
        relocate_place_config[0] += angle2

        end_config = [angle2 - 0.7776] + move_map[0][1:]

        retract_index = max(0, math.floor((depth*0.038 - 0.46)/0.01))
        retract_index = min(retract_index, min(relocate_index, grasp_index))
        if abs(relocate_index - retract_index <= 2 and relocate_index <= grasp_index): retract_index = relocate_index
        retract_move_config = move_map[retract_index][:]
        retract_move_config[0] += angle1
        
        swipe_config = move_map[retract_index][:]
        swipe_config[0] += angle2

        print(relocate_index, retract_index, grasp_index)

        if move:
            #rtde_c.moveJ(relocate_start_config)
            #move grasp
            rtde_c.moveL_FK(retract_move_config)

            rtde_c.moveJ(swipe_config)
            #move down grasp
            rtde_c.moveL_FK(relocate_move_config)
            rtde_c.moveL_FK(relocate_place_config)
            #place
            gripper.move(0, 100, 0)
            time.sleep(1)
            #move up
            #rtde_c.moveL_FK(relocate_move_config)
            #retrieve
            if i == len(plan)-1:
                rtde_c.moveL_FK(end_config)
            #rtde_c.moveJ(start_config_down_2)

        old_angle = angle2
        
        print('move complete')



def place_objects(rtde_c, gripper, move_map, drop_map, mct_plan_name, move = True):
    
    #starting position for the four objects flipping case
    #curr_config = [[20, 3, 'r'], [20, 6, 'g'], [20, 9, 'b'], [20, 12, 'o']]


    #starting position for the six objects bowling case
    # curr_config = [[4, 2, 'r'], [32, 12, 'g'], [4, 7, 'b'], [12, 8, 'o'], [18, 5, 'p'], [30, 3, 'y']]

    # #stating position for the eight objects parallel case
    # curr_config = [[4, 8, 'r'], [36, 8, 'r'],
    #                              [10, 7, 'o'], [22, 12, 'o'],
    #                              [18, 8, 'g'], [24, 2, 'g'],
    #                              [36, 3, 'b'], [30, 7, 'b']]

    #starting position for the nine objects three square case
    #curr_config = [[4, 8, 'o'], [36, 8, 'r'], [10, 7, 'g'],
    #                             [22, 12, 'b'], [18, 8, 'p'], [24, 2, 'b'],
    #                             [36, 3, 'g'], [30, 7, 'r'], [4, 1, 'o']]
    
    #starting position for cab four objects rotation case
    #curr_config = [[14, 2, 'r'], [14, 8, 'p'], [4, 8, 'b'], [4, 2, 'y']]

    #starting position for the object retrieval case
    #curr_config = [[10, 3, 'b'], [7, 6, 'p'], [13, 6, 'y'], [10, 12, 'r']]


    # curr_config.sort(key = lambda x:x[1], reverse = True)


    #start_config_up_2 = [-0.7776, -1.5929, 2.7465, -1.1525, 0.8413, -3.1415]
    #start_config_down_2 = [-0.7776, -1.4129, 2.7716, -1.3577, 0.8413, -3.1415]
    curr_config = np.load(mct_plan_name, allow_pickle=True)[0]['obj_pos_list']
    curr_config = sorted(curr_config, reverse=True)

    for i in curr_config:
        temp = i[0]
        i[0] = -i[1]
        i[1] = temp

        i[1] += 0.07

    start_config_up_2 = move_map[0]
    start_config_down_2 = drop_map[0]

    rtde_c.moveJ(start_config_up_2)

    rx = 0
    #rx = 0.038*10
    ry = -0.12

    for cx, cy in curr_config:
        print(cx, cy)
        #cx *= 0.038
        #cy *= 0.038


        angle1, angle2 = 0, 0

        angle1 = -math.atan2(cx - rx, cy - ry)

        grasp_x, grasp_y, grasp_z = cx, cy, 0.148
        distance = math.sqrt((grasp_x - rx)**2 + (grasp_y - ry)**2) - 0.44
        grasp_index = round(distance/0.01)
        print(grasp_index)

        #pdb.set_trace()

        grasp_start_config = [angle1 - 0.7776] + start_config_up_2[1:]
        grasp_end_config = [angle1 - 0.7776] + start_config_up_2[1:]
        grasp_move_config = move_map[grasp_index][:]
        grasp_move_config[0] += angle1
        grasp_place_config = drop_map[grasp_index][:]
        grasp_place_config[0] += angle1

        if move:
            rtde_c.moveJ(grasp_start_config)
            gripper.move(0, 100, 0)
            time.sleep(2)
            gripper.move(200, 100, 0)
            time.sleep(2)

            #move down grasp
            rtde_c.moveL_FK(grasp_move_config)
            rtde_c.moveL_FK(grasp_place_config)
            #grasp
            gripper.move(0, 100, 0)
            time.sleep(1)
            #move up
            rtde_c.moveL_FK(grasp_move_config)
            #retrieve
            rtde_c.moveL_FK(grasp_end_config)
            rtde_c.moveJ(start_config_up_2)
        

def init_setup(ip_address):
    move_map = {}
    start_distance = 0
    with open('real_experiment_code/robot_move_map_17.txt', 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            raw_line = ''
            for char in line:
                if char != '[' and char != ']' and char != ',' and char != '\n':
                    raw_line += char
            move_map[start_distance] = [float(x) for x in raw_line.split()]
            start_distance += 1

    place_map = {}
    start_distance = 0
    with open('real_experiment_code/robot_place_map_17.txt', 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            raw_line = ''
            for char in line:
                if char != '[' and char != ']' and char != ',' and char != '\n':
                    raw_line += char
            place_map[start_distance] = [float(x) for x in raw_line.split()]
            start_distance += 1

    drop_map = {}
    start_distance = 0
    with open('real_experiment_code/robot_drop_map_17.txt', 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            raw_line = ''
            for char in line:
                if char != '[' and char != ']' and char != ',' and char != '\n':
                    raw_line += char
            drop_map[start_distance] = [float(x) for x in raw_line.split()]
            start_distance += 1

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ip_address, 63352)
    gripper.move(0, 100, 0)
    
    return move_map, place_map, drop_map, rtde_c, rtde_r, gripper

def main(ip_address, plan_file, plan_file2, mcts_plan, cam_dofs, place_obj_name):

    move_map = {}
    start_distance = 0
    with open('real_experiment_code/robot_move_map_17.txt', 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            raw_line = ''
            for char in line:
                if char != '[' and char != ']' and char != ',' and char != '\n':
                    raw_line += char
            move_map[start_distance] = [float(x) for x in raw_line.split()]
            start_distance += 1

    place_map = {}
    start_distance = 0
    with open('real_experiment_code/robot_place_map_17.txt', 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            raw_line = ''
            for char in line:
                if char != '[' and char != ']' and char != ',' and char != '\n':
                    raw_line += char
            place_map[start_distance] = [float(x) for x in raw_line.split()]
            start_distance += 1

    drop_map = {}
    start_distance = 0
    with open('real_experiment_code/robot_drop_map_17.txt', 'r') as f:
        raw_data = f.readlines()
        for line in raw_data:
            raw_line = ''
            for char in line:
                if char != '[' and char != ']' and char != ',' and char != '\n':
                    raw_line += char
            drop_map[start_distance] = [float(x) for x in raw_line.split()]
            start_distance += 1

    rtde_c = rtde_control.RTDEControlInterface(ip_address)
    rtde_r = rtde_receive.RTDEReceiveInterface(ip_address)

    gripper = robotiq_gripper.RobotiqGripper()
    gripper.connect(ip_address, 63352)
    gripper.move(0, 100, 0)
    time.sleep(2)


    #current_config = rtde_r.getActualQ()

    # power_off_pose(rtde_c)
    rtde_c.moveJ([0.7, -2, 2.5, -0.3, 0.7, 0])
    # rtde_c.moveJ(place_map[0])

    # sys.exit(1)

    # reset_pose(rtde_c, move_map)

    gripper.activate()

    # sys.exit(1)

    place_objects(rtde_c, gripper, move_map, drop_map, place_obj_name, True)
    # sys.exit(1)

    # for i in cam_dofs:
    #     cam_move_plan = RC.get_patha2b(rac, [0.7, -2, 2.5, -0.3, 0.7, 0], i, scene_info)
    pdb.set_trace()

    for angles in cam_dofs:
        for angle in angles:
            rtde_c.moveJ(angle, speed=0.4, acceleration=0.4)
        pdb.set_trace()

    # sys.exit(1)
    # rtde_c.moveJ(place_map[0])


    # pdb.set_trace()
    plan = get_plan_npy(mcts_plan)
    linear_motion_planner_old(rtde_c, gripper,  plan, move_map, place_map, drop_map, True)



    for angle in plan_file:
        rtde_c.moveJ(angle, speed=0.4, acceleration=0.4)

    gripper.move(230, 100, 0)
    time.sleep(5)

    for angle in plan_file2:
        rtde_c.moveJ(angle, speed=0.4, acceleration=0.4)


    # reset_pose(rtde_c, move_map)

    #clap indicates the planning is over
    #gripper.activate()
    

if __name__ == '__main__':
    ip_address = '192.168.0.123'
    # test_name = 'real_test1/'
    test_name = 'real_test2/'


    init2grasp_path = np.load("test_data/test_real_experiment/target_path_saved/banana/init2grasp_path.npy", allow_pickle=True)
    grasp2init_path = np.load("test_data/test_real_experiment/target_path_saved/banana/grasp2init_path.npy", allow_pickle=True)
    scene_info = np.load('test_data/test_real_experiment/' + test_name +'MCTS*/test_results/temp_scene3_success.npy', allow_pickle=True)[0]['scene_info']
    # cam_move_plan = np.load('test_data/test_real_experiment/' + test_name + 'MCTS*/test_results/cam_dofs_3.npy')
    
    rearrangement_plan = 'test_data/test_real_experiment/' + test_name +'MCTS*/test_results/test_result_4.npy'
    place_obj_name = 'test_data/test_real_experiment/' + test_name +'MCTS*/test_results/temp_scene3_success.npy'

    cam_move_plan = []
    f = open('test_data/test_real_experiment/' + test_name + 'MCTS*/test_results/cam_plan.txt', "r")
    rawdata = f.readlines()
    for line in rawdata:
        view_list = []
        div = line[1:-2].split(',[] ')
        div = div[0][1:-1].split('], [')
        for l in div:
            new_list = l.split(',')
            new_list = [float(i) for i in new_list]
            view_list.append(new_list)
        cam_move_plan.append(view_list)
        print(div)

    # rac = RC.robot_arm_configuration('../assets/urdf/ur5e/meshes/collision/', np.array([-0.12, 0, 0]), scene_info)
    # place_objects()

    # init2grasp_path = np.load("test_data/test_real_experiment/target_path_saved/mustard/init2grasp_path.npy", allow_pickle=True)
    # grasp2init_path = np.load("test_data/test_real_experiment/target_path_saved/mustard/grasp2init_path.npy", allow_pickle=True)

    # rtde_c = rtde_control.RTDEControlInterface(ip_address)
    # rtde_c.moveJ([0.7, -2, 2.5, -0.3, 0.7, 0])

    # init2grasp_path = data[0]["init2grasp_path"]
    # grasp2init_path = data[0]["grasp2init_path"]
    # obj_pos_list = data[0]["obj_pos_list"]
    # scene_info = data[0]["scene_info"]
    # test_name = data[0]["test_name"]
    # obj_mesh = data[0]["obj_mesh"]
    # w_target = data[0]["w_target"]
    # idx = data[0]["idx"]
    # obj_pos_list = data[0]["obj_pos_list"]
    # target_mesh = data[0]["target_mesh"]
    # obstacles_num = data[0]["obstacles_num"]
    # target_pos = data[0]["target_pos"]

    # unknown_area = data[0]["unknown_area"]
    # valid_area = data[0]["valid_area"]
    # potential_centers = data[0]["potential_centers"]
    
    # plan_file = init2grasp_path
    # plan_file = sys.argv[1]

    main(ip_address, init2grasp_path, grasp2init_path, rearrangement_plan, cam_move_plan, place_obj_name)
