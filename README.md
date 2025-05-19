Code for

# [Integrating Active Sensing and Rearrangement Planning for Efficient Object Retrieval from Unknown, Confined, Cluttered Environments](https://arxiv.org/pdf/2411.11733)
### Junyoung Kim, Hanwen Ren, and Ahmed H. Qureshi

### IEEE ICRA 2025, Atlanta, Georgia

[![Video Title](https://img.youtube.com/vi/tea7I-3RtV0/0.jpg)](https://www.youtube.com/watch?v=tea7I-3RtV0)

## Dependencies
- environment.yml 
- Isaac Gym https://developer.nvidia.com/isaac-gym
- OMPL https://ompl.kavrakilab.org/ 
- trac-IK https://bitbucket.org/traclabs/trac_ik/src/master/ 



## Usage
Under main_code folder:

`run_retrieval.py` to run proposed methods (MAS + OR-MCTS)

`active_sensing_comp.py` to comparision test on active sensing methods and rearrangement methods with randomly generate environment.

Results can be found under `test_data/test_active_sensing/[DATE&TIME]/test_results/[Sensing Methods]/test_result*.txt`