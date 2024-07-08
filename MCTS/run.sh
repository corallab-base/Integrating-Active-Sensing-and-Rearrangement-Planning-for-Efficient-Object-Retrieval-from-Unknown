#!/bin/bash

for i in {0..400}
do
	timeout 60 python3 MCTS_algo_ICRA.py 2 $i 0 f
	#timeout 60 python3 mRS.py 2 $i
done
