#!/bin/bash

for i in {1..10}
do
	python ur5e_test_multi_mcts.py
	echo $i
done
