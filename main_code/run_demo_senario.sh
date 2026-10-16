#!/bin/bash

for i in {1..40}
do
	python robot_arm_configuration_lab_demo.py
	echo $i
done
