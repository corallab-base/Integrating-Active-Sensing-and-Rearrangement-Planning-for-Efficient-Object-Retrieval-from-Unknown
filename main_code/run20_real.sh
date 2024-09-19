#!/bin/bash

for i in {1..40}
do
	python sim_for_real_exp.py
	echo $i
done
