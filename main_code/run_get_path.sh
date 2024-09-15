#!/bin/bash

for i in {1..20}
do
	python sim_for_path.py
	echo $i
done
