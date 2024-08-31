#!/bin/bash

for i in {1..20}
do
	python manual_scene_active_sensing.py
	echo $i
done
