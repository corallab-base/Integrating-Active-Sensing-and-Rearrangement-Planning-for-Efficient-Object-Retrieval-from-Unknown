#!/bin/bash

for i in {1..200}
do
	python manual_scene_active_sensing.py
	echo $i
done
