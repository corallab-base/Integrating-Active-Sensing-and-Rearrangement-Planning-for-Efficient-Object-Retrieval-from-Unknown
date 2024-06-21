#!/bin/bash

input="test_list.txt"
input2="validation_list.txt"
srcs="scene_generation_0602/"
tags="scene_test/"
tags2="scene_validation"
while IFS= read -r line
do
  echo "$srcs$line" &&
  mv "$srcs$line" "$tags"
done < "$input"

while IFS= read -r line
do
  echo "$srcs$line" &&
  mv "$srcs$line" "$tags2"
done < "$input2"
