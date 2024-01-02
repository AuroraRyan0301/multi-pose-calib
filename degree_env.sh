#!/bin/bash

# 创建degree_env文件夹
parent_path=/home/yuruihan/DS-FaceScape/hdr_emitter
cd $parent_path
mkdir degree_env

keywords=("degree" "90degree" "180degree" "270degree")

env_keyword="env"

for keyword in "${keywords[@]}"
do
    cp -R "align_merge_out_2023_12_07_${keyword}_ldr/" "degree_env/"
    cp -RT "new_align_merge_out_2023_12_07_${env_keyword}_ldr/" "degree_env/align_merge_out_2023_12_07_${keyword}_ldr/"
done

echo "数据整理完成！"