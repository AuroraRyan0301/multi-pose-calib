#!/bin/bash

cd "/home/yuruihan/DS-FaceScape/hdr_emitter"

# 定义文件夹列表
# folders=("merge_out_2023_12_07_180degree" "merge_out_2023_12_07_90degree" "merge_out_2023_12_07_degree" "merge_out_2023_12_07_270degree" "merge_out_2023_12_07_env")
folders=("new_align_merge_out_2023_12_07_env")
# 遍历每个文件夹
for folder in "${folders[@]}"
do
    # 检查文件夹是否存在
    if [ -d "$folder" ]; then
        # 创建新文件夹
        new_folder="png_$folder"
        mkdir -p "$new_folder"
        
        # 进入原始文件夹
        cd "$folder"
        
        # 遍历文件夹内的exr文件
        for file in *.exr
        do
            # 检查文件是否存在
            if [ -f "$file" ]; then
                # 转换exr文件为png格式并保存到新文件夹中
                convert "$file" -colorspace sRGB "../$new_folder/${file%.exr}.png"
            fi
        done
        
        # 返回上一级目录
        cd ..
    fi
done