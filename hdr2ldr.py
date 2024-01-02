import os
import cv2
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def linear_to_srgb(img):
    limit = 0.0031308
    img = np.where(img > limit, 1.055 * img ** (1 / 2.4) - 0.055, 12.92 * img)
    img[img > 1] = 1  # "clamp" tonemapper
    return img

# 定义数据集路径
dataset_path = "/home/yuruihan/DS-FaceScape/hdr_emitter"

# 定义文件夹列表
# folders = ["align_merge_out_2023_12_07_180degree", "align_merge_out_2023_12_07_90degree", "align_merge_out_2023_12_07_degree", "align_merge_out_2023_12_07_270degree", "align_merge_out_2023_12_07_env"]
folders  = ["2023_12_30"]
# 遍历每个文件夹
for folder in folders:
    # 检查文件夹是否存在
    if os.path.isdir(os.path.join(dataset_path, folder)):
        # 创建新文件夹
        new_folder = os.path.join(dataset_path, folder + "_ldr")
        os.makedirs(new_folder, exist_ok=True)
        
        # 遍历文件夹内的exr文件
        for file in os.listdir(os.path.join(dataset_path, folder)):
            if file.endswith(".exr"):
                exr_path = os.path.join(dataset_path, folder, file)
                
                # 使用OpenCV读取HDR图像
                hdr_img = cv2.imread(exr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                
                # 将图像转换为LDR格式（应用色调函数）
                ldr_img = linear_to_srgb(hdr_img)
                
                # 将像素值缩放回0-255之间，并保存为PNG格式
                ldr_img = (ldr_img * 255).clip(0, 255).astype(np.uint8)
                png_path = os.path.join(new_folder, os.path.splitext(file)[0] + ".png")
                print(png_path)
                cv2.imwrite(png_path, ldr_img)