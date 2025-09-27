# -*- coding: utf-8 -*-
"""
Created on 2025/9/27

@author: Yifei Sun
"""
import cv2
import os


def video_to_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 读完就退出

        # 保存每一帧为 jpg 图片
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"完成！共提取 {frame_count} 帧。")


# 使用示例
video_to_frames("simulation.mp4", "frames_output")
