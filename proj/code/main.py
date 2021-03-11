# -*- coding: UTF-8 -*-
import cv2 as cv
import argparse
import numpy as np
import time
import os
import pyspin
import EasyPySpin
import sys
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize


parser = argparse.ArgumentParser(description='Gesture control camera based on OpenPose')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# 导入相关模型
#尝试mobile_thin
#estimator = load_pretrain_model('mobilenet_thin')
#estimator = load_pretrain_model('mobilenet_small')
estimator = load_pretrain_model('VGG_origin')
action_classifier = load_action_premodel('Action/own_stand_wave_08.h5')


# 参数初始化
realtime_fps = '0.0000'
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0


#获取被控相机
cap_Receptor = EasyPySpin.VideoCapture(0)

# 获取主控相机
cap_main = choose_run_mode(args)

# 读写视频文件
video_writer = set_video_writer(cap_main, write_fps=int(7.0))
# print("DEBUG:stage 1")


# # 保存多组关节数据的txt文件，用于训练过程(for training),重命名为wave_*.txt/stand_*.txt
# f = open('test_out/origin_data.txt', 'a+')


while cv.waitKey(1) < 0:
    #print("DEBUG:stage 2")
    has_frame, show = cap_main.read()
    
    #print("DEBUG:stage 3")
    if has_frame:
        fps_count += 1
        frame_count += 1

        # 灰度图像转为RGB图像
        show=cv.cvtColor(show, cv.COLOR_GRAY2RGB)
        # 姿势识别,human存储着标记好骨骼的数据（以图像的相对位置表示）
        humans = estimator.inference(show)
        # get pose info，pose中存储着绘制好关键节点的姿势数据，主要目的是计算出关节的实际坐标位置，绘制骨骼，并返回整理完后的数据
        pose = TfPoseVisualizer.draw_pose_rgb(show, humans,)  # return frame, joints, bboxes, xcenter
        # recognize the action framewise
        show = framewise_recognize(pose, action_classifier,cap_Receptor)

        height, width = show.shape[:2]
        # 显示实时FPS值
        if (time.time() - start_time) > fps_interval:
            # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
            realtime_fps = fps_count / (time.time() - start_time)
            fps_count = 0  # 帧数清零
            start_time = time.time()
        fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
        cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示检测到的人数
        num_label = "Human: {0}".format(len(humans))
        cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 显示目前的运行时长及总帧数
        if frame_count == 1:
            run_timer = time.time()
        run_time = time.time() - run_timer
        time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
        cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv.imshow('Gesture control camera based on OpenPose', show)
        video_writer.write(show)

        # 采集数据，用于训练过程(for training)
        # joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
        # f.write(' '.join(joints_norm_per_frame))
        # f.write('\n')

input('The program finish!\n')

video_writer.release()
cap_main.release()
cap_Receptor.release()
#f.close()
