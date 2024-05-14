#!/usr/bin/env python

"""
Real-time grasp detection.
"""

import argparse
from pathlib import Path
import time

import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
import torch

import sys
sys.path.append('src')

from gd import vis
from gd.detection import *
from gd.perception import *
from gd.utils import ros_utils
from gd.utils.transform import Rotation, Transform

# from vgn import vis
# from vgn.detection import *
# from vgn.perception import *
# from vgn.utils import ros_utils
# from vgn.utils.transform import Rotation, Transform
import rospy
#import RobotControl_func_ros1 as RobotControl_func


#可能要考慮undistort
class GraspDetectionServer(object):
    def __init__(self):
        # define frames
        self.task_frame_id = "task"
        self.cam_frame_id = "camera_depth_optical_frame"
        # Rotation_matrix=Rotation.as_matrix([[ 0.99888145, -0.04660927 , 0.00796429], #?
        #                                     [ 0.04537618 , 0.99223829,  0.11577645],
        #                                     [-0.01329873 ,-0.11528555,  0.99324337]])
        # translation_martix=[39.65148707,-1.8229379 ,29.29505309]  #?
        # print("test")
        # print(Rotation.from_quat([-0.679, 0.726, -0.074, -0.081]), [0.166, 0.101, 0.515])
        # self.T_cam_task = Transform(
        #     Rotation.from_quat([-0.679, 0.726, -0.074, -0.081]), [0.166, 0.101, 0.515]  #要換成我們的
        # )
        # print(Rotation.from_quat([-0.679, 0.726, -0.074, -0.081]), [0.166, 0.101, 0.515])
        # print("2323132")
        # print(Transform.from_matrix(np.array([[0.99888145, -0.04660927 , 0.00796429, 39.65148707],  #格式正確但不對,要填入相機外參，表示單一視角物體到相機的座標變換矩陣
        #                          [0.04537618 , 0.99223829,  0.11577645,-1.8229379],
        #                          [-0.01329873 ,-0.11528555,  0.99324337,29.29505309],
        #                          [0,0,0,1]])))
       

        #分成rotatio(或ndarray類別)
        #tralslation_martix=np.load('/home/eric/catkin_ws/src/vgn/scripts/EIH/TC2G.npy')
        
        #讀取translation並轉成spicy.rotation的類別 
    #     rotation_martix=np.asarray([[ 9.97138831e-01,  7.19587478e-02,  2.31536055e-02],
    #    [ -7.36243805e-02,  9.93934636e-01,  8.16908186e-02],
    #    [-1.71348015e-02, -8.31617572e-02,  9.96388740e-01]])
    #     tralslation_martix=[[-8.64144783e+01],[1.14069757e+01],[-2.42644735e+01]]
    #     rotation_martix=Rotation.from_matrix(rotation_martix)

        #tralslation可以用list 或ndarray
        # self.T_cam_task = Transform(rotation_martix,tralslation_martix)

        #原始如下但要y軸變成+的才可以完整容納 並顯示,後面三項式用來顯示再rviz中的xyz設定
        #Rotation.from_quat([ 0.99989849,  0.00114155, -0.01402755,  0.00222255]), [-0.08523, -0.07348 , 0.35595]  #我們的相機外參
        self.T_cam_task = Transform(  #目前的初始位置,偏移矩陣單位是m
            Rotation.from_quat([ 0.99989849,  0.00114155, -0.01402755,  0.00222255]), [-0.08523, 0.07348 , 0.35595]  #我們的
            
        )
        # self.T_cam_task = Transform(
        #     Rotation.from_quat([-0.679, 0.726, -0.074, -0.081]), [0.166, 0.101, 0.515]  #原始的
        # )
        # broadcast the tf tree (for visualization)
        self.tf_tree = ros_utils.TransformTree()
        self.tf_tree.broadcast_static(
            self.T_cam_task, self.cam_frame_id, self.task_frame_id
        )

        # define camera parameters
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        
        #樓上的
        self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)

        #我帶著的realsense 
        #self.intrinsic = CameraIntrinsic(640, 480, 382.94, 382.94, 320.475, 233.976)

        #p[319.108 241.311]  f[611.058 611.026]  來自realsesne
        #self.intrinsic = CameraIntrinsic(640, 480, 382.94, 382.94, 320.475, 233.976)  #??
        # setup a CV bridge 用於把ros影像轉成cv影像
        self.cv_bridge = cv_bridge.CvBridge()

        # construct the grasp planner object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = load_network(model_path, self.device)

        # initialize the visualization
        vis.clear()
        vis.draw_workspace(0.3)  #30cm

        # subscribe to the camera
        self.img = None
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

        # setup cb to detect grasps
        rospy.Timer(rospy.Duration(1), self.detect_grasps) #多久一次

    def sensor_cb(self, msg):
        self.img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001  #0.001?也許是換成m 把msg轉成opencv格式

    def detect_grasps(self, _):
        if self.img is None:
            return

        tic = time.time()
        self.tsdf = TSDFVolume(0.3, 40) #預設0.3
        self.tsdf.integrate(self.img, self.intrinsic, self.T_cam_task.as_matrix())    #改成多個視角同時需要多個變換矩陣camera 到 task
        print("Construct tsdf ", time.time() - tic)                       # 

        tic = time.time()
        tsdf_vol = self.tsdf.get_grid()
        print(f"TSDF:\n {np.any(tsdf_vol)}")
        voxel_size = self.tsdf.voxel_size
        print("Extract grid  ", time.time() - tic)

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        # print("Forward pass   ", time.time() - tic)

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        # print("Filter         ", time.time() - tic)

        # vis.draw_quality(qual_vol, voxel_size, threshold=0.01)

        # tic = time.time()
        # grasps, scores = select(qual_vol, rot_vol, width_vol, 0.90, 1)
        # num_grasps = len(grasps)
        # if num_grasps > 0:
        #     idx = np.random.choice(num_grasps, size=min(5, num_grasps), replace=False)
        #     grasps, scores = np.array(grasps)[idx], np.array(scores)[idx]
        # grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps]
        # print("Select grasps  ", time.time() - tic)

        # vis.clear_grasps()
        # rospy.sleep(0.01)
        # tic = time.time()
        # vis.draw_grasps(grasps, scores, 0.05)
        # print("Visualize      ", time.time() - tic)

        self.img = None
        print()


if __name__ == "__main__":
    rospy.init_node("RobotControl",anonymous=True)
    # robot = RobotControl_func.RobotControl_Func()
    # robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461])
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=Path, default="/home/eric/Grasp_detection_vgn/scripts/data/models/vgn_conv.pth")
    # args = parser.parse_args()

    GraspDetectionServer()
    rospy.spin()
