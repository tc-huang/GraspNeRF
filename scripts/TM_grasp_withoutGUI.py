#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import rospy
import cv_bridge
### tc-huang
import json
import logging
from pathlib import Path
from robotic_drawing.control.robot_arm.yaskawa.mh5l import MH5L
from robotic_drawing.control.tool.robotiq.gripper_2f85 import Gripper2F85
import numpy as np
HOST = "192.168.255.10"
PORT = 11000
SPEED = 600
# Joint T + 52680 after install bin on flang
INIT_JOINT_CONFIGURE = [
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    46069 + 52680
]

Place_JOINT_CONFIGURE = [
    -52956,
    21581,
    -627,
    967,
    -65817,
    70995
]

offset=np.eye(4)
offset[2,3] = 60 #表示z軸向下
print("offset",offset)



# CALIBRATION_PARAMS_SAVE_DIR = Path(__file__).parent / "calibration_params/2024_03_04"
#CALIBRATION_PARAMS_SAVE_DIR = Path(__file__).parent / "calibration_params/2024_03_05" #沒有筆
CALIBRATION_PARAMS_SAVE_DIR = Path(__file__).parent / "calibration_params/2024_03_06" #有筆
### 

import argparse
from pathlib import Path

#import RobotControl_func_ros1 as RobotControl_func
#import franka_msgs.msg
#from gripper import Gripper 
import geometry_msgs.msg


import sensor_msgs.msg

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.perception import *
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
#from vgn.utils.panda_control import PandaCommander


import shutil
import os
import sys
import cv2
import ArUco_detection
from vgn.detection import *
from vgn.perception import *

import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

sys.path.append(os.path.dirname(__file__) + os.sep)  #os.sep 表示当前操作系统的路径分隔符，如在 Windows 上是 \，在 Unix 上是 /。

# tag lies on the table in the center of the workspace 代表工作空間的中心位置
#[266.4832458496094, -380.7168884277344, 34.46820068359375, 3.1219031047025103, 0.021591066711545426, 0.8871005570020483]弧度 mm
                                                            #[178.87187194824222, 1.2370769977569582, 50.827117919921875]這是角度 這是左上角點
# T_base_tag = Transform(Rotation.from_quat([-0.01686208 , 0.91583859 , 0.40110176 ,-0.00852709]), [0.38813415527343756, -0.3717374877929688, 0.04458283996582031]) #手臂下去點下去放置中心就知道位置了!!!!!!!!!! 單位是m

#[263.5168762207031, -395.23748779296875, 44.18534851074219, -3.1125186576532893, 0.06837218354820175, 0.8255917224123874] 透明格正中心corner
#T_base_tag = Transform(Rotation.from_quat([ -0.00455161 , 0.89535781 , 0.44481489, -0.02129305]), [0.2095004119873047, -0.3604853515625, 0.10233863830566406]) #手臂下去點下去放置中心就知道位置了!!!!!!!!!! 單位是m
#右上角aruco中心點
#[212.62576293945312, -574.6563720703125, 114.51171875, 3.102297711705674, 0.023205965984677442, 0.8183563132048056]
T_base_tag = Transform(Rotation.from_quat([ 0.9171186 ,  0.39796103 ,-0.00282702 , 0.02263848]), [0.2126485137939453, -0.5746722412109375, 0.11454066467285156])

#表示相機座標到aruco的座標
T_cam_task_m = Transform(Rotation.from_quat([0.0091755 ,  0.9995211 ,  0.00176319 ,-0.02950025]), [ 0.16363484, -0.14483834 , 0.44753983])
round_id = 0
#[289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886689848271933, 0.045914536701512146, 0.7913491300112848]
#置中座標
# [[ 0.99959658 -0.00447668 -0.02804718]
#  [ 0.00441263  0.99998751 -0.00234523]
#  [ 0.02805733  0.00222052  0.99960385]]
# [[-85.23109251]
#  [-73.48531014]
#  [355.95850582]]  目前位置的相機外參

class PandaGraspController(object):
    def __init__(self, args):
        self.robot_error = False

        self.base_frame_id = "panda_link0" #用途不明
        self.tool0_frame_id = "panda_link8" #用途不明

        
        #self.finger_depth = rospy.geT_tool0_tcpt_param("~finger_depth") 這是從.yaml的設定檔案拉參數出來用的方式
        self.finger_depth = 0.04 # 單位m

        #self.size = 6.0 * self.finger_depth
        self.size = 0.3 #workspace空間


        #self.robot = RobotControl_func.RobotControl_Func() #手臂控制
        ### TODO tc-huang
        #self.grip = Gripper()
        ###

        #self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = VGN(args.model, rviz=True)
        
        rospy.loginfo("Ready to take action")

    def setup_panda_control(self):
        # rospy.Subscriber(
        #     "/franka_state_controller/franka_states",
        #     franka_msgs.msg.FrankaState,
        #     self.robot_state_cb,
        #     queue_size=1,
        # )
        rospy.Subscriber(
            "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        )
        # self.pc = PandaCommander()
        # self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

        
    def define_workspace(self):
        #注意板子比擺放台高了0.5cm 也就是50mm
        z_offset = 0 #表示aruco實際上比點的位置高了10公分
        #-13 [269.8632507324219, -282.0752868652344, 45.686264038085945, 3.116979718795281, 0.06733950122363148, 1.125348279701082]
        #  -15cm - 15cm   7x7 aruco 
        t_tag_task = np.r_[[-0.5 * self.size, 0.5 * self.size, z_offset]]  #雖然意義不明但是猜測是指tag到tsdf建模的起點,表示棋盤格中心到task，看來不能省略
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        #self.T_base_task = T_base_tag * T_tag_task  #物體到base的變換矩陣？？

        self.T_base_task = T_base_tag 
        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")  #檢查這裡
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self): #??
        # collision box for table
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = ros_utils.to_pose_msg(T_base_tag)
        msg.pose.position.z -= 0.01  #?未知
        #下方式避障用
        #self.pc.scene.add_box("table", msg, size=(0.6, 0.6, 0.02))

        rospy.sleep(1.0)  # wait for the scene to be updated

    def robot_state_cb(self, msg):
        detected_error = False
        if np.any(msg.cartesian_collision):
            detected_error = True
        # for s in franka_msgs.msg.Errors.__slots__:
        #     if getattr(msg.current_errors, s):
        #         detected_error = True
        if not self.robot_error and detected_error:
            self.robot_error = True
            rospy.logwarn("Detected robot error")

    def joints_cb(self, msg): #?
        ### TODO tc-huang
        #self.gripper_width = msg.position[7] + msg.position[8]
        pass
        ###

    def run(self, robot_arm,T_cam2gripper, gripper):
        vis.clear()
        vis.draw_workspace(self.size)
        #self.pc.move_gripper(0.08)
        #self.pc.home()
        #沒墊東西的位置
        #self.robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461])
        # TODO: set pose here
        # self.robot.set_TMPos([269.2006225585938, -464.2568359375, 422.3720703125, 3.10223965479391, 0.023221647426189884, 0.8183750884904907])
        ### TODO: tc-huang
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        logging.info("[Robot] Move to initial pose by joint configure")
        ###

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc) #模擬環境跑一次看看
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return

        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")
        ### TODO tc-huang
        #self.grip.gripper_on()
        ###
        # self.pc.home()
        label = self.execute_grasp(grasp, robot_arm=robot_arm, Camera2Gripper=T_cam2gripper, gripper=gripper)
        rospy.loginfo("Grasp execution")
        rospy.sleep(1)
        # if self.robot_error:
        #     self.recover_robot()
        #     return

        # if label:
        #     self.drop()
        # self.pc.home()
        #self.robot.set_TMPos([289.77935791015625, -464.6210632324219, 296.10107421875, 3.0886695174594117, 0.045914965304000084, 0.7913568531784461]) #home
        ### TODO tc-huang
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
        logging.info("[Robot] Move to initial pose by joint configure")
        ###

    def acquire_tsdf(self): #移動去其他視角拍照
        fr = open("./scripts/PosSet.txt", 'r+')  #這時還沒有抓取joint的function所以使用position當作點位
        
        #print(Total_pose)
        # print(j)
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True
        
        # #先拍一張就好
        # for Pose in iter(fr): #拍照用
        #     #QApplication.processEvents()
        #     Pose = Pose.replace('[', '').replace(']', '').replace('\n', '').split(', ')
        #     #print(Pose)
        #     Pose = np.array(Pose, dtype="float")
        #     print('Position:' + str(Pose)) 
        #     self.robot.set_TMPos([Pose[0],Pose[1],Pose[2], Pose[3], Pose[4], Pose[5]])  
        rospy.sleep(2.0)
        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()
        # #測試中＃＃＃＃
        # vis.clear()
        # vis.draw_workspace(0.3)  #30cm
        # from pathlib import Path
        # str_path = "/media/eric/Disk/vgn/scripts/data/models/vgn_conv.pth"
        # path = Path(str_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = load_network(path, self.device)
        # tic = time.time()
        # tsdf_vol = tsdf.get_grid()
        # voxel_size = tsdf.voxel_size
        # print("Extract grid  ", time.time() - tic)

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        # print("Forward pass   ", time.time() - tic)

        # tic = time.time()
        # qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        # print("Filter         ", time.time() - tic)

        # vis.draw_quality(qual_vol, voxel_size, threshold=0.01)
        # #測試中＃＃＃＃

        print("-----------------")
        print(pc)  #目前0點
        print("-----------------")
        return tsdf, pc

    def select_grasp(self, grasps, scores):
        # select the highest grasp
        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score

    def execute_grasp(self, grasp, robot_arm, Camera2Gripper, gripper):  #夾取用的 grasp是從world座標

        T_task_grasp = grasp.pose
        print("模型測出來的位置")
        print(T_task_grasp.translation)
        #T_base_grasp = self.T_base_task
        
        print("grasp tsdf位置",grasp)
        Base_point_t,Base_R_degree=self.EstimateCoord(T_task_grasp, Camera2Gripper, robot_arm)
        print("Base_point_t",Base_point_t)
        print("Base_R_degree",Base_R_degree)
        #T_base_grasp = self.T_base_task *T_task_grasp
        
        #T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])  #移動到前方
        #T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])   #夾起來後往後

        # T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        # T_base_retreat = T_base_grasp * T_grasp_retreat
        # point=T_base_pregrasp * T_tcp_tool0
        # print(point.translation)
        #移動到指定位置
        #或是直接使用T_base_grasp
        # point=T_base_pregrasp * T_tcp_tool0
        
        #degree=T_base_grasp.rotation.as_euler('xyz', degrees=False)

        # t_task_grasp = T_task_grasp.translation
        # R_task_grasp = T_task_grasp.rotation
        # modifed_Rz_degree = Base_R_degree[2]
        # print("original",Base_R_degree)

        # while modifed_Rz_degree<-90 :
        #     modifed_Rz_degree+=180
        #     print("-90")
        # while modifed_Rz_degree>90:
        #     modifed_Rz_degree-=180
        #     print("+90")

        # print("modifed_Rz_degree",modifed_Rz_degree)
        ### TODO: tc-huang
        robot_arm.move_to_pose(
            Tx=Base_point_t[0],
            Ty=Base_point_t[1],
            Tz = Base_point_t[2],
            Rx=Base_R_degree[0],
            Ry=Base_R_degree[1],
            # Rz=modifed_Rz_degree,
            Rz=Base_R_degree[2],

            speed=SPEED)
        #TODO: gripper
        success = gripper.close()
        time.sleep(3)
        robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)  #初始位置

        robot_arm.move_to_joint_config(Place_JOINT_CONFIGURE, SPEED)  #擺放位置
        success = gripper.on()
        # robot_arm.move_to_joint_config([-126330, -19968, 12272, -1538, -100260, 46100 + 52680], SPEED)
        
        ###

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):  #放置位置
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)

    def EstimateCoord(self,T_task_grasp, Camera2Gripper, robot_arm):  #計算座標
        # TODO tc-huang
        tcp_pose = robot_arm.controller.get_robot_pose(tool_num=0)
        print(f"TCP POSE: {tcp_pose}")
        tvec = tcp_pose[:3]

        x_angle, y_angle, z_angle = tcp_pose[3:]
        rvec = Rotation.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=True
        ).as_rotvec(degrees=False)
        
        Gripper2Base= np.eye(4)
        Gripper2Base[:3, :3] = Rotation.from_euler(
            "xyz", [x_angle, y_angle, z_angle], degrees=True
        ).as_matrix()
        Gripper2Base[:3, 3] =np.array(tvec).T

        print(f"TCP POSE mat: {Gripper2Base}")

        # Task2Camera = np.array( #改成算出來的
        #     [
        #         [1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ]
        # )
        print("---------")
        print(T_cam_task_m.as_matrix())
        Task2Camera_r = T_cam_task_m.as_matrix()[:3,:3]
        Task2Camera_t = T_cam_task_m.as_matrix()[:3,3]  #公尺
        print(Task2Camera_r)
        print(Task2Camera_t*1000)

        

        Task2Camera=np.r_[np.c_[Task2Camera_r, Task2Camera_t*1000], [[0, 0, 0, 1]]]
        # grasppoint = np.array([T_task_grasp.translation[0]*1000,T_task_grasp.translation[1]*1000, T_task_grasp.translation[2]*1000, 1]).T
        # grasppoint = np.array([0,0,0,1]).T

        #TODO 限制z軸旋轉
        Grasp2task = np.eye(4)
        T_task_grasp_t = T_task_grasp.as_matrix()[:3,3]
        T_task_grasp_r = T_task_grasp.as_matrix()[:3,:3]
        #TODO 未知
        T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg = Rotation.from_matrix(T_task_grasp_r).as_euler('xyz', degrees=True)
        print("原始角度",T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg )
        # while T_task_grasp_Rz_deg < 0:
        #     T_task_grasp_Rz_deg += 180
        #     print("-90")
        # while T_task_grasp_Rz_deg > 180:
        #     T_task_grasp_Rz_deg -= 180
        #     print("+90")
        # T_task_grasp_r = Rotation.from_euler('xyz', [T_task_grasp_Rx_deg, T_task_grasp_Ry_deg, T_task_grasp_Rz_deg], degrees=True).as_matrix()
        
        Grasp2task[:3,:3] = T_task_grasp_r
        Grasp2task[:3,3] = T_task_grasp_t * 1000

        # Grasp2task=np.r_[np.c_[T_task_grasp_r, T_task_grasp_t], [[0, 0, 0, 1]]]
        print("grasp2task",Grasp2task)

        tm = TransformManager()
        tm.add_transform("grasp", "task", Grasp2task)
        # tm.add_transform("gripper", "robot", Gripper2Base)
        # tm.add_transform("camera", "gripper", Camera2Gripper)
        # tm.add_transform("task", "camera", Task2Camera)

        #ee2object = tm.get_transform("end-effector", "object")

        ax = tm.plot_frames_in("task", s=100)
        ax.set_xlim((-1000, 1000))
        ax.set_ylim((-1000, 1000))
        ax.set_zlim((-1000, 1000))
        plt.show()


        Base_point_T = Gripper2Base @ Camera2Gripper @ Task2Camera @ Grasp2task @ offset    #從右邊往左看,相機座標到夾爪座標再到base座標
        print("Camera2Gripper", Camera2Gripper)
        print("Gripper2Base",Gripper2Base)
        print("Task2Camera",Task2Camera)
        Base_point_t = Base_point_T[:3, 3] #3x1 T
        Base_point_r = Base_point_T[:3,:3] #3x3
        Base_R_degree= Rotation.from_matrix(Base_point_r).as_euler('xyz',degrees=True)
        

        return Base_point_t,Base_R_degree


class TSDFServer(object):
    def __init__(self):
        self.cam_frame_id = "camera_depth_optical_frame"
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        ### TODO change intrinsic tc-huang
        self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)
        self.distortion=np.array([ 4.90913179e-02 , 5.22699002e-01, -2.65209452e-03  ,1.13033224e-03,-2.17133474e+00])

        #

        #self.size = 6.0 * rospy.get_param("~finger_depth") #finger_depth  但是6的意義不明 但我們的爪子應該是0.04 0.05是paper預設大小
        self.size = 0.3 #finger_depth  但是6的意義不明 但我們的爪子應該是0.04 0.05是paper預設大小
        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)
        
    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return

        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        #T_cam_task =  ArUco_detection.Img_ArUco_detect(img,self.intrinsic,self.distortion)
        #用來廣播目標平面到相機的關係，也就是相機外參
        self.tf_tree.broadcast_static(
            T_cam_task_m, self.cam_frame_id, "task"
        )


        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task_m)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task_m)


def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    ### TODO: tc-huang
    with open(str(CALIBRATION_PARAMS_SAVE_DIR / 'calibration_params.json')) as f:
        calibration_params = json.load(f)
        T_cam2gripper = np.array(calibration_params["T_cam2gripper"])
        T_target2cam = np.array(calibration_params["T_target2cam"])
        T_gripper2base = np.array(calibration_params["T_gripper2base"])
        intrinsic_matrix = np.array(calibration_params["intrinsic_matrix"])
        distortion_coefficients = np.array(calibration_params["distortion_coefficients"])
    logging.info("Init")
    gripper = Gripper2F85()
    logging.info("Connect")
    success = gripper.connect()
    logging.info("Reset")
    success = gripper.reset()

    robot_arm = MH5L(host=HOST, port=PORT)
    logging.info("[Robot] Init")
    robot_arm.power_on()
    logging.info("[Robot] Power on")
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")
    
    ###
    ### TODO: times
    for i in range(3):
        robot_arm.move_to_joint_config([-126330, -19968, 12272, -1538, -100260, 46100 + 52680], SPEED)
        panda_grasp.run(robot_arm, T_cam2gripper, gripper) 
    
    ### TODO tc-huang
    robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
    logging.info("[Robot] Move to initial pose by joint configure")

    robot_arm.power_off() 
    logging.info("[Robot] Power off")
    ### 
    


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        S = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / S
        x = (R[2, 1] - R[1, 2]) * S
        y = (R[0, 2] - R[2, 0]) * S
        z = (R[1, 0] - R[0, 1]) * S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])
        
def draw_axis(Gripper2Base,Camera2Gripper,Task2Camera):
    tm = TransformManager()
    tm.add_transform("gripper", "robot", Gripper2Base)
    tm.add_transform("camera", "gripper", Camera2Gripper)
    tm.add_transform("task", "camera", Task2Camera)

    #ee2object = tm.get_transform("end-effector", "object")

    ax = tm.plot_frames_in("robot", s=100)
    ax.set_xlim((-1000, 1000))
    ax.set_ylim((-1000, 1000))
    ax.set_zlim((0.0, 1000))
    #plt.show()

def RotationTrans(v):
    #目前是使用弧度
    # print("----------VVVVVV------")
    # print("v:",v)
    #pi = np.pi / 180 #轉成弧度
    # tmp_v = v[0]
    # v[0] = v[2]
    # v[2] = tmp_v
    # pi =   1
    r1_mat = np.zeros((3, 3), np.float32)
    r2_mat = np.zeros((3, 3), np.float32)
    r3_mat = np.zeros((3, 3), np.float32)

    r = np.zeros((3, 1), np.float32)
    r[0] = 0
    r[1] = 0
    #r[2] = float(v[2]) * pi # 如果是角度轉成弧度
    r[2] = float(v[2]) 
    r3_mat, jacobian = cv2.Rodrigues(r)
    # print("r3_mat:",r3_mat)
    r[0] = 0
    r[1] = float(v[1])
    r[2] = 0
    # print('ys ', math.sin(v[1]))
    # print('yc ', math.cos(v[1]))
    r2_mat, jacobian = cv2.Rodrigues(r)
    # print("r2_mat:",r2_mat)
    r[0] = float(v[0])
    r[1] = 0
    r[2] = 0
    r1_mat, jacobian = cv2.Rodrigues(r)

    result = np.dot(np.dot(r3_mat, r2_mat), r1_mat)
    print(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default="/home/eric/catkin_ws/src/vgn/scripts/data/models/vgn_conv.pth")
    args = parser.parse_args()
    main(args)
