#!/usr/bin/env python

"""
Open-loop grasp execution using a Panda arm and wrist-mounted RealSense camera.
"""

import argparse
from pathlib import Path

import cv_bridge
# import franka_msgs.msg
import geometry_msgs.msg
import numpy as np
import rospy
import sensor_msgs.msg

import sys
sys.path.append('src')

from gd import vis
from gd.experiments.clutter_removal import State
from gd.detection import VGN
from gd.perception import *
from gd.utils import ros_utils
from gd.utils.transform import Rotation, Transform
from gd.utils.panda_control import PandaCommander

# from nr.main import GraspNeRFPlanner

import cv2

from pysurroundcap.util import *
from pysurroundcap.data_struct import LevelPosesConfig
from pysurroundcap.pose import create_poses


# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.42, 0.02, 0.21])
round_id = 0

CALIBRATION_PARAMS_JSON = '/catkin_ws/GraspNeRF/calibration_params_0.json'
import json
with open(CALIBRATION_PARAMS_JSON) as f:
    calibration_params = json.load(f)
    T_cam2gripper = np.array(calibration_params["T_cam2gripper"])
    T_target2cam = np.array(calibration_params["T_target2cam"])
    T_gripper2base = np.array(calibration_params["T_gripper2base"])
    intrinsic_matrix = np.array(calibration_params["intrinsic_matrix"])
    distortion_coefficients = np.array(calibration_params["distortion_coefficients"])

ARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
ARUCO_MARK_LENGTH = 30

WORKSPACE_SIZE_MM = 200

class PandaGraspController(object):
    def __init__(self, args):
        self.robot_error = False

        # self.base_frame_id = rospy.get_param("~base_frame_id")
        # self.tool0_frame_id = rospy.get_param("~tool0_frame_id")
        # self.T_tool0_tcp = Transform.from_dict(rospy.get_param("~T_tool0_tcp"))  # TODO
        # self.T_tcp_tool0 = self.T_tool0_tcp.inverse()
        # self.finger_depth = rospy.get_param("~finger_depth")
        # self.size = 6.0 * self.finger_depth
        # self.scan_joints = rospy.get_param("~scan_joints")
        
        # TODO: need check
        self.base_frame_id = '0'
        self.finger_depth = 0.04 # gripper finger depth meter
        self.size = 0.3 # workspace size in meter
        
        self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        self.plan_grasps = VGN(args.model, rviz=True)
        
        # self.grasp_planner = GraspNeRFPlanner(args)

        rospy.loginfo("Ready to take action")

    def setup_panda_control(self):
        # rospy.Subscriber(
        #     "/franka_state_controller/franka_states",
        #     franka_msgs.msg.FrankaState,
        #     self.robot_state_cb,
        #     queue_size=1,
        # )
        # rospy.Subscriber(
        #     "/joint_states", sensor_msgs.msg.JointState, self.joints_cb, queue_size=1
        # )
        self.pc = PandaCommander()
        # self.pc.move_group.set_end_effector_link(self.tool0_frame_id)

    def define_workspace(self):
        z_offset = -0.06
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task

        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self):
        # collision box for table
        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = self.base_frame_id
        msg.pose = ros_utils.to_pose_msg(T_base_tag)
        msg.pose.position.z -= 0.01
        # self.pc.scene.add_box("table", msg, size=(0.6, 0.6, 0.02)) # moveit related

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

    def joints_cb(self, msg):
        self.gripper_width = msg.position[7] + msg.position[8]

    def recover_robot(self):
        self.pc.recover()
        self.robot_error = False
        rospy.loginfo("Recovered from robot error")

    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        # TODO: need implimentation
        # self.pc.move_gripper(0.08)
        # self.pc.home()

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        # vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size * 0.001) 
        # vis.draw_points(np.asarray(pc.points))
    
        # vis.draw_points(np.asarray([[300, 300, 300]]))
        rospy.loginfo("Reconstructed scene")


        state = State(tsdf, pc)
        
        # TODO: JH need to check 
        grasps, scores, planning_time = self.plan_grasps(state)
        # timings = {}
        # n_grasp = 0
        # grasps, scores, timings["planning"] = grasp_plan_fn(render_frame_list, round_idx, n_grasp, gt_tsdf)
        
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")

        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return
        
        # TODO: JH need to check
        # grasp, score = grasps[0], scores[0]
        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")

        # TODO: need implimentation
        # self.pc.home()
        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")

        if self.robot_error:
            self.recover_robot()
            return

        if label:
            self.drop()
        # TODO: need implimentation
        self.pc.home()
    
    def __aruco_detect(self, image):
        frame = image.copy()       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, ARUCO_DICTIONARY, parameters=ARUCO_PARAMETERS)
        rvec, tvec = None, None
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], ARUCO_MARK_LENGTH, intrinsic_matrix, distortion_coefficients)
                frame = cv2.drawFrameAxes(frame, intrinsic_matrix, distortion_coefficients, rvec, tvec, 30)
                # frame = workspace_AR(frame, tvec[i], rvec[i], intrinsic_matrix, distortion_coefficients, WORKSPACE_SIZE_MM, TSDF_SIZE)
                break
            return tvec[0], rvec[0], frame
        else:
            rospy.logerr("No markers detected")
  
    def __obtain_inition_coordinate_relation(self):
        self.pc.home()
        tcp_pose = self.pc.get_pose()
        tvec = tcp_pose[:3]
        x_fix_angle, y_fix_angle, z_fix_angle = tcp_pose[3:]
        T_gripper2base_init = xyzrpy_to_T(tvec[0], tvec[1], tvec[2], x_fix_angle, y_fix_angle, z_fix_angle)
        # print(f"T_gripper2base_init:\n {T_gripper2base_init}")
        
        self.tsdf_server.catch_image = True
        rospy.sleep(0.1)
        tvec, rvec, aruco_detect_result = self.__aruco_detect(self.tsdf_server.rgb_image)
        # cv2.imshow('image', self.tsdf_server.rgb_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        T_aruco2cam_init = tvec_rvec_to_T(tvec, rvec, use_degree=False)
        print(f"T_aruco2cam_init:\n {T_aruco2cam_init}")
        
        cv2.imshow('image for aruco', aruco_detect_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return T_gripper2base_init, T_aruco2cam_init
    
    def __get_ee_base_poses(self, camera_aruco_poses, T_gripper2base_init, T_target2cam_init):
        results = []
        delta = 0
        # transforms_data = TransformsJson()
        for i, camera_aruco_pose in enumerate(camera_aruco_poses):
            T_cam2aruco = xyzrpy_to_T(*camera_aruco_pose)
            # transforms_data.add_frame(f"ws_size_{WORKSPACE_SIZE_MM}_r_{cfg[0]}_theta_{cfg[1]}_phi_{cfg[2]}.png", i, T_cam2aruco.tolist())
            T_gripper2cam = np.linalg.inv(T_cam2gripper)
            T_aruco2base = T_gripper2base_init @ T_cam2gripper @ T_target2cam_init
            T_new = T_aruco2base @ T_cam2aruco @ T_gripper2cam
            x, y, z = T_new[:3, 3]
            rx, ry, rz = R.from_matrix(T_new[:3, :3]).as_euler("xyz", degrees=True)
            results.append([x, y, z, rx, ry, rz])
        
        # transforms_data.save_json('transforms.json')
        return results
    
    def __camera_pose_2_Ttarget2cam(self, camera_pose):
        x, y, z = camera_pose.x, camera_pose.y, camera_pose.z
        rx_deg, ry_deg, rz_deg = camera_pose.rx_deg, camera_pose.ry_deg, camera_pose.rz_deg
        Ttarget2cam = xyzrpy_to_T(x, y, z, rx_deg, ry_deg, rz_deg) 
        return Ttarget2cam
     
    def acquire_tsdf(self):
        # TODO: need implimentation
        T_gripper2base_init, T_aruco2cam_init = self.__obtain_inition_coordinate_relation()
        level_poses_list = [LevelPosesConfig(theta_deg=30, phi_begin_deg=0, phi_end_deg=360, r=400, num_poses=6)]
        camera_poses = create_poses(level_poses_list, WORKSPACE_SIZE_MM)
        # print(f"camera_poses:\n {camera_poses}")
        ee_base_poses = self.__get_ee_base_poses(camera_poses, T_gripper2base_init, T_aruco2cam_init)
        # print(f"ee_base_pose:\n {ee_base_poses}")
        Ttarget2cam_list = [self.__camera_pose_2_Ttarget2cam(camera_pose) for camera_pose in camera_poses]
        
        self.tsdf_server.T_cam_task = T_aruco2cam_init

        # TODO: need to check   
        # self.pc.goto_joints(self.scan_joints[0])
        rospy.sleep(0.1) 

        self.pc.goto_pose(ee_base_poses[0])
        self.tsdf_server.T_cam_task = Ttarget2cam_list[0]
        # self.tsdf_server.T_cam_task = xyzrpy_to_T(camera_poses[0].x, camera_poses[0].y, camera_poses[0].z, camera_poses[0].rx_deg, camera_poses[0].ry_deg, camera_poses[0].rz_deg)
        
        
        self.tsdf_server.catch_image = True
        rospy.sleep(0.1)
        cv2.imwrite('pose_0.jpg', self.tsdf_server.rgb_image)

        # TODO: need to check
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True
        
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.tsdf_server.depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('depth_colormap', depth_colormap)
        # cv2.imshow('depth_colormap', self.tsdf_server.depth_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # TODO: need to check   
        # for joint_target in self.scan_joints[1:]:
        #     self.pc.goto_joints(joint_target)
        i = 1
        for pose in ee_base_poses[1:]:
            self.pc.goto_pose(pose)
            self.tsdf_server.T_cam_task = Ttarget2cam_list[i]
            rospy.sleep(0.1)
            self.tsdf_server.integrate = True
            self.tsdf_server.catch_image = True
            cv2.imwrite(f'pose_{i}.jpg', self.tsdf_server.rgb_image)
            i += 1
        # TODO: need to check
        # self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()
        # pc = self.tsdf_server.low_res_tsdf.get_cloud()
        
        self.pc.home()
        print(f"TSDF:\n {np.any(tsdf.get_grid()), tsdf.get_grid().shape, tsdf.get_grid().min(), tsdf.get_grid().max()}")
        print(f"pc:\n {np.any(pc)}")

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

    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_base_pregrasp = T_base_grasp * T_grasp_pregrasp
        T_base_retreat = T_base_grasp * T_grasp_retreat

        self.pc.goto_pose(T_base_pregrasp * self.T_tcp_tool0, velocity_scaling=0.2)
        self.approach_grasp(T_base_grasp)

        if self.robot_error:
            return False

        self.pc.grasp(width=0.0, force=20.0)

        if self.robot_error:
            return False

        self.pc.goto_pose(T_base_retreat * self.T_tcp_tool0)

        # lift hand
        T_retreat_lift_base = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
        T_base_lift = T_retreat_lift_base * T_base_retreat
        self.pc.goto_pose(T_base_lift * self.T_tcp_tool0)

        if self.gripper_width > 0.004:
            return True
        else:
            return False

    def approach_grasp(self, T_base_grasp):
        self.pc.goto_pose(T_base_grasp * self.T_tcp_tool0)

    def drop(self):
        self.pc.goto_joints(
            [0.678, 0.097, 0.237, -1.63, -0.031, 1.756, 0.931], 0.2, 0.2
        )
        self.pc.move_gripper(0.08)


class TSDFServer(object):
    def __init__(self):
        # self.cam_frame_id = rospy.get_param("~cam/frame_id")
        # self.cam_topic_name = rospy.get_param("~cam/topic_name")
        self.cam_frame_id = "camera_depth_optical_frame"
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        
        # TODO: need check
        # self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("~cam/intrinsic"))
        
        width, height = 640, 480 # ??
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2] 
        self.intrinsic = CameraIntrinsic(width, height, fx, fy, cx, cy)
        # print(f"width, height, fx, fy, cx, cy: {(width, height, fx, fy, cx, cy)}")
        
        self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)
        
        # TODO: need check
        #self.size = 6.0 * rospy.get_param("~finger_depth")
        self.size = 0.3
        # self.size = 300 # mm
        
        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

        self.cam_rgb_topic_name = "/camera/color/image_raw"
        rospy.Subscriber(self.cam_rgb_topic_name, sensor_msgs.msg.Image, self.__rgb_callback)
        self.catch_image = False
        self.rgb_image = None
        self.depth_image = None

        self.T_cam_task = np.eye(4)

        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

        
    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return
        self.integrate = False
        img = cv2.rotate(self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001, cv2.ROTATE_180)
        # img = cv2.rotate(self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32), cv2.ROTATE_180)
        print(f"img min: {img.min()}")
        print(f"img max: {img.max()}")
        rospy.sleep(0.1)
        # print(f"Depth image: {img.shape}")
        self.depth_image = img.copy()
        # new_size = (480, 640) #(640, 480)
        # img = cv2.resize(self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001, new_size, interpolation=cv2.INTER_AREA)
        # T_cam_task = self.tf_tree.lookup(
        #     self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        # )
        
        self.tf_tree.broadcast_static(
            Transform(Rotation.from_matrix(self.T_cam_task[:3, :3]), self.T_cam_task[:3, 3]), 
            self.cam_frame_id, "task"
        )
        self.T_cam_task = np.linalg.inv(self.T_cam_task)
        # self.T_cam_task[:3, 3] * 100
        self.T_cam_task[:3, 3] /= 1000.0
        print(f"self.T_cam_task:\n {self.T_cam_task}")
        # self.T_cam_task = Transform(Rotation.from_quat([0.0091755 ,  0.9995211 ,  0.00176319 ,-0.02950025]), [ 0.16363484, -0.14483834 , 0.44753983]).as_matrix()

        self.low_res_tsdf.integrate(img, self.intrinsic, self.T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, self.T_cam_task)

    def __rgb_callback(self, msg):
        if self.catch_image is True:
            self.catch_image = False
            self.rgb_image = cv2.rotate(cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg).astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.ROTATE_180)
            rospy.loginfo("[Camera] Catch RGB image")
        else:
            return
            

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("---")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())

def main(args):
    rospy.init_node("panda_grasp")
    panda_grasp = PandaGraspController(args)

    while True:
        panda_grasp.run()
        break


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    parser = ArgumentParserForBlender()
    parser.add_argument("--model", type=Path, required=False, default='/catkin_ws/GraspNeRF/src/nr/ckpt/test/model_best.pth')
    args = parser.parse_args()
    main(args)
    rospy.loginfo("[End]")