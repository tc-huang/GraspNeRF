# from panda_robot import PandaArm
import rospy
import numpy as np
from gd.utils.transform import Transform
from scipy.spatial.transform import Rotation

from robotic_drawing.control.robot_arm.yaskawa.mh5l import MH5L
from robotic_drawing.control.tool.robotiq.gripper_2f85 import Gripper2F85

FAKE = False

HOST = "192.168.255.10"
PORT = 11000
SPEED = 200

INIT_JOINT_CONFIGURE = [
    -126330,
    -18970,
    14748,
    -1553,
    -101441,
    98749
]

PLACE_JOINT_CONFIGURE = [
    -52956,
    21581,
    -627,
    967,
    -65817,
    70995
]

DROP_JOINT_CONFIGURE = [
    -65871,
    43163,
    29053,
    511,
    -74122,
    76055
]

class PandaCommander(object):
    def __init__(self):
        if not FAKE:
            # self.r = PandaArm()
            # self.r.enable_robot()
            # rospy.loginfo("PandaCommander ready")
            self.robot_arm = MH5L(host=HOST, port=PORT)
            rospy.loginfo("[Robot] Init")
            self.robot_arm.power_on()
            rospy.loginfo("[Robot] Power on")
            self.robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
            rospy.loginfo("[Robot] Move to initial pose by joint configure")
            
            rospy.loginfo("[Gripper] Init")
            self.gripper = Gripper2F85()
            rospy.loginfo("[Gripper] Connect")
            success = self.gripper.connect()
            rospy.loginfo("[Gripper] Reset")
            success = self.gripper.reset()
            
            self.moving = False
        
    def __del__(self):
        if not FAKE:
            self.robot_arm.power_off() 
            rospy.loginfo("[Robot] Power off")
        
    def reset(self):
        if not FAKE:
            rospy.logwarn("reset and go home!")
            # self.r.enable_robot()
            self.home()
        
    def clear(self):
        if not FAKE:
            # self.r.exit_control_mode()
            raise NotImplementedError

    def home(self):
        if not FAKE:
            self.moving = True
            # self.r.move_to_neutral()
            self.robot_arm.move_to_joint_config(INIT_JOINT_CONFIGURE, SPEED)
            rospy.loginfo("[Robot] Move to initial pose by joint configure")
            self.moving = False
            rospy.loginfo("PandaCommander: Arrived home!")
    
    def drop(self):
        if not FAKE:
            self.moving = True
            # self.r.move_to_neutral()
            self.robot_arm.move_to_joint_config(DROP_JOINT_CONFIGURE, SPEED)
            rospy.loginfo("[Robot] Move to drop position by joint configure")
            self.gripper.on()
            self.moving = False
            rospy.loginfo("PandaCommander: Arrived drop position")

    def goto_joints(self, joints:list):
        if not FAKE:
            self.moving = True
            # self.r.move_to_joint_position(joints)
            self.robot_arm.move_to_joint_config(joints, SPEED)
            self.moving = False

    def get_joints(self):
        if not FAKE:
            # return self.r.angles()
            raise NotImplementedError
    
    def goto_pose(self, pose:list):
        if not FAKE:
            rospy.loginfo(f"PandaCommander: goto pose {pose}")
            self.moving = True
            # x, y, z, w = pose.rotation.as_quat()
            # self.r.move_to_cartesian_pose(pose.translation.astype(np.float32), np.quaternion(w, x, y, z))
            Tx, Ty, Tz, Rx, Ry, Rz = pose 
            self.robot_arm.move_to_pose(Tx, Ty, Tz, Rx, Ry, Rz, SPEED)
            # TODO: need implement
            # safe = self.r.in_safe_state()
            # if self.r.has_collided():
            #     rospy.logwarn("collided!")
            #     self.r.enable_robot()
            #     return
            # error = self.r.error_in_current_state()
            # if error or not safe:
            #     rospy.logwarn("error or not safe! reset and run again.")
            #     self.reset()
            #     self.goto_pose(pose)
            #     return 
            self.moving = False

    def get_pose(self):
        if not FAKE:
            tcp_pose = self.robot_arm.controller.get_robot_pose(tool_num=0) 
            return tcp_pose
            # pos, rot = self.r.ee_pose()
            # return Transform(Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]), pos)
    
    def grasp(self, width=0.0, force=10.0):
        if not FAKE:
            pass
            success = self.gripper.soft_close()
            rospy.sleep(3)
            # return self.r.exec_gripper_cmd(width, force=force)
    

    def move_gripper(self, width):
        if not FAKE:
            raise NotImplementedError
            # return self.r.exec_gripper_cmd(width)

    def get_gripper_width(self):
        if not FAKE:
            raise NotImplementedError
            # return np.sum(self.r.gripper_state()['position'])
