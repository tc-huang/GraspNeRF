import os
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
# from gd.utils.panda_control import PandaCommander
from gd.grasp import *

from nr.main import GraspNeRFPlanner

import cv2

# from pysurroundcap.util import *
# from pysurroundcap.data_struct import LevelPosesConfig, CameraPose
# from pysurroundcap.pose import create_poses

from scipy import ndimage

from scipy.spatial.transform import Rotation

# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.42, 0.02, 0.21])

class PandaGraspController(object):
    def __init__(self, args, round_idx, gpuid, render_frame_list):
        self.robot_error = False
        self.args = args
        self.round_idx, self.gpuid, self.render_frame_list = round_idx, gpuid, render_frame_list

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
        
        # self.setup_panda_control()
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.create_planning_scene()
        self.tsdf_server = TSDFServer()
        # self.plan_grasps = VGN(args.model, rviz=True)
        
        self.grasp_plan_fn = GraspNeRFPlanner(args)

        self.images = []
        self.extrinsics = []
        self.intrinsics = []

        rospy.loginfo("Ready to take action")
    
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
    
    def run(self):
        vis.clear()
        vis.draw_workspace(self.size)
        
        # level_poses_list = [LevelPosesConfig(theta_deg=THETA, phi_begin_deg=0, phi_end_deg=360, r=R_, num_poses=N)]
        # camera_poses2= create_poses(level_poses_list, 0)
        # extrinsics = np.array([np.linalg.inv(self.__camera_pose_2_Ttarget2cam(camera_pose, rot_z_180=True)) for camera_pose in camera_poses2])
        # for e in extrinsics:
        #     e[:3, 3] /= 1000.0

        # intrinsics = np.array([intrinsic_matrix] * N)
        # depth_range=np.array([[0.2, 0.8] for _ in range(N)])
        # depth_range=np.array([[0.1, 0.6] for _ in range(N)])
        # print(f"images {images.shape}")
        # print(f"extrinsics {extrinsics.shape}")
        # print(f"intrinsics {intrinsics.shape}")
        # print(f"depth_range {depth_range.shape}")
        # extrinsics = np.load('/catkin_ws/GraspNeRF/extrinsics.npy')

        # images = np.load('/catkin_ws/GraspNeRF/sim_images.npy')
        extrinsics = np.load('/catkin_ws/GraspNeRF/sim_extrinsics.npy')
        intrinsics = np.load('/catkin_ws/GraspNeRF/sim_intrinsics.npy')
        depth_range = np.load('/catkin_ws/GraspNeRF/scripts/depth_range.npy')

        # for i, a in enumerate(images):
        #     img = cv2.resize(a.transpose((1, 2, 0)), (512, 288))
        #     cv2.imshow(f'sim{i}', img)
            
        bbox3d = np.array([[-0.15,   -0.15,   -0.0503],
                            [ 0.15,    0.15,    0.2497]])
        
        # bbox3d = np.array([[-0.15,   -0.15,   0.0],
        #                     [ 0.15,    0.15,    0.3]])
        
        # images = np.roll(images, shift=-1, axis=0)
        # extrinsics = np.roll(extrinsics, shift=-1, axis=0)
        
        import glob
        # data_path = '/catkin_ws/GraspNeRF/scripts/test_images_8'
        data_path = '/catkin_ws/GraspNeRF/scripts/pose'
        files = glob.glob(os.path.join(data_path, '*'))
        files.sort()
        print("\nFiles using glob module:")
        images = []
        for i, file in enumerate(files):
            print(file)
            img = cv2.imread(file)
            img = cv2.resize(img, (512, 288))
            cv2.imshow(f'{i}', img)
            images.append(img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        images = np.array(images).transpose((0, 3, 1, 2))
        images = images / 255.0
        
        print(f"images: {images.shape}")
        print(f"extrinsicsn:\n{extrinsics[0]}")
        print(f"intrinsics:\n{intrinsics[0]}")
        print(f"depth_range\n{depth_range}")
        print(f"bbox3d\n{bbox3d}")
        

        n = 6
        print(f"images shape {images.shape}")
        tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori, toc = self.grasp_plan_fn.core(images[:n], extrinsics[:n], intrinsics[:n], depth_range=depth_range[:n], bbox3d=bbox3d[:n])
        rospy.loginfo(f"[TSDF] draw TSDF2")
        vis.draw_tsdf(tsdf_vol, 0.3 / 40)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

class TSDFServer(object):
    def __init__(self):
        # self.cam_frame_id = rospy.get_param("~cam/frame_id")
        # self.cam_topic_name = rospy.get_param("~cam/topic_name")
        self.cam_frame_id = "camera_depth_optical_frame"
        self.cam_topic_name = "/camera/depth/image_rect_raw"
        
        # TODO: need check
        # self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("~cam/intrinsic"))
        
        # width, height = 640, 480 # ??
        # fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        # cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2] 
        # self.intrinsic = CameraIntrinsic(width, height, fx, fy, cx, cy)
        # print(f"width, height, fx, fy, cx, cy: {(width, height, fx, fy, cx, cy)}")
        
        # self.intrinsic = CameraIntrinsic(640, 480, 606.77737126, 606.70030146, 321.63287183, 236.95293136)
        
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
        rospy.sleep(0.1)
        self.depth_image = img.copy()
        # T_cam_task = self.tf_tree.lookup(
        #     self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        # )
        
        self.tf_tree.broadcast_static(
            Transform(Rotation.from_matrix(self.T_cam_task[:3, :3]), self.T_cam_task[:3, 3]), 
            self.cam_frame_id, "task"
        )
        self.T_cam_task = np.linalg.inv(self.T_cam_task)
        self.T_cam_task[:3, 3] /= 1000.0
        # print(f"self.T_cam_task:\n {self.T_cam_task}")

        self.low_res_tsdf.integrate(img, self.intrinsic, self.T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, self.T_cam_task)

    def __rgb_callback(self, msg):
        if self.catch_image is True:
            self.catch_image = False
            # self.rgb_image = cv2.rotate(cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg).astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.ROTATE_180)
            new_size = (512, 288)
            self.rgb_image = cv2.resize(cv2.rotate(cv2.cvtColor(self.cv_bridge.imgmsg_to_cv2(msg).astype(np.uint8), cv2.COLOR_BGR2RGB), cv2.ROTATE_180), new_size, interpolation=cv2.INTER_AREA)
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

def main(args, round_idx, gpuid, render_frame_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    rospy.init_node("panda_grasp")
    
    panda_grasp = PandaGraspController(args, round_idx, gpuid, render_frame_list)

    while True:
        panda_grasp.run()
        break


if __name__ == "__main__":
    """
    # parser = argparse.ArgumentParser()
    parser = ArgumentParserForBlender()
    parser.add_argument("--model", type=Path, required=False, default='/catkin_ws/GraspNeRF/src/nr/ckpt/test/model_best.pth')
    args = parser.parse_args()
    main(args)
    """

    argv = sys.argv
    print(f"argv:\n {argv}")
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    round_idx = int(argv[0])
    gpuid = int(argv[1])
    expname = str(argv[2])
    scene = str(argv[3])
    object_set = str(argv[4])
    check_seen_scene = bool(int(argv[5]))
    material_type = str(argv[6])
    blender_asset_dir = str(argv[7])
    log_root_dir = str(argv[8])
    use_gt_tsdf = bool(int(argv[9]))
    render_frame_list=[int(frame_id) for frame_id in str(argv[10]).replace(' ','').split(",")]
    method = str(argv[11])
    print("########## Simulation Start ##########")
    print("Round %d\nmethod: %s\nmaterial_type: %s\nviews: %s "%(round_idx, method, material_type, str(render_frame_list)))
    print("######################################")

    parser = ArgumentParserForBlender() ### argparse.ArgumentParser()
    parser.add_argument("---model", type=Path, default="")
    parser.add_argument("---logdir", type=Path, default=expname)
    parser.add_argument("---description", type=str, default="")
    parser.add_argument("---scene", type=str, choices=["pile", "packed", "single"], default=scene)
    parser.add_argument("---object-set", type=str, default=object_set)
    parser.add_argument("---num-objects", type=int, default=5)
    parser.add_argument("---num-rounds", type=int, default=200)
    parser.add_argument("---seed", type=int, default=42)
    parser.add_argument("---sim-gui", type=bool, default=True)#False)
    parser.add_argument("---rviz", action="store_true")
    
    ###
    parser.add_argument("---renderer_root_dir", type=str, default=blender_asset_dir)
    parser.add_argument("---log_root_dir", type=str, default=log_root_dir)
    parser.add_argument("---obj_texture_image_root_path", type=str, default=blender_asset_dir+"/imagenet") #TODO   
    parser.add_argument("---cfg_fn", type=str, default="src/nr/configs/nrvgn_sdf.yaml")
    parser.add_argument('---database_name', type=str, default='vgn_syn/train/packed/packed_170-220/032cd891d9be4a16be5ea4be9f7eca2b/w_0.8', help='<dataset_name>/<scene_name>/<scene_setting>')

    parser.add_argument("---gen_scene_descriptor", type=bool, default=False)
    parser.add_argument("---load_scene_descriptor", type=bool, default=True)
    parser.add_argument("---material_type", type=str, default=material_type)
    parser.add_argument("---method", type=str, default=method)

    # pybullet camera parameter
    parser.add_argument("---camera_focal", type=float, default=446.31) #TODO 

    ###
    args = parser.parse_args()
    main(args, round_idx, gpuid, render_frame_list)

    rospy.loginfo("[End]")