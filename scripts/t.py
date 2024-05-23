import sys
sys.path.append('src')

from pysurroundcap.util import *
from pysurroundcap.data_struct import LevelPosesConfig, CameraPose
from pysurroundcap.pose import create_poses

if __name__ == "__main__":
    level_poses_list = [LevelPosesConfig(theta_deg=30, phi_begin_deg=0, phi_end_deg=360, r=500, num_poses=6)]
    camera_poses= create_poses(level_poses_list, 0)
    camera_poses = camera_poses[1:] + camera_poses[0:1]
    
    for p in camera_poses:
        print(p)