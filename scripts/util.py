# Copyright 2024 tc-haung
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union


def get_minimum_r(workspace_size_mm: Union[int, float]) -> Union[int, float]:
    d = np.sqrt((workspace_size_mm / 2) ** (2 + workspace_size_mm) ** 2)
    d = np.sqrt(d**2 + workspace_size_mm**2)
    return d

def rvec_to_rot_mat(rvec: np.ndarray, use_degree = True) -> np.ndarray:
    r = R.from_rotvec(rvec, degrees=use_degree)
    return r.as_matrix()

def tvec_rvec_to_T(tvec: np.ndarray, rvec: np.ndarray, use_degree = True) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rvec_to_rot_mat(rvec, use_degree)
    T[:3, 3] = tvec
    return T

def image_horizontal_flip(image: np.ndarray) -> np.ndarray:
    image = np.flip(image, axis=0)
    image = np.flip(image, axis=1)
    return image

def read_saved_matrix(file_path: str, matrix_keys:list[str]) -> list[np.ndarray]:
    results = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for key in matrix_keys:
            results.append(np.array(data[key]))
    return results

def aruco_camera_pose_to_ee_base_pose(aruco_pose: np.ndarray, T_cam2gripper: np.ndarray) -> np.ndarray:
    aruco_pose = aruco_pose.reshape(4, 4)
    T_cam2gripper = T_cam2gripper.reshape(4, 4)
    return aruco_pose

def xyzrpy_to_T(x, y, z, rx, ry, rz):
    T = np.eye(4)
    T[:3, 3] = np.array([x, y, z])
    # T[:3, :3] = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    T[:3, :3] = (R.from_euler("z", rz, degrees=True)
        * R.from_euler("y", ry, degrees=True)
        * R.from_euler("x", rx, degrees=True)).as_matrix()
    return T

def T_to_xyzrpy(T):
    x, y, z = T[0][3], T[1][3], T[2][3]
    # rx, ry, rz = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    rz, ry, rx = R.from_matrix(T[:3, :3]).as_euler('zyx', degrees=True)
    return x, y, z, rx, ry, rz

def rpy_to_rvec(rx, ry, rz, use_degree):
    if use_degree:
        # return R.from_euler(
        #         "xyz", [rx, ry, rz], degrees=True
        #     ).as_rotvec(degrees=True)
        return (R.from_euler("z", rz, degrees=True)
        * R.from_euler("y", ry, degrees=True)
        * R.from_euler("x", rx, degrees=True)).as_rotvec(degrees=True)
    else:
        return (R.from_euler("z", rz, degrees=False)
        * R.from_euler("y", ry, degrees=False)
        * R.from_euler("x", rx, degrees=False)).as_rotvec(degrees=False)