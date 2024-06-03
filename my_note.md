# Env
1. Dowload blender-2.93.3-linux-x64
2. remane /home/rl/Documents/JH/newGraspNeRF/GraspNeRF/blender-2.93.3-linux-x64/2.93/python -> _python
3. sudo ln -s /home/rl/Documents/JH/newGraspNeRF/GraspNeRF/.conda /home/rl/Documents/JH/newGraspNeRF/GraspNeRF/blender-2.93.3-linux-x64/2.93/python

conda install numpy==1.23.5 pandas scikit-image==0.19.3 h5py torchmetrics tqdm yaml
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install open3d pybullet==2.7.9
conda install conda-forge::easydict
conda install conda-forge::plyfile
conda install conda-forge::pyquaternion
conda install conda-forge::tensorboard
conda install -c conda-forge quaternion

??pip install opencv-python transforms3d
conda activate /home/rl/Documents/JH/newGraspNeRF/GraspNeRF/.conda

# build python blender bpy
https://developer.blender.org/docs/handbook/building_blender/linux/#__tabbed_1_2


# Testing
bash run_simgrasp.sh

# Real Grasp
python scripts/yaskawa_grasp.py --model src/nr/ckpt/test/model_best.pth

/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/blender-2.93.3-linux-x64/blender ./data/assets/material_lib_graspnet-v2.blend --background --python scripts/yaskawa_grasp.py

# Testing planner
bash run_testgrasp.sh

# Known errors
Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown
> sudo apt-get remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 
> sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker
sudo docker compose -f docker/docker-compose-gui-nvidia.yml build
sudo docker compose -f docker/docker-compose-gui-nvidia.yml up
sudo docker exec -it ros_docker /bin/bash
roscore
rosrun rviz rviz -d config/my_grasp.rviz
sudo chmod 777 /dev/ttyUSB0


# Start Realsense ROS node
roslaunch realsense2_camera rs_camera.launch depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=360 color_fps:=30 filters:=pointcloud

## VGN
roslaunch realsense2_camera rs_camera.launch depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 enable_pointcloud:=true publish_tf:=false align_depth:=true
# Knowning Errors
Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown
> sudo apt-get remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 
> sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 

## 2F-85
sudo docker exec -it ros_docker /bin/bash
git clone https://github.com/ros-industrial/robotiq.git
export PYTHON_EXECUTABLE=/opt/conda/envs/myenv/bin/python
pip install empy==3.3.4 pymodbus==2.5.3
rosdep update
rosdep install --from-paths src --ignore-src -r -y
rm -rf build devel
rm -rf src/robotiq_modbus_rtu
rm -rf src/robotiq_2f_gripper_control
cp -r /catkin_ws/GraspNeRF/robotiq_2f_gripper_control /catkin_ws/src
cp -r /catkin_ws/GraspNeRF/robotiq_modbus_rtu /catkin_ws/src
catkin_make
source devel/setup.bash
chmod +x /catkin_ws/src/robotiq_2f_gripper_control/nodes/Robotiq2FGripperRtuNode.py
chmod +x /catkin_ws/src/robotiq_2f_gripper_control/nodes/Robotiq2FGripperSimpleController.py
chmod +x /catkin_ws/src/robotiq_2f_gripper_control/nodes/Robotiq2FGripperStatusListener.py
chmod +x /catkin_ws/src/robotiq_2f_gripper_control/nodes/Robotiq2FGripperTcpNode.py

source devel/setup.bash
rosrun robotiq_2f_gripper_control nodes/Robotiq2FGripperRtuNode.py /dev/ttyUSB0
source devel/setup.bash
rosrun robotiq_2f_gripper_control Robotiq2FGripperSimpleController.py