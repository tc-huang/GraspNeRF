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
rosrun rviz rviz