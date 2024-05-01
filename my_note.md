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
conda activate /home/rl/Documents/JH/newGraspNeRF/GraspNeRF/.conda
# Testing
bash run_simgrasp.sh
