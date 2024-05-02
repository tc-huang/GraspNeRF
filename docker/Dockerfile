##############
# Base image #
##############
FROM ros:noetic-robot as base

ARG CATKIN_WORKSPACE_DIR="/catkin_ws"

LABEL org.opencontainers.image.authors="tobit.flatscher@outlook.com"
LABEL description="ROS Noetic Docker template"
LABEL version="1.0"

WORKDIR ${CATKIN_WORKSPACE_DIR}
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get -y install \
    git \
    python3-catkin-tools \
    python3-osrf-pycommon \
 && rm -rf /var/lib/apt/lists/*

# Add additional installation instructions here...

ENV DEBIAN_FRONTEND=dialog


#####################
# Development image #
#####################
FROM base as dev

ARG CATKIN_WORKSPACE_DIR="/catkin_ws"
ARG USERNAME="developer"
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y \
    ack \
    cloc \
    gdb \
    htop \
    iperf3 \
    iputils-ping \
    mlocate \
    net-tools \
    psmisc \
    python3-vcstool \
    tmux \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y \
    ros-${ROS_DISTRO}-rviz \
 && rm -rf /var/lib/apt/lists/*

# Install additional developer tools here...

RUN apt-get update \
 && apt-get install -y sudo \
 && rm -rf /var/lib/apt/lists/* \
 && addgroup --gid ${GID} ${USERNAME} \
 && adduser --disabled-password --gecos '' --uid ${GID} --gid ${GID} ${USERNAME} \
 && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
 && chown -R ${UID}:${GID} /home/${USERNAME} \
 && chown -R ${UID}:${GID} ${CATKIN_WORKSPACE_DIR}
 
ENV DEBIAN_FRONTEND=dialog

RUN echo "alias rbuild='(cd ${CATKIN_WORKSPACE_DIR} && catkin build)'" >> /home/${USERNAME}/.bash_aliases \
 && echo "alias rclean='(cd ${CATKIN_WORKSPACE_DIR} && catkin clean -y)'" >> /home/${USERNAME}/.bash_aliases \
 && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc

RUN apt-get update \
 && apt-get install -y \
   libxcb-cursor0 \
   ros-${ROS_DISTRO}-librealsense2* \
   ros-${ROS_DISTRO}-realsense2-*
#    python3-pip \
#    python3-venv \

 

# COPY requirements2.txt ${AMENT_WORKSPACE_DIR}/requirements2.txt
# RUN pip install -r ${AMENT_WORKSPACE_DIR}/requirements.txt

# ENV PATH="/usr/bin/python:${PATH}"

# RUN which python3
# RUN pip3 install -r ${AMENT_WORKSPACE_DIR}/requirements2.txt
# RUN pip3 install \
#     numpy==1.23 \
#     pandas \
#     scikit-image==0.19.3 \
#     h5py \
#     tqdm \
#     pyyaml \
#     open3d \
#     pybullet==2.7.9 \
#     networkx==2.8.8
# RUN pip3 install \
#     easydict \
#     plyfile \
#     pyquaternion \
#     tensorboard \
#     quaternion
# RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
# RUN pip3 install torchmetrics
# RUN sudo ln -s /usr/bin/python3 /usr/bin/python
# WORKDIR ${AMENT_WORKSPACE_DIR}
COPY docker/blender-2.93.3 blender-2.93.3
# RUN rm -rf /blender-2.93.3-linux-x64/2.93/python
# RUN sudo ln -s /usr/bin/python3 /blender-2.93.3-linux-x64/2.93/python
# ENTRYPOINT ["python3", "-c", "import tqdm; print(tqdm.__version__)"]
# https://segmentfault.com/a/1190000023358320

 
# RUN sudo python3 -m venv /venv
# RUN . /venv/bin/activate && \
#     pip3 install \
#     numpy==1.23 \
#     pandas
    # scikit-image==0.19.3 \
    # h5py \
    # tqdm \
    # pyyaml \
    # open3d \
    # pybullet==2.7.9 \
    # networkx==2.8.8
# RUN sudo ln -s /ven /catkin_ws/blender-2.93.3/2.93/python
# RUN sudo ln -s /usr/bin/python3 /catkin_ws/blender-2.93.3/2.93/python
# ln -s /ven /catkin_ws/blender-2.93.3/2.93/python
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
# 下载 Miniconda 并安装
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda init bash
# 创建一个新的 Conda 环境
RUN conda create -n myenv python=3.9
ENV PATH /opt/conda/envs/myenv/bin:$PATH

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN conda install numpy==1.23.5 pandas scikit-image==0.19.3 h5py torchmetrics tqdm yaml
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip3 install open3d pybullet==2.7.9
RUN conda install -c conda-forge quaternion
RUN conda install conda-forge::easydict conda-forge::plyfile conda-forge::tensorboard conda-forge::quaternion
# RUN conda uninstall opencv
# RUN pip3 install opencv-python
# RUN conda install -c conda-forge libgcc=5.2.0
# RUN sudo apt-get install -y cmake gcc g++ python3-dev python3-numpy libgtk-3-dev libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev
RUN pip3 install opencv-python
#==3.4.18.65
# RUN conda install -c conda-forge libgcc=5.2.0
# RUN conda install -c anaconda libstdcxx-ng
# RUN conda install gcc scipy
RUN sudo ln -s /opt/conda/envs/myenv /catkin_ws/blender-2.93.3/2.93/python
# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc

RUN sudo cp /opt/conda/envs/myenv/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
RUN sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
RUN sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6

RUN sudo echo "deb http://th.archive.ubuntu.com/ubuntu jammy main" >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y libc6

# ENV LD_LIBRARY_PATH /opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH
USER ${USERNAME}
