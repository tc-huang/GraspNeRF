#!/bin/bash

GPUID=0
BLENDER_BIN=/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/blender-2.93.3-linux-x64/blender
BLENDER_BIN=/catkin_ws/GraspNeRF/blender-2.93.3-linux-x64/blender
# BLENDER_BIN=/blender-2.93.3-linux-x64/blender

RENDERER_ASSET_DIR=./data/assets
BLENDER_PROJ_PATH=./data/assets/material_lib_graspnet-v2.blend
SIM_LOG_DIR="./log/`date '+%Y%m%d-%H%M%S'`"

scene="pile"
object_set="pile_subdiv"
material_type="specular_and_transparent"
render_frame_list="2,6,10,14,18,22"
check_seen_scene=0
expname=0

NUM_TRIALS=1
# NUM_TRIALS=3
METHOD='graspnerf'

# !!!
# eval "$(conda shell.bash hook)"
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate /home/rl/Documents/JH/newGraspNeRF/GraspNeRF/.conda
# python --version
# python -c "import numpy; print(numpy.__version__)"
# !!!

mycount=0 
while (( $mycount < $NUM_TRIALS )); do
   # $BLENDER_BIN $BLENDER_PROJ_PATH --python scripts/sim_grasp.py \
   # -- $mycount $GPUID $expname $scene $object_set $check_seen_scene $material_type \
   # $RENDERER_ASSET_DIR $SIM_LOG_DIR 0 $render_frame_list $METHOD

   $BLENDER_BIN $BLENDER_PROJ_PATH --background --python scripts/sim_grasp.py \
   -- $mycount $GPUID $expname $scene $object_set $check_seen_scene $material_type \
   $RENDERER_ASSET_DIR $SIM_LOG_DIR 0 $render_frame_list $METHOD

   # python ./scripts/stat_expresult.py -- $SIM_LOG_DIR $expname #!!!
((mycount=$mycount+1));
done;

# python ./scripts/stat_expresult.py -- $SIM_LOG_DIR $expname #!!!