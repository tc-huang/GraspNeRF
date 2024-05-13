import argparse
import sys

sys.path.append('src')
sys.path.append('src/nr')
sys.path.append('src/gd')
from train.trainer import Trainer
from utils.base_utils import load_cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='/home/rl/Documents/JH/newGraspNeRF/GraspNeRF/src/nr/configs/nrvgn_sdf.yaml')
flags = parser.parse_args()

trainer = Trainer(load_cfg(flags.cfg))
trainer.run()
print('END')