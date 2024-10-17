import torch
import sb3_contrib
import stable_baselines3
import gym
from typing import Tuple,Deque,List
from collections import deque

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import math
import random

print(torch.__version__)

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'



a = [1,2,3,4,5,6,7]

# 3 -> 1
# 4 -> 1,2
# 5 -> 2
# 6 -> 2,3
# 7 -> 3
# 8 -> 3,4


a = [[0]*10]*10
print(a)

