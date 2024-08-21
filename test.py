import torch
import sb3_contrib
import stable_baselines3
import gym
from typing import Tuple,Deque
from collections import deque

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import math

print(torch.__version__)

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

a = -math.pow(140, (100) /140)

print(a)

repeat_prossibity = np.full((12,12),0,dtype=np.int8)
repeat_rate = 8
max_step = 12*12
step_count = 0
repeat_count = 0
if step_count >= max_step * repeat_rate/2:
    #start repeat,record repeat_prossibity
    pass
if step_count >= max_step * repeat_rate:
    #calculate panlaty accroding to repeat_prossibity
    pass