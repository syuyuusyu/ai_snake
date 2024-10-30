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
import re

print(torch.__version__)

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def normalize_repeat_map(repeat_map):
    """将 repeat_map 归一化为概率形式"""
    total_count = sum(repeat_map.values())  # 总次数
    if total_count == 0:
        # 避免总和为 0 的情况
        return {key: 1 / len(repeat_map) for key in repeat_map}
    return {key: value / total_count for key, value in repeat_map.items()}


repeat_count  = normalize_repeat_map({
    (11, 11):797,
    (11, 0):25,
    (10, 9):100,
    (11, 6):93,
    (10, 8):166,
    (11, 1):26,
    (10, 7):58,
    (0, 10):16,
    (11, 4):13,
    (11, 3):28,
    (0, 11):19,
    (3, 11):9,
    (11, 5):32
})

print(repeat_count)

total = 0
for v in repeat_count.values():
    total = total +v 

print(total)























