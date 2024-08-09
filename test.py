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

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

print(device)

def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed,board_size=12, silent_mode=False)
        env = ActionMasker(env, SnakeEnv.mask_fn)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init


# 创建环境并使用 DummyVecEnv 包装环境
env = DummyVecEnv([make_env(22)])


# 创建并训练模型
# model = MaskablePPO("MlpPolicy", env, verbose=1, device=device)
# model.learn(total_timesteps=10)
board_size = 16
scale = max(1, (32 + board_size - 1) // board_size)  # 确保 board_size * scale >= 32
print(scale)
