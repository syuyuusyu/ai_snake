import torch
import sb3_contrib
import stable_baselines3
import gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed,board_size=12, silent_mode=True)
        env = ActionMasker(env, SnakeEnv.mask_fn)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.training_env.render()
        return True


# 创建环境并使用 DummyVecEnv 包装环境
env = DummyVecEnv([make_env(22)])
model = MaskablePPO("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=2)

def main(render):
    env = DummyVecEnv([make_env(22)])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=2048,
        batch_size=512*2,
        n_epochs=4,
        gamma=0.94,
        learning_rate=0.0003,
        clip_range=0.2,
        tensorboard_log="logs/"
    )
    render_callback = RenderCallback() if render else None
    model.learn(total_timesteps=100000,callback=render_callback)
    env.close()

if __name__ == '__main__':
    main(False)