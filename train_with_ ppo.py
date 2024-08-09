import torch
import sb3_contrib
import stable_baselines3
import gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback
import numpy as np

from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'




def make_env(seed=0,board_size=10):
    def _init():
        env = SnakeEnv(seed=seed,board_size=board_size, silent_mode=True)
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

def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


board_size = 12
def main(render):
    env = DummyVecEnv([make_env(22)])
    model = MaskablePPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=2048,
        batch_size=512*4,
        n_epochs=4,
        gamma=0.94,
        learning_rate=0.0003,
        clip_range=0.2,
        ent_coef = 0.01,
        tensorboard_log="logs/"
    )
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_snake')
    render_callback = RenderCallback() if render else None
    model.learn(total_timesteps=100000,callback=[checkpoint_callback,])
    model.save('pth/ppo_snake_early')
    env.close()

def load(render):
    env = DummyVecEnv([make_env(22)])
    model = MaskablePPO.load("pth/ppo_snake_early.zip", env=env, device=device)
    model.gamma=0.94
    model.learning_rate = 0.0003
    model.ent_coef = 0.01
    render_callback = RenderCallback() if render else None
    model.learn(total_timesteps=100000)
    model.save('pth/ppo_snake_early')
    env.close()

if __name__ == '__main__':
    main(False)