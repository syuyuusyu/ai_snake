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
import random

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

class MonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MonitorCallback, self).__init__(verbose)
        self.att = ['beast_snake_length','back_forward_count','hit_wall_count','collide_self_count','repeat_count','victory_count']
    def _on_rollout_start(self) -> None:
        for name in self.att:
            values = self.training_env.get_attr(name)
            mean_value = np.mean(values)
            print(f'mean_{name}: {mean_value}')
        self.training_env.set_attr('is_new_rollout',True)
    
    def _on_step(self) -> bool:
        return True

def schedule_fn(initial_value, final_value=0.0, schedule_type='linear'):

    def scheduler(progress):
        progress = min(max(progress, 0.0), 1.0)
        if schedule_type == 'linear':
            return final_value + progress * (initial_value - final_value)
        elif schedule_type == 'exponential':
            return initial_value * (final_value / initial_value) ** progress
        else:
            raise ValueError("Unsupported schedule type")

    return scheduler


board_size = 12
def main():
    seed_set = set()
    while len(seed_set) < 32:
        seed_set.add(random.randint(0,1e5))
    env = DummyVecEnv([make_env(seed,board_size) for seed in seed_set])
    lr_schedule = schedule_fn(5e-4, 2.5e-6)
    clip_range_schedule = schedule_fn(0.150, 0.025) 
    model = MaskablePPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=1,
        n_steps=2048,
        batch_size=512*8,
        n_epochs=4,
        gamma=0.8,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef = 0.1,
        tensorboard_log="logs/"
    )
    #checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_snake')
    monitor_callback = MonitorCallback() 
    model.learn(total_timesteps=1e8,callback=[monitor_callback])
    model.save('pth/ppo_snake_early')
    env.close()

def load():
    seed_set = set()
    while len(seed_set) < 32:
        seed_set.add(random.randint(0,1e7))
    env = DummyVecEnv([make_env(seed,board_size) for seed in seed_set])
    lr_schedule = schedule_fn(5e-4, 2.5e-6)
    clip_range_schedule = schedule_fn(0.150, 0.025)
    model = MaskablePPO.load("pth/ppo_snake_early.zip", env=env, device=device)
    model.gamma=0.94
    model.learning_rate = lr_schedule
    model.clip_range = clip_range_schedule
    model.ent_coef = 0.00
    info_callback = MonitorCallback() 
    model.learn(total_timesteps=1e8,callback=[info_callback])
    model.save('pth/ppo_snake_early')
    env.close()

if __name__ == '__main__':
    load()