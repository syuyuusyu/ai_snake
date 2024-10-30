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
from collections import defaultdict

from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'



repeat_map = defaultdict(int)

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
        self.att = ['beast_snake_length','back_forward_count','hit_wall_count','collide_self_count','repeat_count','victory_count','cuttent_snake_length']
    def _on_rollout_start(self) -> None:
        train_info_list = self.training_env.env_method('get_train_info')

        # 初始化计数器
        average_info = {
            'beast_snake_length': 0.0,
            'back_forward_count': 0.0,
            'hit_wall_count': 0.0,
            'collide_self_count': 0.0,
            'repeat_count': 0.0,
            'victory_count': 0.0,
            'rollout_snake_length': 0.0,
        }

        # 环境实例的数量
        num_envs = len(train_info_list)

        # 累加每个环境实例的值
        for train_info in train_info_list:
            for key in average_info.keys():
                average_info[key] += train_info[key]
            dic = train_info['repeat_map']
            for k,v in dic.items():
                repeat_map[k] = repeat_map[k]+v
                #print(repeat_map)
                
                

        # 计算平均值
        for key in average_info.keys():
            average_info[key] /= num_envs

        # 打印结果
        for key, value in average_info.items():
            print(f'Average {key}: {value}')
        self.training_env.env_method('reset_rollout')

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
        gamma=0.7,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef = 0.1,
        tensorboard_log="logs/"
    )
    #checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_snake')
    monitor_callback = MonitorCallback() 
    model.learn(total_timesteps=5e7,callback=[monitor_callback])
    model.save('pth/ppo_snake_early')
    env.close()

def load():
    seed_set = set()
    while len(seed_set) < 32:
        seed_set.add(random.randint(0,1e7))
    env = DummyVecEnv([make_env(seed,board_size) for seed in seed_set])
    #lr_schedule = schedule_fn(5e-4, 2.5e-6)
    lr_schedule = schedule_fn(5e-5, 1e-6)
    clip_range_schedule = schedule_fn(0.150, 0.025)
    model = MaskablePPO.load("pth/final_13.zip", env=env, device=device)
    model.gamma=0.97
    model.learning_rate = lr_schedule
    model.clip_range = clip_range_schedule
    model.ent_coef = 0
    info_callback = MonitorCallback() 
    model.learn(total_timesteps=1e6,callback=[info_callback])
    model.save('pth/final_tt')
    env.close()

if __name__ == '__main__':
    load()

    for k,v in repeat_map.items():
        print(f'{k}:{v}')