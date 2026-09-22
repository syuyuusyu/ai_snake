import random
from collections import defaultdict

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'




repeat_map = defaultdict(int)
bfs_intensity = 1
discount_gamma = 0.995
gae_lambda = 0.97

def make_env(seed=0,board_size=12):
    def _init():
        env = SnakeEnv(
            seed=seed,
            board_size=board_size,
            silent_mode=True,
            bfs_intensity=bfs_intensity,
            reward_gamma=discount_gamma,
        )
        env = ActionMasker(env, SnakeEnv.mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

class MonitorCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
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
            for key in average_info:
                average_info[key] += train_info[key]
            # dic = train_info['repeat_map']
            # for k,v in dic.items():
            #     repeat_map[k] = repeat_map[k]+v
                #print(repeat_map)
                
        # 计算平均值
        for key in average_info:
            average_info[key] /= num_envs

        # 打印结果
        for key, value in average_info.items():
            print(f'Average {key}: {value}')
        self.training_env.env_method('reset_rollout')

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        metric_keys = [
            'reward_food',
            'reward_death',
            'reward_victory',
            'reward_timeout',
            'reward_step',
            'reward_safety',
            'reward_total',
            'safety_before',
            'safety_after',
            'snake_length',
        ]
        for key in metric_keys:
            values = [info[key] for info in infos if key in info]
            if values:
                self.logger.record_mean(f'env/{key}', float(np.mean(values)))
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
        gamma=discount_gamma,
        gae_lambda=gae_lambda,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef=0.001,
        tensorboard_log="logs/"
    )
    #checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_snake')
    monitor_callback = MonitorCallback() 
    model.learn(total_timesteps=5e7,callback=[monitor_callback])
    model.save('pth/ppo_snake_early')
    env.close()

def load():
    seed_set = set()
    while len(seed_set) < 128:
        seed_set.add(random.randint(0,1e7))
    env = SubprocVecEnv([make_env(seed,board_size) for seed in seed_set])
    #lr_schedule = schedule_fn(5e-4, 2.5e-6)
    lr_schedule = schedule_fn(2e-5, 1e-6)
    clip_range_schedule = schedule_fn(0.150, 0.030)
    model = MaskablePPO.load(
        "pth/stable_4.zip",
        env=env,
        device=device,
        custom_objects={
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'learning_rate': lr_schedule,
            'lr_schedule': lr_schedule,
            'clip_range': clip_range_schedule,
        },
    )
    model.gamma = discount_gamma
    model.gae_lambda = gae_lambda
    model.learning_rate = lr_schedule
    model.lr_schedule = lr_schedule
    model.clip_range = clip_range_schedule
    model.ent_coef = 0.001
    #model.n_steps = 2048
    model.batch_size = 512 * 8
    info_callback = MonitorCallback() 
    model.learn(total_timesteps=1e7,callback=[info_callback])
    model.save('pth/stable_5')
    env.close()

if __name__ == '__main__':
    load()

    for k,v in repeat_map.items():
        print(f'{k}:{v},')
