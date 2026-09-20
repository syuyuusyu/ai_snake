import random

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv
from snake_game import SnakeGame

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

board_size = 12
bfs_intensity=1

def make_env(seed=0,board_size=12):
    def _init():
        env = SnakeEnv(seed=seed,board_size=board_size, silent_mode=True,bfs_intensity=bfs_intensity)
        env = ActionMasker(env, SnakeEnv.mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init
seed_set = set()
while len(seed_set) < 32:
    seed_set.add(random.randint(0,1e7))
random_seed = random.randint(0,1e7)
print(random_seed)
env = DummyVecEnv([make_env(random_seed,board_size)])

# Old checkpoints may contain gym-era serialized objects; override them at load time.
model = MaskablePPO.load(
    'pth/stable_4.zip',
    env=env,
    device=device,
    custom_objects={
        'observation_space': env.observation_space,
        'action_space': env.action_space,
        'learning_rate': 0.0,
        'lr_schedule': lambda _: 0.0,
        'clip_range': lambda _: 0.0,
    },
)

#print(model.observation_space.shape)

if __name__ == "__main__":  
    game = SnakeGame(board_size=board_size,silent_mode=False,train_mode=True,model=model,seed=random_seed,bfs_intensity=bfs_intensity)
    game.run()