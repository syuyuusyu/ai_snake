import argparse
import random

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 导入你定义的 SnakeEnv 类
from board_config import DEFAULT_BOARD_HIGH, DEFAULT_BOARD_WIDTH, add_board_arguments
from game_env import SnakeEnv
from jev_assist import safe_mask_fn
from snake_game import SnakeGame

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

bfs_intensity=1

def make_env(seed=0, board_width=DEFAULT_BOARD_WIDTH, board_high=DEFAULT_BOARD_HIGH):
    def _init():
        env = SnakeEnv(seed=seed,board_width=board_width, board_high=board_high, silent_mode=True,bfs_intensity=bfs_intensity)
        env = ActionMasker(env, safe_mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init
def main():
    parser = argparse.ArgumentParser()
    add_board_arguments(parser)
    parser.add_argument("--model", default="pth/stable_5.zip")
    args = parser.parse_args()
    board_width, board_high = args.board_width, args.board_high
    seed_set = set()
    while len(seed_set) < 32:
        seed_set.add(random.randint(0, 10_000_000))
    random_seed = random.randint(0, 10_000_000)
    print(random_seed)
    env = DummyVecEnv([make_env(random_seed, board_width, board_high)])

    # Old checkpoints may contain gym-era serialized objects; override them at load time.
    model = MaskablePPO.load(
        args.model,
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

    try:
        game = SnakeGame(board_width=board_width, board_high=board_high, silent_mode=False,
                         train_mode=True, model=model, seed=random_seed, bfs_intensity=bfs_intensity)
        game.run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
