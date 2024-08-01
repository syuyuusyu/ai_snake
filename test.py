import torch
import sb3_contrib
import stable_baselines3
import gymnasium

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

print(sb3_contrib.__version__)
print(stable_baselines3.__version__)
print(gymnasium.__version__)
# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv




device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def mask_fn(env: SnakeEnv):
    # 根据游戏状态返回合法动作的掩码
    # 这里是一个例子，你需要根据你的环境定义掩码逻辑
    mask = [1] * env.action_space.n  # 默认为所有动作都可用
    if env.game.direction == 'left':
        mask[env.action_space.index('right')] = 0  # 禁止向相反方向移动
    elif env.game.direction == 'right':
        mask[env.action_space.index('left')] = 0
    elif env.game.direction == 'up':
        mask[env.action_space.index('down')] = 0
    elif env.game.direction == 'down':
        mask[env.action_space.index('up')] = 0
    return mask

def make_env():
    return SnakeEnv(seed=23, board_size=12, silent_mode=False)

# 使用 DummyVecEnv 包装环境
env = SubprocVecEnv([make_env])
#env = ActionMasker(env, mask_fn)
check_env(env, warn=True, skip_render_check=True)

# 创建并训练模型
model = MaskablePPO("MlpPolicy", env, verbose=1, device=device)