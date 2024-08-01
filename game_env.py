import gymnasium
from gymnasium import spaces
import numpy as np
from snake_game import SnakeGame
from typing import Optional

class SnakeEnv(gymnasium.Env):
    def __init__(self,board_size = 10,silent_mode = True,seed = 0):
        super(SnakeEnv, self).__init__()
        self.game = SnakeGame(board_size=board_size,silent_mode=silent_mode,seed=seed,train_mode=True)
        # 定义动作空间和观察空间
        self.action_space = spaces.Discrete(4)  # 假设有4个动作：上、下、左、右
        self.observation_space = spaces.Box(low=0, high=6, shape=(board_size*board_size,), dtype=np.int32)  # 假设游戏板是10x10的
    
    def _get_state(self):
        # 先转置，再翻转行，然后展平为1D向量，并转换为int32类型
        return np.copy(self.game.play_ground).T[::-1].flatten().astype(np.int32)
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # 处理随机种子
        print(seed,'-----')
        if seed is not None:
            np.random.seed(seed)
        # 重置游戏状态
        self.game.reset(seed=seed)
        return self._get_state(),{}
    
    def step(self, action):
        self.game.direction = self.game.directions[action]
        terminated,reward,state = self.game.step()
        self.game.update_play_ground()
        state_dic = {
            0: 'the head leave the food',
            1: 'this head approch the food',
            2: 'hit wall',
            3: 'collied self',
            4: 'eat food'
        }
        truncated = False
        observation = self._get_state()
        info = {
            'snake_length' : len(self.game.snake),
            'step_count' : self.game.step_count,
            'game_loop': self.game.game_loop,
            'step_state': state_dic[state]
        }
        return observation, reward, terminated,truncated, info
    
    def render(self, mode='human'):
        self.game.draw()
    
    def close(self):
        # 可选的方法，用于释放资源
        pass