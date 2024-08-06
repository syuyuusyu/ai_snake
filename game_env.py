import gym
from gym import spaces
import numpy as np
from snake_game import SnakeGame
from typing import Optional
import time

class SnakeEnv(gym.Env):
    def __init__(self, board_size=10, silent_mode=True, seed=0):
        super().__init__()
        self.game = SnakeGame(board_size=board_size, silent_mode=silent_mode, seed=seed, train_mode=True)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=6, shape=(board_size * board_size,), dtype=np.int32)
        self.render_delay = 1.0 / 60  # 60 FPS
        self.last_render_time = 0   

    def _get_state(self):
        state = np.copy(self.game.play_ground).T[::-1].flatten().astype(np.int32)
        return state
    
    def reset(self):
        self.game.reset()
        state = self._get_state()
        return state
    
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
        observation = self._get_state()
        info = {
            'snake_length' : len(self.game.snake),
            'step_count' : self.game.step_count,
            'game_loop': self.game.game_loop,
            'step_state': state_dic[state]
        }
        return observation, reward, terminated, info
    
    def render(self, mode='human', **kwargs):
        current_time = time.time()
        if current_time - self.last_render_time >= self.render_delay:
            self.game.draw()
            self.last_render_time = current_time
    
    def close(self):
        self.game.close()

    def mask_fn(self):
        # env 是 DummyVecEnv 包装后的环境，不能直接访问 env.game
        # 使用 env.get_attr 来获取原始环境中的 game 属性
        game = self.game
        mask = [1] * self.action_space.n
        if game.direction == 'left':
            mask[1] = 0  # 禁止向右移动
        elif game.direction == 'right':
            mask[0] = 0  # 禁止向左移动
        elif game.direction == 'up':
            mask[3] = 0  # 禁止向下移动
        elif game.direction == 'down':
            mask[2] = 0  # 禁止向上移动
        return mask