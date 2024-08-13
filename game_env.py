import gym
from gym import spaces
import numpy as np
from snake_game import SnakeGame
from typing import Optional
import math

class SnakeEnv(gym.Env):
    state_dic = {
        0: 'the head leave the food',
        1: 'this head approch the food',
        2: 'hit wall',
        3: 'collied self',
        4: 'eat food',
        5: 'Victory'
    }
    def __init__(self, board_size=10, silent_mode=True, seed=0):
        super().__init__()
        self.game = SnakeGame(board_size=board_size, silent_mode=silent_mode, seed=seed, train_mode=True)
        self.action_space = spaces.Discrete(4)
        shape_size = self.game.board_size * self.game.scale+2*self.game.scale
        self.observation_space = spaces.Box(low=0, high=255, shape= (3,shape_size, shape_size), dtype=np.uint8)
        self.max_snake_length = board_size ** 2
        self.max_growth = self.max_snake_length - len(self.game.snake)
        self.eat_count = 0

    def _get_obs(self):
        return self.game.get_obs()
    
    def reset(self):
        self.game.reset()
        obs = self._get_obs()
        self.eat_count = 0
        return obs
    
    def step(self, action):
        self.eat_count += 1
        self.game.direction = self.game.directions[action]
        terminated,state = self.game.step()

        snake_length = len(self.game.snake)
        observation = self._get_obs()
        info = {
            'snake_length' : snake_length,
            'step_count' : self.game.step_count,
            'game_loop': self.game.game_loop,
            'step_state': SnakeEnv.state_dic[state]
        }
        reward = 0.0
        if self.eat_count == self.max_snake_length:
            print(f'without eat in {self.eat_count} receive penalty')
            reward = -10 / snake_length
            reward = reward * 0.1
            return observation, reward, terminated, info
        if state == 2 or state == 3:
            reward = -math.pow(self.max_growth, (self.max_snake_length - snake_length) / self.max_growth)
            reward = reward * 0.1
            return observation, reward, terminated, info
        if state == 0:
            reward = - 0.5 / snake_length
        elif state == 1:
            reward = 1 / snake_length
        elif self == 4:
            self.eat_count = 0
            reward = snake_length / self.max_snake_length
        reward += 0.1 #one step reward
        reward = reward * 0.1
        return observation, reward, terminated, info
    
    def render(self, mode='human', **kwargs):
        self.game.draw()
    
    def close(self):
        self.game.close()

    def mask_fn(self):
        game = self.game
        mask = [1] * self.action_space.n
        if game.direction == 'left':
            mask[3] = 0  # 禁止向右移动
        elif game.direction == 'right':
            mask[2] = 0  # 禁止向左移动
        elif game.direction == 'up':
            mask[1] = 0  # 禁止向下移动
        elif game.direction == 'down':
            mask[0] = 0  # 禁止向上移动
        return mask
