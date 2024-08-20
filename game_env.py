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
        self.step_count = 0

        self.beast_snake_length = 0
        self.back_forward_count = 0
        self.hit_wall_count = 0
        self.collide_self_count = 0
        self.repeat_count = 0
        self.victory_count = 0
        self.is_new_rollout = False


    def _get_obs(self):
        return self.game.get_obs()
    
    def reset(self):
        self.beast_snake_length = max(self.beast_snake_length,len(self.game.snake))
        self.game.reset()
        obs = self._get_obs()
        self.step_count = 0
        return obs
    
    def step(self, action):
        if self.is_new_rollout:
            self.back_forward_count = 0
            self.hit_wall_count = 0
            self.collide_self_count = 0
            self.repeat_count = 0
            self.victory_count = 0
            self.is_new_rollout = False

        p_action =  self.game.directions.index(self.game.direction)
        if (p_action == 0 and action == 1) or (p_action == 1 and action == 0) or (p_action == 2 and action == 3) or (p_action == 3 and action == 2):
            self.back_forward_count += 1
        self.step_count += 1
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
    
        if self.step_count == self.max_snake_length * 100:
            #without eat food in step_count
            self.repeat_count += 1
            reward = -2 * self.repeat_count
            reward = reward * 0.1
            return observation, reward, True, info
        if state ==5:
            reward = 100
            self.victory_count += 1
            return observation, reward, True, info
        if state == 2 or state == 3:
            if state == 2:
                self.hit_wall_count += 1
            if state == 3:
                self.collide_self_count += 1
            reward = -math.pow(self.max_growth, (self.max_snake_length - snake_length) / self.max_growth)
            reward = reward * 0.1
            return observation, reward, terminated, info
        if state == 0:
            reward = - 1 / snake_length
        elif state == 1:
            reward = 1 / snake_length
        elif self == 4:
            self.step_count = 0
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
        #directions = ['up','down','left','right']
        arr = ['down','up','right','left']
        mask[arr.index(game.direction)] = 0
        return mask
        #return np.array([1, 1, 1, 1], dtype=np.uint8)
