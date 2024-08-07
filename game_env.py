import torch
import torch.nn as nn
import gym
from gym import spaces
import numpy as np
from snake_game import SnakeGame
from typing import Optional
import time
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SnakeEnv(gym.Env):
    def __init__(self, board_size=10, silent_mode=True, seed=0):
        super().__init__()
        self.game = SnakeGame(board_size=board_size, silent_mode=silent_mode, seed=seed, train_mode=True)
        self.action_space = spaces.Discrete(4)
        self.scale = max(1, (32 + board_size - 1) // board_size)  # 确保 board_size * scale >= 32
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, self.game.board_size * self.scale, self.game.board_size * self.scale), dtype=np.uint8)
        self.render_delay = 1.0 / 60  # 60 FPS
        self.last_render_time = 0

    def _get_state(self):
        state = np.zeros((4, self.game.board_size * self.scale, self.game.board_size * self.scale), dtype=np.uint8)  # 4个通道
        # 颜色值（使用0-255范围的整数表示颜色）
        snake_body_value = [0, 0, 255]  # 蓝色
        food_value = [0, 255, 0]  # 绿色
        snake_head_value = [255, 0, 0]  # 红色

        for x in range(self.game.board_size):
            for y in range(self.game.board_size):
                value = self.game.play_ground[x][y]
                if value == 1:  # 蛇身
                    for i in range(3):
                        state[i, x*self.scale:(x+1)*self.scale, y*self.scale:(y+1)*self.scale] = snake_body_value[i]
                elif value == 2:  # 食物
                    for i in range(3):
                        state[i, x*self.scale:(x+1)*self.scale, y*self.scale:(y+1)*self.scale] = food_value[i]
                elif value == 3:  # 蛇头
                    for i in range(3):
                        state[i, x*self.scale:(x+1)*self.scale, y*self.scale:(y+1)*self.scale] = snake_head_value[i]

        # 添加方向信息
        direction = self.game.directions.index(self.game.direction)
        state[3, :, :] = direction * 85  # 将方向索引映射到0-255范围

        # 对蛇身添加渐变颜色，模拟颜色渐变效果
        for index, (x, y, _) in enumerate(self.game.snake):
            gradient_value = int((index + 1) / len(self.game.snake) * 255)
            for i in range(3):
                state[i, x*self.scale:(x+1)*self.scale, y*self.scale:(y+1)*self.scale] *= gradient_value
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