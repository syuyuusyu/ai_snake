import gym
from gym import spaces
import numpy as np
from snake_game import SnakeGame
from typing import Optional,Tuple
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
        self.repeat_prossibility = np.full((self.game.board_size,self.game.board_size),0,dtype=np.float16)
        self.rollout_snake_length = len(self.game.snake)
    
    def get_train_info(self):
        return {
            'beast_snake_length': self.beast_snake_length,
            'back_forward_count': self.back_forward_count,
            'hit_wall_count': self.hit_wall_count,
            'collide_self_count': self.collide_self_count,
            'repeat_count': self.repeat_count,
            'victory_count': self.victory_count,
            'rollout_snake_length': self.rollout_snake_length,
        }
    
    def reset_rollout(self):
        self.is_new_rollout = True
        
    def _get_obs(self):
        return self.game.get_obs()
    
    def reset(self):
        self.rollout_snake_length = len(self.game.snake)
        self.beast_snake_length = max(self.beast_snake_length,len(self.game.snake))
        self.game.reset()
        obs = self._get_obs()
        self.step_count = 0
        return obs
    
    def calculate_penalty_factor(self,x, y, board_size):
        # 棋盘中心点
        cx, cy = board_size // 2, board_size // 2
        # 计算欧几里得距离
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # 计算最大可能距离（从中心到角落）
        max_distance = np.sqrt((cx) ** 2 + (cy) ** 2)
        # 归一化距离 (得到的值在0到1之间，中心为0，边缘为1)
        normalized_distance = distance / max_distance
        # 惩罚调整系数
        penalty_factor = 1 + normalized_distance  # 中心为1，边缘最大为2
        return penalty_factor
    
    def is_on_edge(self,point:Tuple[int,int])->bool:
        x, y = point
        max_index = self.game.board_size -1
        return x == 0 or x == max_index or y == 0 or y == max_index

    def step(self, action):
        if self.is_new_rollout:
            self.back_forward_count = 0
            self.hit_wall_count = 0
            self.collide_self_count = 0
            self.repeat_count = 0
            self.victory_count = 0
            self.repeat_prossibility = np.full((self.game.board_size,self.game.board_size),0,dtype=np.float16)
            self.is_new_rollout = False

        p_action =  self.game.directions.index(self.game.direction)
        self.step_count += 1
        self.game.direction = self.game.directions[action]
        terminated,state = self.game.step()
        self.rollout_snake_length = len(self.game.snake)

        snake_length = len(self.game.snake)
        observation = self._get_obs()
        info = {
            'snake_length' : snake_length,
            'step_count' : self.game.step_count,
            'game_loop': self.game.game_loop,
            'step_state': SnakeEnv.state_dic[state]
        }
        reward = 0.0

        repeat_rate = 4
        repeat_peanlity = 0
        if self.step_count >= self.max_snake_length:
            x,y = self.game.snake[0]
            penalty_factor = self.calculate_penalty_factor(x, y, self.game.board_size)
            self.repeat_prossibility[x][y] = self.repeat_prossibility[x][y] - 0.001* penalty_factor
            repeat_peanlity = self.repeat_prossibility[x][y]

        if self.step_count == self.max_snake_length * repeat_rate :
            #without eat food in step_count
            self.repeat_count += 1
            #reward = -math.pow(self.max_growth, (self.max_snake_length - snake_length) / self.max_growth)
            #reward = reward * 0.1
            terminated = True

        if (p_action == 0 and action == 1) or (p_action == 1 and action == 0) or (p_action == 2 and action == 3) or (p_action == 3 and action == 2):
            self.back_forward_count += 1

        if state == 2 or state == 3:
            if state == 2:
                self.hit_wall_count += 1
            if state == 3:
                self.collide_self_count += 1

        if terminated:
            reward = -math.pow(self.max_growth, (self.max_snake_length - snake_length) / self.max_growth)
            reward = reward * 0.1
            return observation, reward, terminated, info
        
        if state ==5:
            reward = 100
            self.victory_count += 1
            return observation, reward, True, info

        if state == 0:
            reward = reward - 1 / snake_length
        elif state == 1:
            reward = reward + 1.01 / snake_length
        elif state == 4:
            # coefficient = 10
            # if self.is_on_edge(self.game.snake[0]):
            #     coefficient = 20
            #     print('eat on edge')
            coefficient = 20 if self.is_on_edge(self.game.snake[0]) else 10
            self.step_count = 0
            reward = reward + coefficient * (snake_length / self.max_snake_length)
        if self.is_on_edge(self.game.snake[0]):
            reward += 0.1 #step on edge
        reward = reward * 0.1
        #print(reward,repeat_peanlity)
        return observation, reward+repeat_peanlity, terminated, info
    
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
        #return mask
        return np.array([1, 1, 1, 1], dtype=np.uint8)
