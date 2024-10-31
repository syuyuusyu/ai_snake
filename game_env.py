import gym
from gym import spaces
import numpy as np
from snake_game import SnakeGame
from typing import Optional,Tuple
import math
from collections import defaultdict

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
        self.repeat_map = defaultdict(int)
        self.repeat_point_count  = self.normalize_repeat_map({
(11, 11):31382136,
(0, 11):4080000,
(10, 8):3035921,
(11, 5):4070558,
(11, 6):3431652,
(10, 7):2766976,
(7, 11):189349,
(10, 9):3040044,
(3, 11):1862715,
(0, 10):439287,
(11, 0):4844609,
(4, 11):1724243,
(5, 11):1180403,
(11, 1):1948050,
(11, 4):1264811,
(5, 9):7252,
(1, 11):2881681,
(2, 11):1905028,
(9, 3):3884,
(0, 8):12784,
(6, 11):96992,
(11, 3):508202,
(6, 2):4871,
(11, 2):286611,
(11, 10):499628,
(10, 11):2497577,
(2, 10):7514,
(11, 9):348938,
(11, 8):401854,
(3, 0):11970,
(11, 7):377543,
(0, 9):6875,
(2, 5):6070,
(8, 10):6192,
(10, 6):7119,
(8, 11):649745,
(8, 5):5049,
(9, 10):15647,
(7, 4):6122,
(9, 6):8588,
(9, 11):1562445,
(6, 8):7650,
(4, 4):7596,
(4, 0):11447,
(7, 3):6954,
(9, 2):9491,
(3, 2):5635,
(2, 6):4107,
(8, 4):7859,
(10, 5):8132,
(3, 7):4214,
(8, 6):5512,
(0, 5):7040,
(4, 6):3598,
(7, 7):4961,
(7, 10):10305,
(0, 2):5324,
(2, 0):5290,
(2, 1):3714,
(9, 5):8838,
(8, 9):6662,
(7, 2):6214,
(3, 1):3809,
(5, 6):4612,
(7, 5):3795,
(1, 0):3668,
(3, 8):5806,
(5, 8):2688,
(3, 9):5370,
(0, 6):5377,
(9, 4):4035,
(7, 1):4722,
(0, 3):5062,
(5, 7):3146,
(8, 8):4789,
(2, 7):7224,
(2, 8):3670,
(1, 3):3444,
(6, 4):5868,
(0, 4):6322,
(10, 0):2236,
(9, 0):3731,
(5, 2):2200,
(4, 8):7420,
(8, 0):5535,
(0, 7):4579,
(5, 5):6606,
(6, 5):2470,
(10, 1):3176,
(10, 10):5238,
(9, 7):6648,
(1, 4):4777,
(4, 3):1898,
(8, 3):1864,
(0, 1):1845,
(6, 7):2353,
(1, 2):3366,
(1, 5):1811,
(10, 2):2538,
(0, 0):2822,
(1, 10):1801,
(3, 3):3588,
(6, 1):1751,
(7, 8):2366,
(5, 3):2029,
(1, 9):4222,
(6, 10):3404,
(1, 8):1656,
(4, 10):1550,
(1, 6):2879,
(1, 1):2535,
(4, 7):1535,
(5, 4):1386,
(3, 10):3334,
(4, 9):1259,
(6, 6):2345,
(8, 2):2282,
(2, 2):1577,
(5, 0):1056,
(9, 1):925,
(10, 3):808,
(1, 7):1457,
(10, 4):712,
(7, 0):946,
(6, 3):572,
(4, 1):555,
(2, 4):785,
(2, 9):382,
(4, 2):336,
(9, 9):325,
(4, 5):227,
(7, 6):131
        })
    
    def get_train_info(self):
        return {
            'beast_snake_length': self.beast_snake_length,
            'back_forward_count': self.back_forward_count,
            'hit_wall_count': self.hit_wall_count,
            'collide_self_count': self.collide_self_count,
            'repeat_count': self.repeat_count,
            'victory_count': self.victory_count,
            'rollout_snake_length': self.rollout_snake_length,
            'repeat_map':self.repeat_map
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
    
    def is_on_right_and_down(self,point:Tuple[int,int])->bool:
        x,y = point
        max_index = self.game.board_size -1
        return x == max_index or y == max_index
    
    def is_on_right(self,point:Tuple[int,int])->bool:
        x,y = point
        max_index = self.game.board_size -1
        return x == max_index
    
    def normalize_repeat_map(self,repeat_map):
        """将 repeat_map 归一化为概率形式"""
        total_count = sum(repeat_map.values())  # 总次数
        if total_count == 0:
            # 避免总和为 0 的情况
            return {key: 1 / len(repeat_map) for key in repeat_map}
        return {key: value / total_count for key, value in repeat_map.items()}
    def process_probability_with_log(self,probability):
        """使用对数平滑处理概率"""
        return math.log(1 + probability)
    def process_probability_with_sqrt(self,probability):
        """使用平方根平滑处理概率"""
        return math.sqrt(probability)    
    def calculate_coefficient(self,probability, method='log'):
        if probability == 0:
            return 0
        """根据不同的方法计算奖励系数"""
        if method == 'log':
            value = self.process_probability_with_log(probability)
        elif method == 'sqrt':
            value = self.process_probability_with_sqrt(probability)
        else:
            value = probability  # 不做处理
    
        return value  # 最终的奖励系数

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
            'step_state': SnakeEnv.state_dic[state],
            'repeat_map': self.repeat_map
        }
        reward = 0.0

        repeat_rate = 4
        repeat_peanlity = 0



        # if self.step_count >= self.max_snake_length:
        #     x,y = self.game.snake[0]
        #     penalty_factor = self.calculate_penalty_factor(x, y, self.game.board_size)
        #     self.repeat_prossibility[x][y] = self.repeat_prossibility[x][y] - 0.001* penalty_factor
        #     repeat_peanlity = self.repeat_prossibility[x][y]

        if self.step_count % self.max_snake_length * repeat_rate ==0 :
            #without eat food in step_count
            self.repeat_count += 1
            
            self.repeat_map[self.game.food] = self.repeat_map[self.game.food]+1
            #print(f'repeat:{self.rollout_snake_length} {self.game.food} {self.repeat_map}')
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

        # Remove step-based rewards for longer snake
        # if state == 0:
        #     reward = reward - 1 / snake_length
        # elif state == 1:
        #     reward = reward + 1 / snake_length
        elif state == 4:
            repeat_probability = self.repeat_point_count.get(self.game.snake[0], 0)
            repeat_adjust = 0
            if repeat_probability > 0:
                repeat_adjust = self.calculate_coefficient(repeat_probability)
            coefficient = 1
            self.step_count = 0
            # if repeat_adjust > 0:
            #     print(self.game.snake[0],repeat_adjust)
            reward = reward + coefficient * (snake_length / self.max_snake_length) + repeat_adjust
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
