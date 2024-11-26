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
           (11, 11):16841754,
            (0, 11):2345112,
            (11, 5):2290995,
            (10, 11):2246846,
            (11, 0):2190087,
            (11, 6):1939690,
            (1, 11):1807822,
            (4, 11):1571729,
            (11, 1):1561247,
            (9, 11):1529389,
            (3, 11):1525072,
            (2, 11):1409112,
            (11, 4):811222,
            (5, 11):798017,
            (8, 11):602816,
            (11, 3):449968,
            (11, 7):323815,
            (0, 10):300496,
            (11, 2):279206,
            (7, 11):236105,
            (6, 11):181205,
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
    
    def reachable_space_reward(self, threshold_ratio=0.6):
        """
        计算可达空间奖励，用于避免蛇体自我封闭。
        
        参数:
        - threshold_ratio: float, 控制蛇体占用空间比例的阈值，当蛇体长度超过该比例时启用奖励

        返回:
        - reward: float, 根据蛇体周围的可达空间大小计算的奖励值
        """
        # 计算棋盘总面积和蛇体长度
        board_area = self.max_snake_length
        snake_length = len(self.game.snake)
        
        # 计算蛇体长度的临界值，当蛇体长度超过该值时启用可达空间奖励
        snake_length_threshold = int(board_area * threshold_ratio)
        
        # 判断蛇体长度是否超过临界值
        if snake_length <= snake_length_threshold:
            # 如果未达到临界值，返回0奖励
            return 0.0

        # 从蛇尾开始计算可达空间
        reachable_spaces = self.bfs_reachable_area(
            start=self.game.snake[-1],  # 从蛇尾开始
            snake_body=self.game.snake,  # 蛇体位置
            board_size=self.game.board_size
        )
        
        # 计算奖励：可达空间比例，值越大奖励越高
        reward = len(reachable_spaces) / board_area
        return reward

    def bfs_reachable_area(self, start, snake_body, board_size):
        """
        使用 BFS 从给定的起点（蛇尾）查找可达空间。
        
        参数:
        - start: tuple, BFS 搜索的起点（蛇尾位置）
        - snake_body: list, 包含蛇体的位置，防止访问
        - board_size: int, 棋盘的大小

        返回:
        - reachable_spaces: set, 可达空间的集合
        """
        queue = [start]
        visited = set(snake_body) -{start} # 蛇体位置不可访问
        reachable_spaces = set()

        while queue:
            position = queue.pop(0)
            if position in visited:
                continue
            visited.add(position)
            reachable_spaces.add(position)

            # 添加邻居位置并检查边界
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_position = (position[0] + dx, position[1] + dy)
                if 0 <= new_position[0] < board_size and 0 <= new_position[1] < board_size:
                    queue.append(new_position)
        return reachable_spaces
    
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
        reward = self.reachable_space_reward()

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
            
            #self.repeat_map[self.game.food] = self.repeat_map[self.game.food]+1
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
        if state == 0 and snake_length<=30:
            reward = reward - 1 / snake_length
        elif state == 1 and snake_length<=30:
            reward = reward + 1 / snake_length
        elif state == 4:
            repeat_adjust = 0
            # if len(self.game.snake[0]) <10:
            #     repeat_probability = self.repeat_point_count.get(self.game.snake[0], 0)
            #     if repeat_probability > 0:
            #         repeat_adjust = self.calculate_coefficient(repeat_probability)
            coefficient = 1
            self.step_count = 0
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
