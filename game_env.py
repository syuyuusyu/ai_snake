import math
from collections import defaultdict
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from snake_game import SnakeGame


class SnakeEnv(gym.Env):
    state_dic : ClassVar[dict[int, str]] = {
        0: 'the head leave the food',
        1: 'this head approch the food',
        2: 'hit wall',
        3: 'collied self',
        4: 'eat food',
        5: 'Victory'
    }
    def __init__(self, board_size=10, silent_mode=True, seed=0,bfs_intensity=0, reward_gamma=0.995):
        super().__init__()
        print(f'SnakeEnv {bfs_intensity}')
        self.game = SnakeGame(board_size=board_size, silent_mode=silent_mode, seed=seed, train_mode=True,bfs_intensity=bfs_intensity)
        self.action_space = spaces.Discrete(4)
        shape_size = self.game.board_size * self.game.scale+2*self.game.scale
        self.observation_space = spaces.Box(low=0, high=255, shape= (3,shape_size, shape_size), dtype=np.uint8)
        self.max_snake_length = board_size ** 2
        self.reward_gamma = reward_gamma
        self.safety_reward_weight = 0.5
        self._safety_score = self.game.safety_score()
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
    
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rollout_snake_length = len(self.game.snake)
        self.beast_snake_length = max(self.beast_snake_length,len(self.game.snake))
        self.game.reset()
        obs = self._get_obs()
        self._safety_score = self.game.safety_score()
        self.step_count = 0
        return obs, {}
    
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
    
    def is_on_edge(self,point:tuple[int,int])->bool:
        x, y = point
        max_index = self.game.board_size -1
        return x == 0 or x == max_index or y == 0 or y == max_index
    
    def is_on_right_and_down(self,point:tuple[int,int])->bool:
        x,y = point
        max_index = self.game.board_size -1
        return x == max_index or y == max_index
    
    def is_on_right(self,point:tuple[int,int])->bool:
        x,_ = point
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
    
    def reachable_space_reward(self, threshold_ratio=0.5):
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
        reachable_spaces = self.game._bfs_reachable_area
        
        # 计算奖励：可达空间比例，值越大奖励越高
        reward = len(reachable_spaces) / board_area
        #print(snake_length,snake_length_threshold ,snake_length <= snake_length_threshold,reward)
        return reward
    
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
        truncated = False
        self.step_count += 1
        safety_before = self._safety_score
        self.game.direction = self.game.directions[action]
        terminated,state = self.game.step()
        self.rollout_snake_length = len(self.game.snake)

        snake_length = len(self.game.snake)
        observation = self._get_obs()
        info = {
            'snake_length' : snake_length,
            'step_count' : self.step_count,
            'game_loop': self.game.game_loop,
            'step_state': SnakeEnv.state_dic[state],
            'repeat_map': self.repeat_map
        }
        repeat_rate = 4

        if state == 4:
            self.step_count = 0
        elif not terminated and self.step_count >= self.max_snake_length * repeat_rate:
            self.repeat_count += 1
            truncated = True

        if (p_action == 0 and action == 1) or (p_action == 1 and action == 0) or (p_action == 2 and action == 3) or (p_action == 3 and action == 2):
            self.back_forward_count += 1

        if state == 2 or state == 3:
            if state == 2:
                self.hit_wall_count += 1
            if state == 3:
                self.collide_self_count += 1

        reward_food = 0.0
        reward_death = 0.0
        reward_victory = 0.0
        reward_timeout = 0.0
        reward_step = 0.0

        progress = snake_length / self.max_snake_length
        if state == 5:
            reward_victory = 10.0
            self.victory_count += 1
        elif terminated:
            reward_death = -(1.0 + 4.0 * progress ** 2)
        elif truncated:
            reward_timeout = -1.0
        elif state == 4:
            reward_food = 1.0
        else:
            reward_step = -0.002

        safety_after = 0.0 if terminated else self.game.safety_score()
        self._safety_score = safety_after
        safety_reward = self.safety_reward_weight * (
            self.reward_gamma * safety_after - safety_before
        )
        task_reward = (
            reward_food
            + reward_death
            + reward_victory
            + reward_timeout
            + reward_step
        )
        reward = task_reward + safety_reward
        info.update({
            'reward_food': reward_food,
            'reward_death': reward_death,
            'reward_victory': reward_victory,
            'reward_timeout': reward_timeout,
            'reward_step': reward_step,
            'reward_safety': safety_reward,
            'task_reward': task_reward,
            'reward_total': reward,
            'safety_before': safety_before,
            'safety_after': safety_after,
        })
        return observation, reward, terminated, truncated, info
    
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
        return np.asarray(mask, dtype=np.int8)
