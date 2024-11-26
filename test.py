import torch
import sb3_contrib
import stable_baselines3
import gym
from typing import Tuple,Deque,List
from collections import deque,defaultdict

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor,SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np
import math
import random
import re

print(torch.__version__)

# 导入你定义的 SnakeEnv 类
from game_env import SnakeEnv

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

def normalize_repeat_map(repeat_map):
    """将 repeat_map 归一化为概率形式"""
    total_count = sum(repeat_map.values())  # 总次数
    if total_count == 0:
        # 避免总和为 0 的情况
        return {key: 1 / len(repeat_map) for key in repeat_map}
    return {key: value / total_count for key, value in repeat_map.items()}


def calculate_reachable_space_from_tail(self, snake_head, snake_tail, snake_body, board_size):
    """
    计算从蛇尾开始的可达空间，避免蛇体封闭自己。
    """
    reachable_spaces = self.bfs_reachable_area(snake_tail, snake_body, board_size)
    return len(reachable_spaces) / (board_size ** 2)  # 可达空间比例，越大越好

def bfs_reachable_area(self, start, snake_body, board_size):
    """
    使用 BFS 从给定的起点（蛇尾）查找可达空间。
    start: 搜索的起点，即蛇尾位置
    """
    queue = [start]
    visited = set(snake_body)  # 蛇体位置不可访问
    reachable_spaces = set()

    while queue:
        position = queue.pop(0)
        if position in visited:
            continue
        visited.add(position)
        reachable_spaces.add(position)

        # 添加邻居位置
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_position = (position[0] + dx, position[1] + dy)
            if 0 <= new_position[0] < board_size and 0 <= new_position[1] < board_size:
                queue.append(new_position)

    return reachable_spaces
#---------
def calculate_reachable_space_reward(self, snake_head, snake_body, board_size):
    reachable_spaces = self.bfs_reachable_area(snake_head, snake_body, board_size)
    # 可达空间越大奖励越高
    return len(reachable_spaces) / (board_size ** 2)

def bfs_reachable_area(self, snake_head, snake_body, board_size):
    # 使用 BFS 查找蛇头位置周围的可达区域
    queue = [snake_head]
    visited = set(snake_body)
    reachable_spaces = set()

    while queue:
        position = queue.pop(0)
        if position in visited:
            continue
        visited.add(position)
        reachable_spaces.add(position)

        # 添加邻居位置
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_position = (position[0] + dx, position[1] + dy)
            if 0 <= new_position[0] < board_size and 0 <= new_position[1] < board_size:
                queue.append(new_position)

    return reachable_spaces

#print(total)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def list_to_tree_node(lst):
    if not lst:
        return None
    
    root = TreeNode(lst[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(lst):
        current = queue.popleft()  # 使用 deque.popleft()，更高效
        if current:
            # 添加左子节点
            if i < len(lst) and lst[i] is not None:
                current.left = TreeNode(lst[i])
                queue.append(current.left)
            i += 1
            
            # 添加右子节点
            if i < len(lst) and lst[i] is not None:
                current.right = TreeNode(lst[i])
                queue.append(current.right)
            i += 1
            
    return root

# 示例使用
lst = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
root = list_to_tree_node(lst)

import pandas as pd

# Column headers for the Excel file
headers = [
    "机构唯一号", "监管平台机构ID", "机构名称", "监管平台药品分类Code", "监管平台药品ID",
    "监管平台药品通用名", "医院药品ID", "医院药品通用名", "医院药品商品名", "医药药品别名",
    "医院药品包装规格", "医院药品产地名称", "医院药品单价", "医院药品有效标志", "上传时间"
]

# Create an empty DataFrame with the specified headers
df = pd.DataFrame(columns=headers)

# Save the DataFrame to an Excel file
file_path = "药品目录.xlsx"
df.to_excel(file_path, index=False)

print(f"Excel file saved as {file_path}")


























