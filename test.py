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


repeat_count = {
     (11, 11):16841754,
(9, 11):1529389,
(11, 1):1561247,
(11, 0):2190087,
(1, 11):1807822,
(10, 11):2246846,
(11, 3):449968,
(0, 11):2345112,
(10, 9):84519,
(11, 4):811222,
(2, 11):1409112,
(11, 6):1939690,
(11, 5):2290995,
(3, 11):1525072,
(5, 11):798017,
(6, 11):181205,
(7, 11):236105,
(8, 11):602816,
(11, 7):323815,
(11, 8):125972,
(0, 10):300496,
(4, 11):1571729,
(1, 3):3599,
(11, 2):279206,
(11, 9):117026,
(5, 7):4474,
(6, 10):7945,
(4, 7):7724,
(9, 0):2229,
(9, 2):4154,
(4, 4):2226,
(6, 5):4408,
(11, 10):139027,
(8, 0):5897,
(9, 4):2194,
(7, 9):2997,
(10, 5):5267,
(5, 10):3126,
(10, 2):3991,
(4, 3):3707,
(1, 4):3780,
(4, 8):2141,
(1, 0):3433,
(9, 3):3002,
(10, 1):3297,
(10, 3):2870,
(3, 8):3896,
(0, 8):5702,
(7, 8):3709,
(2, 3):3249,
(8, 4):2796,
(1, 8):6890,
(6, 6):1938,
(4, 0):2720,
(10, 7):7366,
(8, 5):2023,
(2, 9):3544,
(10, 8):4792,
(7, 7):3265,
(0, 6):3577,
(3, 7):4611,
(8, 2):1841,
(3, 2):3893,
(6, 4):2644,
(4, 9):5091,
(6, 7):2515,
(0, 9):1751,
(2, 0):3530,
(5, 4):2392,
(8, 3):1685,
(5, 5):2191,
(3, 0):5561,
(6, 0):1622,
(6, 9):2976,
(2, 7):3574,
(5, 0):2015,
(9, 5):1577,
(1, 10):2021,
(2, 2):2917,
(3, 9):1534,
(6, 8):2385,
(8, 6):1498,
(6, 2):3650,
(1, 6):2927,
(3, 3):1585,
(1, 7):2841,
(2, 10):3664,
(0, 5):2008,
(9, 10):4198,
(8, 8):1400,
(4, 10):3550,
(1, 9):1359,
(8, 1):1481,
(5, 8):2607,
(5, 2):1277,
(0, 2):1239,
(5, 6):2261,
(8, 9):2099,
(4, 1):1127,
(9, 7):3106,
(1, 1):1097,
(9, 9):2208,
(3, 5):1067,
(8, 7):1942,
(7, 2):1063,
(10, 0):1560,
(2, 6):1049,
(3, 1):1501,
(0, 7):1079,
(7, 4):993,
(2, 8):1287,
(5, 3):1420,
(5, 9):814,
(7, 6):800,
(8, 10):949,
(9, 6):706,
(3, 4):661,
(0, 0):911,
(2, 1):580,
(5, 1):487,
(3, 10):436,
(10, 6):414,
(4, 5):367,
(1, 5):327,
(4, 6):316,
(6, 3):265,
(7, 5):72,
(0, 4):14,
        
}

arr = []
for k,v in div.items():
    arr.append(v)
retult = {}
arr = sorted(arr,reverse=True)
for index,v in enumerate(arr):
    if index<=20:
        for k,p in div.items():
            if v==p:
                retult[k] = p

for k,v in retult.items():
    print(f'{k}:{v},')


total = 0
for v in repeat_count.values():
    total = total +v 

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

























