import numpy as np

# 假设 play_ground 是一个 20x20 的二维数组
play_ground = np.zeros((20, 20))

# 假设当前方向是 "上"，则方向编码为 [1, 0, 0, 0]
direction = [1, 0, 0, 0]

# 将方向信息扩展为与 play_ground 形状匹配的矩阵
direction_info = np.full(play_ground.shape, direction)

# 将 play_ground 和 direction_info 组合为输入
input_data = np.stack([play_ground, direction_info], axis=-1) 
print(input_data)