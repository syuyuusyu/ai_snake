import pygame
import random
#from pygame.locals import *

# 初始化游戏
pygame.init()

# 游戏窗口大小
WIDTH = 400
HEIGHT = 400

# 方块大小
BLOCK_SIZE = 20

# 颜色定义
colors = {
    'background': (0, 0, 0),
    'snake': (0, 255, 0),
    'food': (255, 0, 0)
}

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('贪吃蛇游戏')

# 蛇的初始位置和方向
snake_pos = [{'x': 10, 'y': 10}]
direction = 'right'
next_direction = direction

# 食物的位置
food_pos = {'x': random.randint(0, (WIDTH // BLOCK_SIZE) - 1),
            'y': random.randint(0, (HEIGHT // BLOCK_SIZE) - 1)}

# 游戏速度和分数
clock = pygame.time.Clock()
speed = 10
score = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != 'down':
                next_direction = 'up'
            elif event.key == pygame.K_DOWN and direction != 'up':
                next_direction = 'down'
            elif event.key == pygame.K_LEFT and direction != 'right':
                next_direction = 'left'
            elif event.key == pygame.K_RIGHT and direction != 'left':
                next_direction = 'right'

    # 更新方向
    direction = next_direction

    # 移动头部
    head = snake_pos[0].copy()
    if direction == 'up':
        head['y'] -= 1
    elif direction == 'down':
        head['y'] += 1
    elif direction == 'left':
        head['x'] -= 1
    else:
        head['x'] += 1

    # 检查边界碰撞
    if (head['x'] < 0 or head['x'] >= WIDTH // BLOCK_SIZE or
        head['y'] < 0 or head['y'] >= HEIGHT // BLOCK_SIZE):
        print('游戏结束，得分:', score)
        pygame.quit()
        exit()

    # 检查自身碰撞
    if head in snake_pos:
        print('游戏结束，得分:', score)
        pygame.quit()
        exit()

    # 插入新的头部
    snake_pos.insert(0, head)

    # 检测是否吃到食物
    if head['x'] == food_pos['x'] and head['y'] == food_pos['y']:
        score += 1
        speed += 0.5
        # 生成新食物
        while True:
            new_food = {'x': random.randint(0, (WIDTH // BLOCK_SIZE) - 1),
                        'y': random.randint(0, (HEIGHT // BLOCK_SIZE) - 1)}
            if new_food not in snake_pos:
                food_pos = new_food
                break
    else:
        # 移除尾部，保持长度不变
        snake_pos.pop()

    # 绘制背景
    window.fill(colors['background'])

    # 绘制蛇
    for pos in snake_pos:
        pygame.draw.rect(window, colors['snake'],
                        (pos['x'] * BLOCK_SIZE, pos['y'] * BLOCK_SIZE,
                         BLOCK_SIZE - 1, BLOCK_SIZE - 1))

    # 绘制食物
    pygame.draw.rect(window, colors['food'],
                    (food_pos['x'] * BLOCK_SIZE, food_pos['y'] * BLOCK_SIZE,
                     BLOCK_SIZE - 1, BLOCK_SIZE - 1))

    # 更新画面和控制速度
    pygame.display.flip()
    clock.tick(speed)