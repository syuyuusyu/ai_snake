import argparse
import random
from collections import deque

import numpy as np
import pygame

from board_config import (
    DEFAULT_BOARD_HIGH,
    DEFAULT_BOARD_WIDTH,
    add_board_arguments,
    observation_scale,
    validate_board_dimensions,
)


class SnakeGame:

    white = (255, 255, 255)
    yellow = (255, 255, 102)
    black = (0, 0, 0)
    red = (213, 50, 80)
    green = (0, 255, 0)
    blue = (50, 153, 213)

    def __init__(self,board_width=DEFAULT_BOARD_WIDTH, board_high=DEFAULT_BOARD_HIGH,silent_mode = True,seed = 0,train_mode = False,model = None,bfs_intensity = 0) -> None:
        print(f'SnakeGame {bfs_intensity}')
        validate_board_dimensions(board_width, board_high)
        self.board_width = board_width
        self.board_high = board_high
        self.directions = ['up','down','left','right']
        self.snake:deque[tuple[int,int]] = deque()
        self.food = (0,0)
        self.direction = 'left'
        self.game_quit = False
        self.game_loss = False
        self.game_win = False
        self.silent_mode = silent_mode
        self.train_mode = train_mode
        self.cell_size = 100
        self.screen_width = self.cell_size * self.board_width
        self.screen_high = self.cell_size * self.board_high
        self.step_count = 0
        self.game_loop = 0
        self._bfs_reachable_area = set()
        self.rng = random.Random(seed)
        self.model = model
        self.bfs_intensity = bfs_intensity
        self.reset()
        self.scale = observation_scale(self.board_width, self.board_high)
        if not silent_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_high))
            pygame.display.set_caption('Snake Game by Edureka')
            self.clock = pygame.time.Clock()
            self.display_intervial = 1
            self.display_count = 0
            if not train_mode:
                self.font_style = pygame.font.SysFont(None, 50)
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.game_loop += 1
        self.step_count = 0
        self.snake.clear()
        x, y = min(self.board_width // 2, self.board_width - 4), self.board_high // 2
        self.snake.append((x,y))
        for i in range(1,4):
            self.snake.append((x+i,y))
        self.direction = 'left'
        self.food = self.create_food()
        self.game_quit = False
        self.game_loss = False
        self.game_win = False
    def get_bfs_color(self):
        """根据强度系数计算BFS区域的颜色"""
        base_color = (250, 255, 250)  # 初始浅色
        target_color = (200, 240, 200)  # 最终颜色
        current_color = tuple(
            int(base + (target - base) * self.bfs_intensity)
            for base, target in zip(base_color, target_color)
        )
        return current_color
    
    
    def get_play_ground(self):
        play_ground = np.full((self.board_width,self.board_high,3),255,dtype=np.uint8)
        snake_body_value = [[0, 0, v] for v in np.linspace(100,255,len(self.snake),dtype=np.uint8 ) ] # 蓝色
        food_value = SnakeGame.green  # 绿色
        snake_head_value = SnakeGame.red
        bfs_color = self.get_bfs_color()
        self._bfs_reachable_area = self.bfs_reachable_area()
        for (x,y) in self._bfs_reachable_area:
            play_ground[(x,y)] = bfs_color
        play_ground[self.food] = food_value
        for index, (x,y) in enumerate(self.snake):
            if index ==0:
                play_ground[(x,y)] = snake_head_value
            # elif index == len(self.snake) -1:
            #     play_ground[(x,y)] = SnakeGame.yellow
            else:
                play_ground[(x,y)] = snake_body_value[index]
        #obs = np.repeat(np.repeat(obs,self.scale,axis=0),self.scale, axis=1)
        #obs = np.transpose(obs,(1,0,2))
        return play_ground
    
    def get_obs(self):
        obs = self.get_play_ground()

        # 增加黑色边界
        obs_with_border = np.full((self.board_width + 2, self.board_high + 2, 3), 0, dtype=np.uint8)  # 初始化为全黑色
        obs_with_border[1:-1, 1:-1, :] = obs  # 在黑色背景中嵌入原始的游戏画面

        # 放大图像
        obs_with_border = np.repeat(np.repeat(obs_with_border, self.scale, axis=0), self.scale, axis=1)
        
        # 转置为 (3, w+2*self.scale, h+2*self.scale) 格式
        obs_with_border = np.transpose(obs_with_border, (2, 0, 1))
        return obs_with_border
    
    def draw_obs(self):
        obs = self.get_obs()
        obs = np.transpose(obs,(1,2,0))
        cell_width = self.screen_width / obs.shape[0]
        cell_high = self.screen_high / obs.shape[1]
        for x,arr in enumerate(obs):
            for y,color in enumerate(arr):
                pygame.draw.rect(self.screen, color, pygame.Rect(x * cell_width, y * cell_high, cell_width, cell_high))
        pygame.display.flip()      

    def bfs_reachable_area(self):
        """
        使用 BFS 从给定的起点（蛇尾）查找可达空间。

        参数:
        - start: tuple, BFS 搜索的起点（蛇尾位置）
        - snake_body: list, 包含蛇体的位置，防止访问
        - board_width / board_high: int, 棋盘的宽和高

        返回:
        - reachable_spaces: set, 可达空间的集合
        """
        start = self.snake[-1]
        queue = [start]
        visited = set(self.snake) -{start} # 蛇体位置不可访问
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
                if 0 <= new_position[0] < self.board_width and 0 <= new_position[1] < self.board_high:
                    queue.append(new_position)
        return reachable_spaces

    def safety_score(self) -> float:
        """Score how much of the board the head can safely reach."""
        if not self.snake:
            return 0.0

        snake = list(self.snake)
        head = snake[0]
        tail = snake[-1]
        blocked = set(snake[1:-1])
        queue = deque([head])
        reachable = set()

        while queue:
            position = queue.popleft()
            if position in reachable or position in blocked:
                continue

            x, y = position
            if not (0 <= x < self.board_width and 0 <= y < self.board_high):
                continue

            reachable.add(position)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((x + dx, y + dy))

        traversable_count = self.board_width * self.board_high - len(blocked)
        reachable_ratio = len(reachable) / traversable_count
        head_can_reach_tail = float(tail in reachable)
        return 0.7 * reachable_ratio + 0.3 * head_can_reach_tail
    
    
    def draw(self):
        if self.silent_mode:
            return
        #self.draw_reachable_spaces() 
        for x,arr in enumerate(self.get_play_ground()):
            for y,color in enumerate(arr):
                pygame.draw.rect(self.screen, color, pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()      


    def create_food(self)->tuple[int,int]:
        # 生成所有可能的位置
        all_positions = {(x, y) for x in range(self.board_width) for y in range(self.board_high)}
        # 移除蛇占据的位置
        snake_positions = {(x,y) for x,y in self.snake}
        available_positions = list(all_positions - snake_positions)

        #如果蛇的长度小于10，并且随机条件满足，则在右下角区域强化训练
        # if len(self.snake) < 10 and random.randrange(0, 11) > 8:
        #     right_positions = {(x, y) for x in range(self.board_width - 1, self.board_width) for y in range(self.board_high)}  # 最右1列
        #     bottom_positions = {(x, y) for x in range(self.board_width) for y in range(self.board_high - 1, self.board_high)}  # 最下1行
        #     left_positions = {(x, y) for x in range(0, 1) for y in range(self.board_high)}  # 最左1列
        #     up_positions = {(x, y) for x in range(self.board_width) for y in range(0, 1)}  # 最上1行
        #     corner_positions = right_positions | bottom_positions | left_positions | up_positions
        #     available_positions = list(set(available_positions) & corner_positions)  # 只保留边缘的可用位置

        if not available_positions:
            self.geme_win = True
            raise ValueError("No available positions to place the food")
            #available_positions = list(all_positions - snake_positions)
        x, y = self.rng.choice(available_positions)
        return (x,y)
    
    def step(self)-> tuple[bool,int]:
        """Move the snake one step in the current direction.
        
        Returns:
            Tuple[bool, int]: A tuple containing a boolean indicating if the game is over.

                                int representing the stat of this this step 

                                    0: the head leave the food
                                    1: this head approch the food
                                    2: hit wall
                                    3: collied self
                                    4: eat food
                                    5: Victory
        """
        x_origin,y_origin = self.snake[0]
        step_point = [[0,-1],[0,1],[-1,0],[1,0]]
        index = self.directions.index(self.direction)
        x,y = x_origin + step_point[index][0], y_origin+step_point[index][1]

        if (x,y) == self.food:
            self.snake.appendleft((x,y))
            if len(self.snake) == self.board_width * self.board_high:
                #Victory!!!
                return (True,5) 
            if self.model is not None:
                print(len(self.snake))
            self.food = self.create_food()
            #print('eat food')
            return (False,4)
        self.snake.pop()
        for _x,_y in self.snake:
            if (x,y) == (_x,_y):
                #print('collied self')
                self.game_loss = True
                return (True,3)
        if x < 0 or x > self.board_width-1 or y< 0 or y> self.board_high-1:
            #print('hit wall')
            self.game_loss = True
            return (True,2)
        self.snake.appendleft((x,y))
        state = 1 if np.linalg.norm(np.array(self.snake[0]) - np.array(self.food)) < np.linalg.norm(np.array(self.snake[1]) - np.array(self.food)) else 0
        return (False,state)
    
    def close(self):
        if not self.silent_mode:
            pygame.quit()   

    def run(self):
        from jev_assist import Shield

        shield = Shield(hunger=Shield.default_hunger(self))
        shield.record(self, ate=False)
        while not self.game_quit:
            if self.game_loss:
                shield.reset()
            while self.game_loss:
                if not self.train_mode and not self.silent_mode:
                    self.screen.fill(SnakeGame.blue)
                    mesg = self.font_style.render("You Lost! Press Q-Quit or C-Play Again", True,SnakeGame.red)
                    self.screen.blit(mesg, [self.screen_width / 6, self.screen_high / 6])
                    pygame.display.update()
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                self.game_quit = True
                                self.game_loss = False
                            if event.key == pygame.K_c:
                                self.reset()
                else:
                    self.reset()
            if not self.silent_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_quit = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            if self.direction != 'right':
                                self.direction = 'left'
                        elif event.key == pygame.K_RIGHT:
                            if self.direction != 'left':
                                self.direction = 'right'
                        elif event.key == pygame.K_UP:
                            if self.direction != 'down':
                                self.direction = 'up'
                        elif event.key == pygame.K_DOWN and self.direction != 'up':
                            self.direction = 'down'
            if self.display_count == 0:
                if self.model is not None:
                    p_direction = self.direction
                    action, _ = self.model.predict(self.get_obs(), action_masks=shield.mask(self), deterministic=True)
                    
                    self.direction = self.directions[action]
                    if (p_direction == 'left' and self.direction == 'right') \
                        or (p_direction == 'right' and self.direction == 'left') \
                        or (p_direction == 'up' and self.direction == 'down') \
                        or (p_direction == 'down' and self.direction == 'up'):
                        print(f'---back_forawrd {p_direction} {self.direction}')
                _, state = self.step()
                shield.record(self, ate=state == 4)
                self.draw()
            self.display_count = (self.display_count + 1) % self.display_intervial
            if not self.silent_mode:
                self.clock.tick(10)
        pygame.quit()

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    add_board_arguments(parser)
    args = parser.parse_args()
    game = SnakeGame(board_width=args.board_width, board_high=args.board_high,
                     silent_mode=False, train_mode=False)
    game.run()
