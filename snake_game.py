import pygame
from collections import deque
from typing import Tuple,Deque
import random
import math
import numpy as np

class SnakeGame:

    white = (255, 255, 255)
    yellow = (255, 255, 102)
    black = (0, 0, 0)
    red = (213, 50, 80)
    green = (0, 255, 0)
    blue = (50, 153, 213)

    def __init__(self,board_size = 10,silent_mode = True,seed = 0,train_mode = False,model = None) -> None:
        self.board_size = board_size
        self.directions = ['up','down','left','right']
        self.snake:Deque[Tuple[int,int]] = deque()
        self.food = (0,0)
        self.direction = 'left'
        self.game_quit = False
        self.game_loss = False
        self.geme_win = False
        self.silent_mode = silent_mode
        self.train_mode = train_mode
        self.cell_size = 100
        self.screen_size = self.cell_size * self.board_size
        self.step_count = 0
        self.game_loop = 0
        random.seed(seed)
        self.model = model
        self.reset()
        self.scale = max(1, (32 + board_size - 1) // board_size)  # 确保 board_size * scale >= 32
        if not silent_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Snake Game by Edureka')
            self.clock = pygame.time.Clock()
            self.display_intervial = 1
            self.display_count = 0
            if not train_mode:
                self.font_style = pygame.font.SysFont(None, 50)
    
    def reset(self):
        self.game_loop += 1
        self.step_count = 0
        self.snake.clear()
        x, y = int(self.board_size/2),int(self.board_size/2)
        self.snake.append((x,y))
        for i in range(1,4):
            self.snake.append((x+i,y))
        self.direction = 'left'
        self.food = self.create_food()
        self.game_quit = False
        self.game_loss = False
    
    
    def get_play_ground(self):
        play_ground = np.full((self.board_size,self.board_size,3),255,dtype=np.uint8)
        snake_body_value = [[0, 0, v] for v in np.linspace(100,255,len(self.snake),dtype=np.uint8 ) ] # 蓝色
        food_value = SnakeGame.green  # 绿色
        snake_head_value = SnakeGame.red  # 红色
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
        obs_with_border = np.full((self.board_size + 2, self.board_size + 2, 3), 0, dtype=np.uint8)  # 初始化为全黑色
        obs_with_border[1:-1, 1:-1, :] = obs  # 在黑色背景中嵌入原始的游戏画面

        # 放大图像
        obs_with_border = np.repeat(np.repeat(obs_with_border, self.scale, axis=0), self.scale, axis=1)
        
        # 转置为 (3, w+2*self.scale, h+2*self.scale) 格式
        obs_with_border = np.transpose(obs_with_border, (2, 0, 1))
        return obs_with_border
    
    def draw_obs(self):
        obs = self.get_obs()
        obs = np.transpose(obs,(1,2,0))
        cell_size = self.screen_size / len(obs[0])
        for x,arr in enumerate(obs):
            for y,color in enumerate(arr):
                pygame.draw.rect(self.screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
        pygame.display.flip()      

    
    def draw(self):
        if self.silent_mode:
            return    
        for x,arr in enumerate(self.get_play_ground()):
            for y,color in enumerate(arr):
                pygame.draw.rect(self.screen, color, pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()      


    def create_food(self)->Tuple[int,int]:
        # 生成所有可能的位置
        all_positions = {(x, y) for x in range(self.board_size) for y in range(self.board_size)}
        # 移除蛇占据的位置
        snake_positions = {(x,y) for x,y in self.snake}
        available_positions = list(all_positions - snake_positions)

        #如果蛇的长度小于10，并且随机条件满足，则在右下角区域强化训练
        # if len(self.snake) < 10 and random.randrange(0, 11) > 8:
        #     right_positions = {(x, y) for x in range(self.board_size - 1, self.board_size) for y in range(self.board_size)}  # 最右1列
        #     bottom_positions = {(x, y) for x in range(self.board_size) for y in range(self.board_size - 1, self.board_size)}  # 最下1行
        #     left_positions = {(x, y) for x in range(0, 1) for y in range(self.board_size)}  # 最左1列
        #     up_positions = {(x, y) for x in range(self.board_size) for y in range(0, 1)}  # 最上1行
        #     corner_positions = right_positions | bottom_positions | left_positions | up_positions
        #     available_positions = list(set(available_positions) & corner_positions)  # 只保留边缘的可用位置

        if not available_positions:
            self.geme_win = True
            raise ValueError("No available positions to place the food")
            #available_positions = list(all_positions - snake_positions)
        x, y = random.choice(available_positions)
        return (x,y)
    
    def step(self)-> Tuple[bool,int]:
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
            if len(self.snake) == self.board_size**2:
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
        if x < 0 or x > self.board_size-1 or y< 0 or y> self.board_size-1:
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
        while not self.game_quit:
            while self.game_loss:
                if not self.train_mode and not self.silent_mode:
                    self.screen.fill(SnakeGame.blue)
                    mesg = self.font_style.render("You Lost! Press Q-Quit or C-Play Again", True,SnakeGame.red)
                    self.screen.blit(mesg, [self.screen_size / 6, self.screen_size / 6])
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
                        elif event.key == pygame.K_DOWN:
                            if self.direction != 'up':
                                self.direction = 'down'
            if self.display_count == 0:
                if self.model is not None:
                    p_direction = self.direction
                    action, _ = self.model.predict(self.get_obs(), deterministic=True)
                    
                    self.direction = self.directions[action]
                    if (p_direction == 'left' and self.direction == 'right') \
                        or (p_direction == 'right' and self.direction == 'left') \
                        or (p_direction == 'up' and self.direction == 'down') \
                        or (p_direction == 'down' and self.direction == 'up'):
                        print(f'---back_forawrd {p_direction} {self.direction}') 
                self.step()
                self.draw()
            self.display_count = (self.display_count + 1) % self.display_intervial
            if not self.silent_mode:
                self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":  
    game = SnakeGame(board_size=12,silent_mode=False,train_mode=False)
    game.run()