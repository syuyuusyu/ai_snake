import pygame
from collections import deque
from typing import Tuple,Deque
import random
import math

class SnakeGame:

    white = (255, 255, 255)
    yellow = (255, 255, 102)
    black = (0, 0, 0)
    red = (213, 50, 80)
    green = (0, 255, 0)
    blue = (50, 153, 213)
    color_dic = {
        1 : blue,
        2 : green,
        3 : blue,
        4 : blue,
        5 : blue,
        6 : blue,
        7 : blue,
    }

    def __init__(self,board_size = 10,silent_mode = True,seed = 0,train_mode = False) -> None:
        self.board_size = board_size
        self.directions = ['up','down','left','right']
        self.snake:Deque[Tuple[int,int,int]] = deque()
        self.play_ground = [[0]*board_size for _ in range(0,board_size)]
        self.food = (0,0,2)
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
        self.reset()
        if not silent_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            print(self.screen)
            pygame.display.set_caption('Snake Game by Edureka')
            self.clock = pygame.time.Clock()
            self.display_intial = 10
            self.display_count = 0
            if not train_mode:
                self.font_style = pygame.font.SysFont(None, 50)
    
    def reset(self):
        self.game_loop += 1
        self.step_count = 0
        self.play_ground = [ [0]*self.board_size for _ in range(0,self.board_size) ]
        self.snake.clear()
        x, y = int(self.board_size/2),int(self.board_size/2)
        self.snake.append((x,y,1))
        for i in range(1,4):
            self.snake.append((x+i,y,1))
        self.direction = 'left'
        self.food = self.create_food()
        self.game_quit = False
        self.game_loss = False


    def create_food(self)->Tuple[int,int,int]:
        # 生成所有可能的位置
        all_positions = {(x, y) for x in range(self.board_size) for y in range(self.board_size)}
        # 移除蛇占据的位置
        snake_positions = {(x,y) for x,y,_ in self.snake}
        available_positions = list(all_positions - snake_positions)
        
        if not available_positions:
            self.geme_win = True
            raise ValueError("No available positions to place the food")
        x, y = random.choice(available_positions)
        return (x,y,1)
    
    def update_play_ground(self):
        self.play_ground = [ [0]*self.board_size for _ in range(0,self.board_size) ]
        self.play_ground[self.food[0]][self.food[1]] = self.food[2]
        for index,(x,y,v) in enumerate(self.snake):
            if index == 0:
                _v = self.directions.index(self.direction)+1
                self.play_ground[x][y] = 2 +_v
            else:
                self.play_ground[x][y] = v
    
    def draw(self):
        if self.silent_mode:
            return
        self.screen.fill(SnakeGame.white)
        for x,arr in enumerate(self.play_ground):
            for y,value in enumerate(arr):
                if value != 0:
                    pygame.draw.rect(self.screen, SnakeGame.color_dic[value], pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def step(self)-> Tuple[bool,float,int]:
        """Move the snake one step in the current direction.
        
        Returns:
            Tuple[bool,float, int]: A tuple containing a boolean indicating
                                if the game is over and a float representing the reward. 

                                int representing the stat of this this step 
                                    0: the head leave the food
                                    1: this head approch the food
                                    2: hit wall
                                    3: collied self
                                    4: eat food
        """
        snake_length = len(self.snake)
        x,y,_ = self.snake[0]
        step_point = [[0,-1],[0,1],[-1,0],[1,0]]
        index = self.directions.index(self.direction)
        previous_distance = math.sqrt((x - self.food[0])**2 + (y - self.food[1])**2)
        x,y = x + step_point[index][0], y+step_point[index][1]
        distance = math.sqrt((x - self.food[0])**2 + (y - self.food[1])**2)
        if (x,y) == self.food[:2]:
            self.snake.appendleft((x,y,1))
            self.food = self.create_food()
            #print('eat food')
            return (False,float(snake_length*1),4)
        self.snake.pop()
        for _x,_y,_ in self.snake:
            if (x,y) == (_x,_y):
                #print('collied self')
                self.game_loss = True
                return (True,-5.0,3)
        if x < 0 or x > self.board_size-1 or y< 0 or y> self.board_size-1:
            #print('hit wall')
            self.game_loss = True
            return (True,-5.0,2)
        reward,state = (0.1,1) if previous_distance > distance else (-0.1,0)
        self.snake.appendleft((x,y,1))
    
        return (False,reward,state)
    
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
                self.step()
                self.update_play_ground()
                self.draw()
            self.display_count = (self.display_count + 1) % self.display_intial
            if not self.silent_mode:
                self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":  
    game = SnakeGame(board_size=10,silent_mode=False,train_mode=True)
    game.run()