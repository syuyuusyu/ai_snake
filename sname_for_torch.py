import torch
import pygame
import random
from collections import deque
from module import SnakeNet
import torch.optim as optim
import torch.nn.functional as F


board_size = 20
play_ground = []

class Block():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y
        self.value = 1
        self.direction = 'left'
    def change_direction(self,direction):
        self.direction = direction
    def move(self,snake,foods) -> bool:
        x , y = self.x,self.y
        if self.direction == 'left':
            x -= 1
        elif self.direction == 'right':
            x += 1
        elif self.direction == 'up':
            y -= 1
        elif self.direction == 'down':
            y += 1
        if x < 0:
            x = board_size -1
        if x > board_size-1:
            x = 0
        if y < 0:
            y = board_size-1
        if y > board_size -1:
            y = 0
        bead = False
        for food in foods:
            if food.x == x and food.y == y:
                new_head = Block(x, y)
                new_head.direction = self.direction
                snake.appendleft(new_head)
                foods.pop()
                foods.appendleft(Food.create(board_size,snake))
                return bead          
        snake.pop()
        for block in snake:
            if x == block.x and y == block.y:
                bead = True
        new_head = Block(x, y)
        new_head.direction = self.direction
        snake.appendleft(new_head)
        return bead

class Food(Block):
    def __init__(self, x, y) -> None:
        super().__init__(x, y)
        self.value = 2
    
    @classmethod
    def create(cls, board_size, snake) -> 'Food':
        # 生成所有可能的位置
        all_positions = {(x, y) for x in range(board_size) for y in range(board_size)}
        # 移除蛇占据的位置
        snake_positions = {(block.x, block.y) for block in snake}
        available_positions = list(all_positions - snake_positions)
        
        if not available_positions:
            raise ValueError("No available positions to place the food")
        x, y = random.choice(available_positions)
        return cls(x, y)

def init(foods,snake):
    for _ in range(0,board_size):
        play_ground.append([0]*board_size)
    x, y = int(board_size/2),int(board_size/2)
    snake.append(Block(x,y))
    for i in range(1,4):
        snake.append(Block(x+i,y))
    foods.appendleft(Food.create(board_size=board_size,snake=snake))

def update_play_ground(foods,snake):
    for i in range(0,board_size):
        play_ground[i] = [0]*board_size  
    for food in foods:
        play_ground[food.x][food.y] = food.value
    for i, block in enumerate(snake):
        if i == 0:  # 蛇头
            if block.direction == 'up':
                play_ground[block.x][block.y] = 3
            elif block.direction == 'down':
                play_ground[block.x][block.y] = 4
            elif block.direction == 'left':
                play_ground[block.x][block.y] = 5
            elif block.direction == 'right':
                play_ground[block.x][block.y] = 6
        else:  # 蛇身
            play_ground[block.x][block.y] = block.value


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
}

screen_width = 1000
block_size = screen_width/board_size
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_width))
pygame.display.set_caption('Snake Game by Edureka')
clock = pygame.time.Clock()

font_style = pygame.font.SysFont(None, 50)

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    screen.blit(mesg, [screen_width / 6, screen_width / 6])

def draw():
    for x,arr in enumerate(play_ground):
        for y,value in enumerate(arr):
            if value != 0:
                pygame.draw.rect(screen, color_dic[value], pygame.Rect(x * block_size, y * block_size, block_size, block_size))

model = SnakeNet(board_size=board_size)
device = 'mps'
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 探索率
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
memory = deque(maxlen=10000)  # 经验回放缓冲区
batch_size = 128
loss_fn = F.mse_loss
from train import train,predict

def game_loop():
    input_delay = 6  # 设定输入处理延迟
    input_counter = 0
    game_quit = False
    game_loss = False
    snake = deque()
    foods = deque()
    init(foods,snake)
    while not game_quit:
        #训练模型。评估模型的效能
        train(model,optimizer,loss_fn,memory,gamma,batch_size,device)
        while game_loss == True:
            game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_quit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if snake[0].direction != 'right':
                        snake[0].change_direction('left')
                elif event.key == pygame.K_RIGHT:
                    if snake[0].direction != 'left':
                        snake[0].change_direction('right')
                elif event.key == pygame.K_UP:
                    if snake[0].direction != 'down':
                        snake[0].change_direction('up')
                elif event.key == pygame.K_DOWN:
                    if snake[0].direction != 'up':
                        snake[0].change_direction('down')
        if input_counter == 0:
            update_play_ground(foods,snake)
            screen.fill(white)
            draw()
            previous_length = len(snake)
            previous_state = torch.tensor(play_ground, dtype=torch.float32).unsqueeze(0).to(device)
            action = ['left', 'right', 'up', 'down'].index(snake[0].direction)
            game_loss = snake[0].move(snake,foods)
            # 计算奖励
            reward = 0
            if len(snake) > previous_length:
                reward = 10  # 吃到食物的奖励
            elif game_loss:
                reward = -100  # 撞到自己或墙壁的惩罚
            else:
                reward = -1  # 其他每步的惩罚
            next_state = torch.tensor(play_ground, dtype=torch.float32).unsqueeze(0).to(device)
            memory.append((previous_state, action, reward, next_state, game_loss))                
            pygame.display.flip()
            #预测新的方向
            snake[0].direction = predict(model,play_ground,device)
        input_counter = (input_counter + 1) % input_delay
        clock.tick(60) 
    pygame.quit()


if __name__ == "__main__":  
    game_loop()

