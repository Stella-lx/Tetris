import random
import keyboard
import os
import time
import pygame

"""
写俄罗斯方块PPO强化学习环境python代码
"""
class Env:
    def __init__(self, height = 12, width = 10):

        self.height = height
        self.width = width
        self.secene_row = height + 4
        self.state = [[0 for _ in range(width)] for _ in range(self.secene_row)]
        self.x = 0 #
        self.y = 0 #
        self.done = False
        self.next_tetris = None
        self.score = 0 # 落地18，消除1行100
        # self.reward = 0
        self.one_step = 0
        self.hard_drop = False
        self.cleans = 0
        self.max_step = 10000
        self.rotation = 0
        self.tetris_shape = { # 逆时针旋转
            0: [[[1,1,0],[0,1,1]],
                [[0,1],[1,1],[1,0]]
            ], #"Z"
            1: [[[1,1,1,1]],
                [[1],[1],[1],[1]]
                ], #"一"
            2: [[[1,1],[1,1]]
            ], #"田"
            3: [[[0,1,0],[1,1,1]],
                [[0,1],[1,1],[0,1]],
                [[1,1,1],[0,1,0]],
                [[1,0],[1,1],[1,0]]
            ], #"-|-"
            4: [[[0,0,1],[1,1,1]],
                [[1,1],[0,1],[0,1]],
                [[1,1,1],[1,0,0]],
                [[1,0],[1,0],[1,1]]
            ], #"--|"
            5: [[[1,0,0],[1,1,1]],
                [[0,1],[0,1],[1,1]],
                [[1,1,1],[0,0,1]],
                [[1,1],[1,0],[1,0]]
            ], #"|--"
            6: [[[0,1,1],[1,1,0]],
                [[1,0],[1,1],[0,1]]
            ] #"S"
        }

        # Pygame初始化（新增）
        pygame.init()
        self.block_size = 30  # 每个方块的像素大小
        self.screen_width = self.width * self.block_size
        self.screen_height = self.secene_row * self.block_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("俄罗斯方块 PPO环境")
        self.clock = pygame.time.Clock()  # 控制帧率

        # 方块颜色映射（新增，对应不同形状的方块）
        self.colors = [
            (0, 0, 0),          # 0: 空白
            (255, 255, 255),    # 1: 活动方块（白色）
            (255, 0, 0),        # 2: Z形（红色）
            (0, 255, 0),        # 3: I形（绿色）
            (0, 0, 255),        # 4: O形（蓝色）
            (255, 255, 0),      # 5: T形（黄色）
            (255, 0, 255),      # 6: J形（紫色）
            (0, 255, 255),      # 7: L形（青色）
            (128, 0, 128)       # 8: S形（深紫）
        ]

    def step(self, action):
        # action[0]左，change[1]右，change[2]旋转，change[3]下落
        # action = self.key_control()
        action_list = [0, 0, 0, 0]
        # 改规则
        if action != None:
            if self.one_step >= 9:
                action = 3
                self.hard_drop = True
                self.one_step = 0
            else:
                self.hard_drop = False
                self.one_step += 1

        if action == 0:
            action_list = [1, 0, 0, 0]
        elif action == 1:
            action_list = [0, 1, 0, 0]
        elif action == 2:
            action_list = [0, 0, 1, 0]
        elif action == 3:
            action_list = [0, 0, 0, 1]
        clean_line, nextstate = self.clean_lines(self.state)
        self.cleans = clean_line
        next_state = self.get_next_state(action_list)

        # 环境基础奖励
        reward = 0

        score = self.score
        done = self.done

        return next_state, done, reward, score

    def get_next_state(self, action):
        clean_lines, next_state = self.clean_lines(self.state)
        if self.is_block_end(self.block):
            self.score += 18
            self.one_step = 0
            if self.is_done():
                self.done = True
                return self.state
            else:
                self.block, self.next_block = self.update_block()
                return self.block_in_secene(self.x, self.y)
        else:
            if clean_lines == 0:
                if self.move_is_valid(action):
                # is_valid = True
                # if is_valid:
                    self.move(action)
                        # Clear temporary block positions (value 1) before placing new block
                    for row in range(len(self.state)):
                        for col in range(len(self.state[row])):
                            if self.state[row][col] == 1:
                                self.state[row][col] = 0
                    return self.block_in_secene(self.x, self.y)
                else:
                    return self.state
            else:
                return next_state

    
    def reset(self):
        self.next_block_index = random.randint(0, 6)
        # self.next_block_index = 1
        self.rotation = 0
        self.next_block = self.tetris_shape[self.next_block_index][self.rotation]
        # self.next_block = self.tetris_shape[1][self.rotation]
        self.block = self.next_block
        self.block_index = self.next_block_index
        self.score = 0
        self.cleans = 0
        self.reward = 0  # 重置奖励！
        self.done = False  # 重置游戏状态
        self.one_step = 0
        if self.next_block_index == 1:
            self.x = self.width // 2 - 1
            self.y = 1           
        else:
            self.x = self.width // 2
            self.y = 0
        self.state = [[0 for _ in range(self.width)] for _ in range(self.secene_row)]
        return self.state

    def render(self, state):
        # 填充背景为黑色
        self.screen.fill((0, 0, 0))

        # 绘制每个方块
        for row in range(self.secene_row):
            for col in range(self.width):
                # 获取方块值（0-8）
                block_value = state[row][col]
                # 计算像素位置
                x = col * self.block_size
                y = row * self.block_size
                # 绘制方块（带边框）
                rect = pygame.Rect(x, y, self.block_size - 1, self.block_size - 1)  # -1留边框间隙
                pygame.draw.rect(self.screen, self.colors[block_value], rect)

        # 更新显示
        pygame.display.flip()
        # 控制帧率（可选，避免过快）
        self.clock.tick(10)  # 10FPS，可调整
    
    
    def block_in_secene(self, x, y):
        for row in range(len(self.block)):
            for col in range(len(self.block[row])):
                x_ = x + col
                y_ = y + row
                if 0 <= x_ < self.width and 0 <= y_ < len(self.state):
                    if self.block[row][col] == 1:
                        if self.is_block_end(self.block): 
                            self.state[y_][x_] = self.block_index + 2
                        else:
                            self.state[y_][x_] = 1
        return self.state
    
    
    def move(self, action):
        # action[0]左，action[1]右，action[2]旋转，action[3]下落
        if not self.is_block_end(self.block):
            if action[0] == 1:
                if self.move_is_valid(action):
                    self.x -= 1
            elif action[1] == 1:
                if self.move_is_valid(action):
                    self.x += 1
            elif action[2] == 1: # 旋转
                if self.block_index == 0:
                    if self.move_is_valid(action):
                        self.rotation = (self.rotation + 1) % 2
                        self.block = self.tetris_shape[self.block_index][self.rotation]
                elif self.block_index == 1:
                    # print(f"1:{self.y}")
                    if self.move_is_valid(action):
                        # print(f"is_valid:{self.move_is_valid(action)}")
                        # print(f"2:{self.y}")
                        self.rotation = (self.rotation + 1) % 2
                        self.block = self.tetris_shape[self.block_index][self.rotation]
                        # print(self.block)
                        if self.rotation == 1:
                            self.x += 1
                            self.y -= 1
                        else:
                            self.x -= 1
                            self.y += 1
                        # print(f"3:{self.y}")
                elif self.block_index == 3: 
                    if self.move_is_valid(action):
                        self.rotation = (self.rotation + 1) % 4
                        self.block = self.tetris_shape[self.block_index][self.rotation]
                        if self.rotation == 2:
                            self.y += 1
                        elif self.rotation == 3:
                            self.x += 1
                            self.y -= 1
                        elif self.rotation == 0:
                            self.x -= 1
                elif self.block_index == 4: 
                    if self.move_is_valid(action):
                        self.rotation = (self.rotation + 1) % 4
                        self.block = self.tetris_shape[self.block_index][self.rotation]
                elif self.block_index == 5: 
                    if self.move_is_valid(action):
                        self.rotation = (self.rotation + 1) % 4
                        self.block = self.tetris_shape[self.block_index][self.rotation]
                elif self.block_index == 6: 
                    if self.move_is_valid(action):
                        self.rotation = (self.rotation + 1) % 2
                        self.block = self.tetris_shape[self.block_index][self.rotation]

            elif action[3] == 1:
                while self.move_is_valid(action):
                    self.y += 1
                if self.is_done():
                    self.done = True
                
        else:
            if self.is_done():
                self.done = True
            else:
                self.score += 18
                self.block, self.next_block = self.update_block()
                self.one_step = 0

    """局内更新方块"""
    def update_block(self):
        self.block_index = self.next_block_index
        self.block = self.next_block
        self.rotation = 0
        self.next_block_index = random.randint(0, 6)
        self.next_block = self.tetris_shape[self.next_block_index][self.rotation]
        # self.next_block = 1
        # self.next_block = self.tetris_shape[1][self.rotation]
        if self.block_index == 1:
            self.x = self.width // 2 - 2
            self.y = 1
        else:
            self.x = self.width // 2 - 1
            self.y = 0
        return self.block, self.next_block

    """判断方块落地或者底部碰撞的情况""" #
    def is_block_end(self,block):
        for row in range(len(block)):
            for col in range(len(block[row])):
                if block[row][col] == 1:
                    board_x= self.x + col
                    board_y = self.y + row
            
                    # 检查是否触底
                    if board_y >= self.secene_row - 1:
                        return True

                    # 检查下方是否有固定方块
                    if board_y < self.secene_row - 1 and board_x >= 0 and board_x <= self.width - 1 and self.state[board_y + 1][board_x] > 1:
                        return True
        return False
        # row = len(block)
        # last_row = block[-1]
        # for i in range(len(last_row)):
        #     if self.y + len(block) - 1 >= self.height:
        #         return True
        #     elif last_row[i] == 1 and self.state[self.y + row][self.x + i] > 1:
        #         return True
        #     else:
        #         return False
    
    """判断游戏是否结束"""
    def is_done(self):
        if any(x > 0 for x in self.state[4]):
            return True
        else:
            return False
        # if self.y <= 4:
        #     return True
        # else:
        #     return False 
    
    """键盘控制动作"""

    def key_control(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # 退出游戏
                    pygame.quit()
                    exit()
                # 键盘按键（对应原key_control逻辑）
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        return 0  # 左移
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        return 1  # 右移
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        return 2  # 旋转
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        return 3  # 下落
            return None  # 无操作


    """判断移动是否有效""" #
    def move_is_valid(self, action):
        if action[0] == 1: # 左移
            a = 0
            for row in self.block:
                if self.x <= 0:
                    return False
                for i in range(len(row)):
                    if row[i] == 1 and self.state[self.y + a][self.x + i - 1] > 1:
                        return False
                    return True
                a += 1
        elif action[1] == 1: # 右移
            a = 0
            for row in self.block:
                for i in range(len(row)):
                    if self.x + i >= self.width - 1:
                        return False
                    if row[i] == 1 and self.state[self.y + a][self.x + i + 1] > 1:
                        return False
                a += 1
            return True
        # 旋转
        elif action[2] == 1:
            if self.block_index == 0 or self.block_index == 1 or self.block_index == 6:
                new_block = self.tetris_shape[self.block_index][(self.rotation + 1) % 2]
            elif self.block_index == 3 or self.block_index == 4 or self.block_index == 5:
                new_block = self.tetris_shape[self.block_index][(self.rotation + 1) % 4]
            elif self.block_index == 2:
                new_block = self.block

            a = 0
            for row in new_block:
                for i in range(len(row)):
                    if self.block_index == 0 or self.block_index == 4 or self.block_index == 5 or self.block_index == 6:
                        new_x = self.x
                        new_y = self.y
                    elif self.block_index == 1:
                        if self.rotation == 0:
                            new_x = self.x + 1
                            new_y = self.y - 1
                        else:
                            new_x = self.x - 1
                            new_y = self.y + 1
                    elif self.block_index == 3:
                        if self.rotation == 1:
                            new_x = self.x
                            new_y = self.y + 1
                        elif self.rotation == 2:
                            new_x = self.x + 1
                            new_y = self.y - 1
                        elif self.rotation == 3:
                            new_x = self.x - 1
                            new_y = self.y
                        else:
                            new_x = self.x
                            new_y = self.y
                    else:
                        return False
                    # if new_x < 0 or new_x + i >= self.width or new_y < 0 or new_y + a >= self.height:
                    if new_x < 0 or new_x + i >= self.width:
                    #     # print(f"self_y:{self.y}")
                    #     # print(f"new_y:{new_y}")
                        return False
                    if row[i] == 1 and self.state[new_y + a][new_x + i] > 1:
                        return False
                a += 1
            return True
        #下落
        elif action[3] == 1:
            if self.is_block_end(self.block):
                return False
        return True
    
    def clean_lines(self, state):
        lines = 0
        for row in state:
            if 0 not in row:
                lines += 1
                state.remove(row)
                # self.state.insert(0, [0 for _ in range(self.width)])
                state.insert(0, [0] * self.width)
        # 消行分数
        if lines == 1:
            self.score += 100
        elif lines == 2:
            self.score += 300
        elif lines == 3:
            self.score += 500
        elif lines == 4:
            self.score += 800
        return lines, state
    