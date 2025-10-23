import random
import keyboard
import os
import time

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

    def step(self, action):
        # action[0]左，change[1]右，change[2]旋转，change[3]下落
        # action =self.key_control()
        action_list = [0, 0, 0, 0]
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
        if clean_line > 0:
            # 根据消除行数给予基础奖励
            if clean_line == 1:
                reward += 10
            elif clean_line == 2:
                reward += 30
            elif clean_line == 3:
                reward += 60
            elif clean_line == 4:
                reward += 100

        score = self.score
        done = self.done

        return next_state, done, reward, score

    def get_next_state(self, action):
        clean_lines, next_state = self.clean_lines(self.state)
        if self.is_block_end(self.block):
            self.score += 18
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
        if self.next_block_index == 1:
            self.x = self.width // 2 - 1
            self.y = 1           
        else:
            self.x = self.width // 2
            self.y = 0
        self.state = [[0 for _ in range(self.width)] for _ in range(self.secene_row)]
        return self.state

    def render(self, state):
        os.system('cls')
        print("--------------------")
        for i in range(self.secene_row):
            for j in range(self.width):
                if state[i][j] == 0:
                    print("..", end="")
                else:
                    print("##", end="")
            print()
        print("--------------------")
    
    
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
        # action[0]左，action[1]右，action[2]下，action[3]旋转,action[4]下落, action[5]重置
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
        else:
            if self.is_done():
                self.done = True
            else:
                self.score += 18
                self.block, self.next_block = self.update_block()

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
        if self.y <= 4:
            return True
        else:
            return False
        # # 检查新方块是否可以放置在初始位置
        # for row in range(len(self.block)):
        #     for cell in range(len(self.block[row])):
        #         if self.block[row][cell] == 1:  # 只检查方块的实体部分
        #             new_x = self.y + row
        #             new_y = self.x + cell
        #             # 检查是否超出边界或与已有方块重叠
        #             if (new_x >= self.secene_row or new_y < 0 or new_y >= self.width or
        #                 (new_x >= 0 and self.state[new_x][new_y] > 1)):
        #                 return True
        # return False
    
        # # 检查当前方块是否结束
        # for row in range(len(self.block)):
        #     for cell in range(len(self.block[row])):
        #         new_x = self.y + row
        #         new_y = self.x + cell 
        #         if 0 <= new_x < self.secene_row and 0 <= new_y < self.width:
        #             if self.is_block_end(self.block):
        #                 if new_x <= 4:
        #                     done = True
        # return done

        # for row_idx, row in enumerate(self.block):
        #     for col_idx, cell in enumerate(row):
        #         if cell == 1:
        #             board_x = self.x + col_idx
        #             board_y = self.y + row_idx
                    
        #             # 如果方块位置与已落地方块重叠
        #             if 0 <= board_y < self.height and 0 <= board_x < self.width:
        #                 if self.state[board_y][board_x] > 1:
        #                     return True
        # return False   
    
    """键盘控制动作"""
    def key_control(self):
        action = [0, 0, 0, 0]
        # ←键a或是左，→键或d是右，↓键或s是下，↑或w是旋转，空格是下落
        if keyboard.is_pressed('left') or keyboard.is_pressed('a'):
            action[0] = 1  # 左移
        elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):
            action[1] = 1  # 右移
        elif keyboard.is_pressed('up') or keyboard.is_pressed('w'):
            action[2] = 1  # 旋转
        elif keyboard.is_pressed('space') or keyboard.is_pressed('s') or keyboard.is_pressed('down'):
            action[3] = 1  # 硬下落
        
        return action


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
    