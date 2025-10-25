import numpy as np
from conf.conf import Config
from collections import defaultdict

pos_dict = defaultdict(int)

_last_pos = None

def reward_shaping(done, state, down, clean_line):


    reward = 0

    # 游戏结束惩罚 
    if done:
        reward -= 100

    # 消除行的额外奖励
    if clean_line > 0:
        reward += clean_line * 20  # 额外的消除行奖励
        
    if down:
        reward += 1

    return reward

