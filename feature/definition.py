import numpy as np
from conf.conf import Config
from collections import defaultdict

pos_dict = defaultdict(int)

_last_pos = None

def reward_shaping(done, clean_line, x, y, block, blocks, locations, is_done):


    reward = 0

    # 游戏结束惩罚 
    if done:
        reward -= 5

    # 消除行的额外奖励
    if clean_line > 0:
        reward += clean_line * 5  # 额外的消除行奖励

    #重复步的惩罚
    if not is_done:
        if blocks and locations:
            for i in range(len(blocks)):
                if locations[i] == (x, y) and blocks[i] == block:
                    reward -= 0.02
                    break
        
    return reward

