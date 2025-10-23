import numpy as np
from conf.conf import Config
from collections import defaultdict

pos_dict = defaultdict(int)

_last_pos = None

def reward_shaping(done, state, down, clean_line):
    """
    俄罗斯方块的奖励塑形函数
    简化版本，主要提供正向激励
    """
    # 基础奖励 - 专注于正向激励
    reward = 0

    # 存活奖励 - 鼓励智能体活得更久
    if done:
        reward -= 100 # 每步存活奖励

    # 消除行的额外奖励
    if clean_line > 0:
        reward += clean_line * 10  # 额外的消除行奖励
        
    if down:
        reward += 1


    return reward

