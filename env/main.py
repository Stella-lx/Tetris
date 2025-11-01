from env import Env
import time

env = Env()
state = env.reset()
clean = 0
while True:
    action = env.key_control()
    # next_state, reward, done, log_prob, value = env.step(action)
    env.render(state) 
    print(env.block,env.next_block)
    next_state, done, reward, score = env.step(action)
    print(f"奖励: {reward}")
    print(f"得分: {env.score}, 游戏结束: {done}")
    clean += env.cleans
    print(f"cleans: {env.cleans}")
    print(f"clean_num: {clean}")
    state = next_state
    if done:
        break 
    time.sleep(0.1)
