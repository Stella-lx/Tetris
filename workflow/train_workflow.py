from feature.definition import reward_shaping
import time
import math
import os
from collections import namedtuple
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

# 定义样本数据结构
Sample = namedtuple('Sample', [
    'obs', 'action', 'reward', 'done', 'value', 'logp'
])

# 初始化TensorBoard写入器（自动创建时间戳目录）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/ppo_tetris_{timestamp}"
writer = SummaryWriter(log_dir)

def workflow(envs, agents, episodes=10000):

    try:

        env, agent = envs[0], agents[0]
        EPISODES = episodes

        # 初始化列表，记录训练数据
        scores = []
        average_scores = []
        average_100_scores = []
        rewards = []
        avg_rewards = []

        last_report_monitor_time = time.time()

        print("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t

        total_rew, episode_cnt, win_cnt = (
            0,
            0,
            0,
        )
        episode_reward = 0  # 当前episode的奖励

        max_step = 100

        episode = 0
        while True:

            step_cnt = 0
            state = env.reset()
            episode_reward = 0  # 重置当前episode奖励

            # Feature processing
            # 特征处理
            obs_data = agent.observation_process(env, state)

            # Task loop
            # 任务循环
            done = False
            trajectory_buffer = []
            while not done:
                # Agent performs inference to obtain the predicted action for the next frame
                # Agent 进行推理, 获取下一帧的预测动作
                # print(obs_data)
                action, value, logp = agent.predict(obs_data)

                # Interact with the environment, perform actions, and obtain the next state
                # 与环境交互, 执行动作, 获取下一步的状态
                next_state, is_done, reward, scoress = env.step(action)

                # Feature processing
                # 特征处理
                _obs_data = agent.observation_process(env, next_state)

                # 计算奖励 - 结合环境奖励和奖励塑形
                shaped_reward = reward_shaping(is_done, state, env.is_block_end(env.block), env.cleans)
                reward = reward + shaped_reward  # 组合环境奖励和塑形奖励
            
                # Determine over and update the win count
                # 判断结束, 并更新胜利次数
                if step_cnt < max_step:
                    done = is_done
                else:
                    done = True
                
                if done and step_cnt == max_step:
                    win_cnt += 1
                # print(f"step_cnt: {step_cnt}, done: {done}, is_done: {is_done}, max_step: {max_step}")
                # 更新步数
                step_cnt += 1

                state = next_state

                # Updating data and generating frames for training
                # 数据更新, 生成训练需要的 frame
                sample = Sample(
                    obs=obs_data,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value,
                    logp=logp
                )
                trajectory_buffer.append(
                    sample
                )


                # Update total reward and observation
                # 更新总奖励和观察
                total_rew += reward
                episode_reward += reward  # 累积当前episode奖励
                obs_data = _obs_data

            # 训练智能体
            agent.learn(trajectory_buffer)

            # Tensorboard 记录
            writer.add_scalar('Loss/Policy Loss', agent.algorithm.policy_loss, episode)
            writer.add_scalar('Loss/Value Loss', agent.algorithm.value_loss, episode)
            writer.add_scalar('Loss/Entropy', agent.algorithm.entropy, episode)
            writer.add_scalar('Loss/RND Loss', agent.algorithm.rnd_loss, episode)
            writer.add_scalar('Learning Rate', agent.algorithm.lr, episode)
            writer.add_scalar('Loss/Total Loss', agent.algorithm.loss, episode)

            # 计算得分训练数据
            score = env.score # 更新得分
            scores.append(score)
            average_scores.append(sum(scores) / len(scores))
            if (episode + 1) % 100 == 0:
                average_100_scores.append(sum(scores[-100:]) / 100)

            # 记录当前episode的奖励（不是累积奖励）
            rewards.append(episode_reward) 
            # 计算平均奖励（基于所有episode的奖励）
            current_avg_reward = sum(rewards) / len(rewards) if len(rewards) > 0 else 0
            avg_rewards.append(current_avg_reward)

            # Tensorboard记录
            writer.add_scalar('reward/Episode Reward', episode_reward, episode)
            writer.add_scalar('reward/Average Reward', current_avg_reward, episode)
            writer.add_scalar('reward/Total Reward', total_rew, episode)
            writer.add_scalar('Score/Score', score, episode)
            writer.add_scalar('Score/Average Score', average_scores[-1], episode)
            if average_100_scores:  # 确保列表不为空
                writer.add_scalar('Score/Average 100 Score', average_100_scores[-1], episode)
            else:
                writer.add_scalar('Score/Average 100 Score', 0, episode)
            
            # Reporting training progress
            # 上报训练进度
            episode_cnt += 1
            now = time.time()
            is_converged = win_cnt / (episode + 1) > 0.9 and episode > 2000

            if is_converged or now - last_report_monitor_time > 60:
                print(f"Episode {episode + 1}: Avg Reward = {current_avg_reward:.4f}, Episode Reward = {episode_reward:.4f}")
                last_report_monitor_time = now

                # The model has converged, training is complete
                # 模型收敛, 结束训练
                if is_converged:
                    print(f"Training Converged at Episode: {episode + 1}")
                    break

            # Saving the model every 5 minutes
            # 每5分钟保存一次模型
            if now - last_save_model_time > 100000:
                print(f"Saving Model at Episode: {episode + 1}")
                agent.save_model(id = episode + 1)
                last_save_model_time = now
            episode += 1

        end_t = time.time()
        print(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
        agent.episodes = episode + 1

        # model saving
        # 保存模型
        agent.save_model()

        # 训练结束关闭TensorBoard
        writer.close()

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise RuntimeError(f"workflow error: {e}")
