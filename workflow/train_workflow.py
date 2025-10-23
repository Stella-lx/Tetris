from feature.definition import reward_shaping
import time
import math
import os
from collections import namedtuple
from .plot import plot_and_save_scores, plot_and_save_reward, plot_and_save_loss

# 定义样本数据结构
Sample = namedtuple('Sample', [
    'obs', 'action', 'reward', 'done', 'value', 'logp'
])

def workflow(envs, agents, episodes=10000):
    # # 如果没有提供logger，创建默认的
    # if logger is None:
    #     logger = create_logger("TetrisTraining")

    # # 如果没有提供monitor，创建默认的
    # if monitor is None:
    #     monitor = create_monitor()
    try:
        # # Read and validate configuration file
        # # 配置文件读取和校验
        # usr_conf = "agent_diy/conf/train_env_conf.toml", logger
        # if usr_conf is None:
        #     logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        #     return

        env, agent = envs[0], agents[0]
        EPISODES = episodes

        # 初始化得分列表，记录训练数据
        scores = []
        average_scores = []
        average_100_scores = []
        rewards = []
        avg_rewards = []

        # 训练指标记录 - 按episode记录
        episode_policy_losses = []
        episode_value_losses = []
        episode_entropies = []
        episode_rnd_losses = []
        episode_learning_rates = []

        # # Initializing monitoring data
        # # 监控数据初始化
        # monitor_data = {
        #     "reward": 0,
        #     "diy_1": 0,
        #     "diy_2": 0,
        #     "diy_3": 0,
        #     "diy_4": 0,
        #     "diy_5": 0,
        # }
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

            # # Retrieving training metrics
            # # 获取训练中的指标
            # training_metrics = get_training_metrics()
            # if training_metrics:
            #     logger.info(f"training_metrics is {training_metrics}")

            # Reset the game and get the initial state
            # 重置游戏, 并获取初始状态
            # obs, extra_info = env.reset(usr_conf=usr_conf)
            # if extra_info["result_code"] != 0:
            #     logger.error(
            #         f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
            #     )
            #     raise RuntimeError(extra_info["result_message"])
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
                # print(state)
                # print(f"action: {action}")

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

            # 记录训练后的损失指标 (每个episode记录一次)
            episode_policy_losses.append(agent.algorithm.policy_loss)
            episode_value_losses.append(agent.algorithm.value_loss)
            episode_entropies.append(agent.algorithm.entropy)
            episode_rnd_losses.append(agent.algorithm.rnd_loss)
            episode_learning_rates.append(agent.algorithm.lr)

            # 记录得分训练数据
            score = env.score # 更新得分
            scores.append(score)
            average_scores.append(sum(scores) / len(scores))
            if (episode + 1) % 100 == 0:
                average_100_scores.append(sum(scores[-100:]) / 100)

            # 记录当前episode的奖励（不是累积奖励）
            rewards.append(episode_reward)  # 修复：使用episode_reward而不是total_rew

            # 计算平均奖励（基于所有episode的奖励）
            current_avg_reward = sum(rewards) / len(rewards) if len(rewards) > 0 else 0
            avg_rewards.append(current_avg_reward)

            # Reporting training progress
            # 上报训练进度
            episode_cnt += 1
            now = time.time()
            is_converged = win_cnt / (episode + 1) > 0.9 and episode > 2000

            if is_converged or now - last_report_monitor_time > 60:
                print(f"Episode {episode + 1}: Avg Reward = {current_avg_reward:.4f}, Episode Reward = {episode_reward:.4f}")

                # 不要重置total_rew和episode_cnt，保持累积统计
                last_report_monitor_time = now

                # The model has converged, training is complete
                # 模型收敛, 结束训练
                if is_converged:
                    print(f"Training Converged at Episode: {episode + 1}")
                    break

            # 每100个episode绘制一次图表
            if (episode + 1) % 100 == 0:
                print(f"生成训练图表 (Episode {episode + 1})")
                plot_and_save_reward(rewards, avg_rewards)
                plot_and_save_loss(episode_value_losses, episode_policy_losses, episode_entropies, episode_rnd_losses, episode_learning_rates)
                plot_and_save_scores(scores, average_scores, average_100_scores)

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

        # 最终绘制所有图表
        plot_and_save_scores(scores, average_scores, average_100_scores)
        plot_and_save_reward(rewards, scores)
        plot_and_save_loss(episode_value_losses, episode_policy_losses, episode_entropies, episode_rnd_losses, episode_learning_rates)

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        raise RuntimeError(f"workflow error: {e}")
