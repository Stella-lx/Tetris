import sys
import torch
import os
import time
from datetime import datetime

# 添加项目路径
sys.path.append('.')

# 导入项目组件
from env.env import Env
from agent import Agent
from workflow.train_workflow import workflow

def find_specific_model(model_name="model.ckpt-179395.pth"):
    """查找指定的模型文件"""
    models_dir = "models"
    
    # 遍历所有子目录查找指定模型
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file == model_name:
                model_path = os.path.join(root, file)
                print(f"找到指定模型: {model_path}")
                return model_path
    
    print(f"未找到指定模型: {model_name}")
    return None

def load_specific_model(agent, model_path):
    """加载指定的模型"""
    try:
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
            
        state_dict = torch.load(model_path, map_location=agent.device)
        agent.algorithm.model.load_state_dict(state_dict)
        print(f"成功加载模型: {model_path}")
        return True
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False

def test_model(env, agent, episodes=10):
    """测试模型性能"""
    print(f"\n开始测试模型，将运行 {episodes} 个episode...")
    print("=" * 60)
    
    total_score = 0
    total_reward = 0
    win_count = 0
    
    for episode in range(episodes):
        state = env.reset()
        obs_data = agent.observation_process(env, state)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.exploit(obs_data)
            env.render(state) 
            print(env.block,env.next_block)
            next_state, done, reward, score = env.step(action)
            print(f"奖励: {reward}")
            print(f"得分: {env.score}, 游戏结束: {done}")
            state = next_state
            if done:
                break 
            time.sleep(0.1)
        
        score = env.score
        total_score += score
        total_reward += episode_reward
        
        if score > 0:  # 假设得分大于0表示胜利
            win_count += 1
        
        print(f"Episode {episode+1}: 得分 = {score}, 奖励 = {episode_reward:.2f}")
    
    avg_score = total_score / episodes
    avg_reward = total_reward / episodes
    win_rate = win_count / episodes * 100
    
    print("\n测试结果:")
    print(f"平均得分: {avg_score:.2f}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"胜率: {win_rate:.2f}%")
    print("=" * 60)

def main():
    """主函数 - 加载指定模型并测试"""
    print("俄罗斯方块PPO强化学习模型测试")
    print("=" * 60)

    # 设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        # 创建环境
        print("创建环境...")
        env = Env(height=12, width=10)

        # 创建智能体
        print("创建智能体...")
        agent = Agent(device=device)

        # 查找并加载指定模型
        model_name = "model.ckpt-179395.pth"
        model_path = find_specific_model(model_name)
        
        if model_path:
            if load_specific_model(agent, model_path):
                print(f"成功加载模型: {model_name}")
                # 进行模型测试
                test_model(env, agent, episodes=10)
            else:
                print(f"加载模型失败: {model_name}")
        else:
            print(f"未找到模型: {model_name}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()