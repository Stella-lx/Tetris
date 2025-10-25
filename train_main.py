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


def main():
    """主训练函数"""

    print("俄罗斯方块PPO强化学习训练 - 改进版")
    print("=" * 60)

    # 设置计算设备
    if torch.cuda.is_available():
        print(f"CUDA 可用，使用的 GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA 不可用，使用 CPU。")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建必要目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        # 创建环境
        print("创建环境...")
        env = Env(height=12, width=10)
        
        # 创建智能体
        print("创建智能体...")
        agent = Agent(device=device)
        
        # 准备训练数据
        envs = [env]
        agents = [agent]

        # 初始化得分
        score = 0 

        print(f" 开始训练")
        print("-" * 60)
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 启动训练工作流程
        workflow(envs, agents)
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        print(f"\n 训练完成! 总用时: {training_time:.2f}秒")
        # logger.info(f"训练完成，总用时: {training_time:.2f}秒")
        
        # 保存最终模型
        final_model_path = f"models/tetris_ppo_final_{timestamp}.pth"
        agent.save_model(final_model_path)
        print(f" 最终模型已保存: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n 训练被用户中断")
        
        # 保存中断时的模型
        interrupted_model_path = f"models/tetris_ppo_interrupted_{timestamp}.pth"
        try:
            agent.save_model(interrupted_model_path)
            print(f"中断模型已保存: {interrupted_model_path}")
        except Exception as e:
            print(f"保存中断模型失败: {e}")

if __name__ == "__main__":
    main()
