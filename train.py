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


def find_latest_interrupted_model():
    """查找最新的中断模型"""
    models_dir = "models"

    if not os.path.exists(models_dir):
        return None

    # 查找中断的模型目录
    interrupted_dirs = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and 'interrupted' in item:
            interrupted_dirs.append((item, item_path))

    if not interrupted_dirs:
        return None

    # 按时间戳排序，获取最新的
    interrupted_dirs.sort(reverse=True)
    latest_dir_name, latest_dir_path = interrupted_dirs[0]

    # 查找该目录下的 model.ckpt-*.pth 文件
    for file in os.listdir(latest_dir_path):
        if file.endswith('.pth') and 'model.ckpt-' in file:
            model_path = os.path.join(latest_dir_path, file)
            return model_path, latest_dir_name

    return None


def load_interrupted_model(agent, model_path):
    """加载中断的模型"""
    try:
        state_dict = torch.load(model_path, map_location=agent.device)
        agent.algorithm.model.load_state_dict(state_dict)
        print(f"成功加载中断模型: {model_path}")
        return True
    except Exception as e:
        print(f"加载中断模型失败: {e}")
        return False


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

        # 检查是否有中断的模型
        print("\n 检查中断的模型...")
        interrupted_model_info = find_latest_interrupted_model()

        if interrupted_model_info:
            model_path, model_name = interrupted_model_info
            print(f"找到中断模型: {model_name}")

            # 询问用户是否加载
            user_input = input("是否从中断的模型继续训练? (y/n, 默认: y): ").strip().lower()
            if user_input != 'n':
                if load_interrupted_model(agent, model_path):
                    print("将从中断的模型继续训练")
                else:
                    print("将从新模型开始训练")
            else:
                print("将从新模型开始训练")
        else:
            print("没有找到中断的模型，将从新模型开始训练")

        # 准备训练数据
        envs = [env]
        agents = [agent]

        # 初始化得分
        score = 0

        print(f"\n 开始训练 (目标回合数: {agents[0].episodes})...")
        print("-" * 60)

        # 记录训练开始时间
        start_time = time.time()

        # 启动训练工作流程
        workflow(envs, agents)

        # 计算训练时间
        training_time = time.time() - start_time

        print(f"\n 训练完成 总用时: {training_time:.2f}秒")
        # logger.info(f"训练完成，总用时: {training_time:.2f}秒")

        # 保存最终模型
        final_model_path = f"models/tetris_ppo_final_{timestamp}.pth"
        agent.save_model(final_model_path)
        print(f"最终模型已保存: {final_model_path}")

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
