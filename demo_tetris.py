
"""
展示训练效果
"""

import os
import sys
import time
import torch

# 添加项目路径
sys.path.append('.')

from env.env import Env
from agent import Agent

def show_tetris_demo():
    """展示俄罗斯方块训练效果"""
    print(" 俄罗斯方块PPO训练效果展示")
    print("=" * 40)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 查找最新模型
    print(" 查找训练模型")
    models_dir = "models"

    if not os.path.exists(models_dir):
        print("模型目录不存在，请先进行训练")
        return

    # 查找最新的模型文件（支持两种格式）
    model_path = None
    model_name = None

    # 查找中断的模型（在子目录中）
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path) and 'interrupted' in item:
            # 查找该目录下的 model.ckpt-*.pth 文件
            for file in os.listdir(item_path):
                if file.endswith('.pth') and 'model.ckpt-' in file:
                    model_path = os.path.join(item_path, file)
                    model_name = f"{item}/{file}"
                    break
            if model_path:
                break

    # 如果没找到中断的模型，查找直接在models目录下的模型文件
    if not model_path:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth') and 'model.ckpt-' in f]
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            model_path = os.path.join(models_dir, model_files[0])
            model_name = model_files[0]

    if not model_path:
        print("没有找到训练模型")
        return

    print(f"使用模型: {model_name}")
    
    # 加载智能体
    print("加载训练模型...")
    try:
        agent = Agent(device=device)
        state_dict = torch.load(model_path, map_location=device)
        agent.algorithm.model.load_state_dict(state_dict)
        agent.algorithm.model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建环境
    print("创建游戏环境...")
    env = Env(height=12, width=10)
    print("环境创建成功")
    
    # 开始演示
    print("\n 开始游戏演示...")
    print("=" * 40)
    print("提示: 按 Ctrl+C 可以随时停止演示")
    
    try:
        game_count = 0
        while True:
            game_count += 1
            print(f"\n🎮 第 {game_count} 局游戏")
            print("-" * 30)
            
            # 重置环境
            state = env.reset()
            obs_data = agent.observation_process(env, state)
            
            step_count = 0
            max_steps = 1000  # 限制最大步数
            logp_list = []
            
            while step_count < max_steps:
                # 智能体预测动作
                action, value, logp = agent.predict(obs_data)
                
                # 执行动作
                next_state, done, reward, score = env.step(action)
                obs_data = agent.observation_process(env, state)
                
                step_count += 1
                
                # 渲染游戏画面
                env.render(state)
                state = next_state
                
                # 显示游戏信息
                action_names = ["左移", "右移", "下移", "旋转"]
                action_name = action_names[action] if action < 4 else f"动作{action}"



                print(f"步骤: {step_count:3d} | 动作: {action_name:4s} | 得分: {env.score:4d} | 概率分布：{logp}")
                logp_list.append(logp)
                # 控制显示速度
                time.sleep(0.1)
                
                # 检查游戏是否结束
                if done:
                    print(f"\n 游戏结束!")
                    print(f"   最终得分: {env.score}")
                    print(f"   总步数: {step_count}")
                    print(f"   消除行数: {env.cleans}")
                    break

            print(logp_list)
            
            if step_count >= max_steps:
                print(f"\n 达到最大步数限制 ({max_steps})")
                print(f"   当前得分: {env.score}")
            
            # 询问是否继续
            print("\n是否继续下一局? (按 Enter 继续，输入 'q' 退出)")
            user_input = input().strip().lower()
            if user_input == 'q':
                break
                
    except KeyboardInterrupt:
        print("\n\n  演示被用户中断")
    
    print(f"\n 训练结果演示结束,总共进行了 {game_count} 局游戏")

if __name__ == "__main__":
    show_tetris_demo()
