import matplotlib.pyplot as plt

def plot_and_save_scores(scores, average_scores,average_100_scores):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(18, 6))
    
    # 绘制得分曲线
    plt.subplot(1, 3, 1)
    plt.plot(scores, 'b-', alpha=0.6, label='单局得分')
    plt.xlabel('回合数')
    plt.ylabel('得分')
    plt.title('俄罗斯方块PPO训练得分曲线')
    plt.grid(True)
    plt.legend()
    
    # 绘制平均得分曲线
    plt.subplot(1, 3, 2)
    plt.plot(average_scores, 'r-', label='平均得分')
    plt.xlabel('回合数')
    plt.ylabel('平均得分')
    plt.title('累计平均得分变化曲线')
    plt.grid(True)
    plt.legend()
        
    if average_100_scores:
        x = [i * 100 for i in range(1, len(average_100_scores) + 1)]
        plt.subplot(1, 3, 3)
        plt.plot(x, average_100_scores, 'g-', label='每100回合平均得分')
        plt.xlabel('回合数')
        plt.ylabel('平均得分')
        plt.title('每100回合平均得分变化曲线')
        plt.grid(True)
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('tetris_ppo_training_results.png')
    print(f"训练结果图像已保存为 'tetris_ppo_training_results.png'")
    plt.close('all')  # 添加这一行

def plot_and_save_reward(reward, avg_reward):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(18, 6))
    
    # 绘制曲线
    plt.subplot(1, 3, 1)
    plt.plot(reward, 'b-', alpha=0.6, label='reward')
    plt.xlabel('回合数')
    plt.ylabel('reward')
    plt.title('俄罗斯方块PPO训练reward曲线')
    plt.grid(True)
    plt.legend()
    
        # 绘制平均得分曲线
    plt.subplot(1, 3, 2)
    plt.plot(avg_reward, 'r-', label='平均reward')
    plt.xlabel('回合数')
    plt.ylabel('平均reward')
    plt.title('累计平均reward变化曲线')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('tetris_ppo_training_reward.png')
    print(f"训练结果图像已保存为 'tetris_ppo_training_reward.png'")
    plt.close('all')  # 添加这一行

def plot_and_save_loss(value_losses, policy_losses, entropies, rnd_losses=None, learning_rates=None):
    """
    绘制训练损失曲线
    参数:
    - value_losses: 值函数损失列表 (按episode)
    - policy_losses: 策略损失列表 (按episode)
    - entropies: 熵损失列表 (按episode)
    - rnd_losses: RND损失列表 (按episode, 可选)
    - learning_rates: 学习率列表 (按episode, 可选)
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 根据是否有额外数据决定子图布局
    if rnd_losses is not None and learning_rates is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPO训练损失和指标曲线', fontsize=16, fontweight='bold')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('PPO训练损失曲线', fontsize=16, fontweight='bold')
        axes = [axes]  # 统一为二维数组格式

    episodes = range(1, len(value_losses) + 1)

    # 第一行：主要损失指标
    # 值函数损失
    ax1 = axes[0][0] if len(axes) > 1 else axes[0]
    ax1.plot(episodes, value_losses, 'b-', alpha=0.8, linewidth=2, label='Value Loss')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Value Loss')
    ax1.set_title('值函数损失')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 策略损失
    ax2 = axes[0][1] if len(axes) > 1 else axes[1]
    ax2.plot(episodes, policy_losses, 'r-', alpha=0.8, linewidth=2, label='Policy Loss')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Policy Loss')
    ax2.set_title('策略损失')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 熵
    ax3 = axes[0][2] if len(axes) > 1 else axes[2]
    ax3.plot(episodes, entropies, 'g-', alpha=0.8, linewidth=2, label='Entropy')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy')
    ax3.set_title('策略熵')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 第二行：额外指标 (如果有的话)
    if len(axes) > 1 and rnd_losses is not None and learning_rates is not None:
        # RND损失
        ax4 = axes[1][0]
        ax4.plot(episodes, rnd_losses, 'm-', alpha=0.8, linewidth=2, label='RND Loss')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('RND Loss')
        ax4.set_title('RND好奇心损失')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # 学习率
        ax5 = axes[1][1]
        ax5.plot(episodes, learning_rates, 'orange', alpha=0.8, linewidth=2, label='Learning Rate')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Learning Rate')
        ax5.set_title('学习率变化')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # 综合损失趋势
        ax6 = axes[1][2]
        # 计算总损失 (归一化后相加)
        if len(value_losses) > 0 and len(policy_losses) > 0:
            import numpy as np
            # 简单归一化
            norm_value = np.array(value_losses) / (np.max(value_losses) + 1e-8)
            norm_policy = np.array(policy_losses) / (np.max(policy_losses) + 1e-8)
            total_loss = norm_value + norm_policy

            ax6.plot(episodes, total_loss, 'purple', alpha=0.8, linewidth=2, label='Normalized Total Loss')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Normalized Loss')
            ax6.set_title('归一化总损失趋势')
            ax6.grid(True, alpha=0.3)
            ax6.legend()

    plt.tight_layout()
    plt.savefig('tetris_ppo_training_loss.png', dpi=300, bbox_inches='tight')
    print(f"训练损失图像已保存为 'tetris_ppo_training_loss.png'")
    plt.close('all')  # 添加这一行

