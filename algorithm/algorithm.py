# RND 好奇心激励
import numpy as np
from model.model import Model, RNDModel
import torch
import torch.nn.functional as F
import random
from collections import namedtuple
from conf.conf import Config

class Algorithm:
    def __init__(self,
                 observation_size, 
                 action_size, 
                 gamma, 
                 learning_rate, 
                 lam,
                 clip_coef,
                 epochs,
                 batch_size,
                 device = torch.device("cpu")):
        # 基础参数设置
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lam = lam
        self.epochs = epochs
        self.batch_size = batch_size

        # 训练指标初始化，这些变量用于记录训练过程中的关键指标，方便监控和调试
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.entropy = 0.0
        # 算法超参数设置
        self.max_grad_norm = Config.MAX_GRAD_NORM 
        self.clip_eps = clip_coef
        
        # 设备与环境设置
        self.device = device
        self.observation_size = observation_size
        self.action_size = action_size
        #模型初始化
        self.model = Model(
            state_shape=observation_size, 
            action_shape=action_size, 
            hidden_size=Config.HIDDEN_SIZE, 
        ).to(self.device) 
        self.rnd_model = RNDModel(observation_size, Config.HIDDEN_SIZE).to(device) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5) 
        self.rnd_optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr=self.learning_rate)

        # RND相关参数
        self.rnd_coef = Config.RND_COEF 
        self.rnd_loss = 0.0 

    def compute_intrinsic_reward(self, states):
        """计算内在奖励"""
        predicted, target = self.rnd_model(states)
        intrinsic_reward = F.mse_loss(predicted, target, reduction='none').mean(-1)
        # self.logger.info(f"intrinsic_reward: {intrinsic_reward.shape}")
        rnd_reward_value = intrinsic_reward.mean() * self.rnd_coef
        # print(f"rnd_reward: {rnd_reward_value}")

        # # 如果有可视化器，更新数据
        # if hasattr(self, 'visualizer') and self.visualizer is not None:
        #     try:
        #         self.visualizer.update(float(rnd_reward_value))
        #     except Exception as e:
        #         pass  # 忽略可视化错误，不影响训练

        return intrinsic_reward.detach()
        
    def update_rnd(self, states):
        """更新RND预测网络"""
        predicted, target = self.rnd_model(states)
        loss = F.mse_loss(predicted, target)
        
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        
        self.rnd_loss = loss.item()
        
    def learn(self, list_sample_data):
        # print(f"训练数据样本数: {len(list_sample_data)}")
        # 解包
        states = torch.stack([torch.tensor(sample.obs, dtype=torch.float32) for sample in list_sample_data]).to(self.device) # [N, state_size]
        actions = torch.tensor([sample.action for sample in list_sample_data], dtype=torch.long).to(self.device) # [N]
        rewards = [sample.reward for sample in list_sample_data]
        dones = [sample.done for sample in list_sample_data]
        old_values = torch.tensor([sample.value for sample in list_sample_data], dtype=torch.float32).to(self.device)
        logps_old = torch.tensor([sample.logp for sample in list_sample_data], dtype=torch.float32).to(self.device)

        # 计算GAE
        returns, advantages = self.compute_gae(rewards, old_values.tolist(), dones)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 1e-8防止标准差为0，分母为0

        # 优势函数标准化，增加数值稳定性检查
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:  # 只有当标准差足够大时才进行标准化
                advantages = (advantages - advantages.mean()) / adv_std
            else:
                advantages = advantages - advantages.mean()  # 只进行中心化
        # 如果只有一个样本，不进行标准化

        # 训练模型
        dataset_size = len(states)


        for _ in range(self.epochs):
            indices = list(range(dataset_size))
            random.shuffle(indices)
            for i in range(0, dataset_size, self.batch_size):
                idx = indices[i:i + self.batch_size]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                batch_values = old_values[idx]
                batch_logps_old = logps_old[idx]

                intrinsic_reward = self.compute_intrinsic_reward(batch_states)
                batch_returns = batch_returns + self.rnd_coef * intrinsic_reward
                # print(f"batch_returns: {batch_returns}")

                logits, values = self.model(batch_states)
                # 动作的原始分数（未归一化），每个动作对应一个分数 [batch_size, action_dim]，状态价值估计，形状 [batch_size, 1]或 [batch_size]

                # # 检查logits是否包含NaN或Inf
                # if torch.isnan(logits).any() or torch.isinf(logits).any():
                #     print(f"警告: 检测到NaN或Inf的logits值")
                #     print(f"logits: {logits}")
                #     # 重置logits为小的随机值
                #     logits = torch.randn_like(logits) * 0.01

                # 创建动作分布
                dist = torch.distributions.Categorical(logits=logits)
                # 计算对数概率
                logps = dist.log_prob(batch_actions)
                # 计算策略熵
                entropy = dist.entropy()

                # 策略损失计算
                # 重要性采样比率
                ratio = torch.exp(logps - batch_logps_old) # 衡量新旧策略差异 # 使用对数域计算避免除零错误
                # 未裁剪目标
                surr1 = ratio * batch_advantages
                # 裁剪目标
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                # 策略损失
                policy_loss = -torch.min(surr1, surr2).mean()

                # 值函数损失计算
                # 确保张量形状匹配
                values_flat = values.squeeze()
                returns_flat = batch_returns.squeeze()

                # 确保两个张量都是1维的，且形状相同
                if values_flat.dim() == 0:  # 如果是标量，转换为1维张量
                    values_flat = values_flat.unsqueeze(0)
                if returns_flat.dim() == 0:
                    returns_flat = returns_flat.unsqueeze(0)

                # 裁剪值函数损失（可选）
                if Config.CLIP_VLOSS:
                    # 计算未裁剪损失
                    value_loss_unclipped = F.mse_loss(values_flat, returns_flat)

                    # 创建裁剪值,限制新旧预测的差异，防止值函数突变
                    batch_values_flat = batch_values.squeeze()
                    if batch_values_flat.dim() == 0:
                        batch_values_flat = batch_values_flat.unsqueeze(0)

                    values_clipped = batch_values_flat + torch.clamp(values_flat - batch_values_flat, -self.clip_eps, self.clip_eps)
                    # 裁剪损失计算
                    value_loss_clipped = F.mse_loss(values_clipped, returns_flat)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                # 标准值函数损失
                else:
                    value_loss = F.mse_loss(values_flat, returns_flat)
                loss = policy_loss + Config.VF_COEF * value_loss - 0.01 * entropy.mean()

                # # 检查损失是否为NaN
                # if torch.isnan(loss) or torch.isinf(loss):
                #     print(f"警告: 检测到NaN或Inf损失值: {loss.item()}")
                #     print(f"policy_loss: {policy_loss.item()}, value_loss: {value_loss.item()}, entropy: {entropy.mean().item()}")
                #     continue  # 跳过这个批次的更新

                # 优化步骤
                # 梯度清零
                self.optimizer.zero_grad()
                # 反向传播
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # 计算所有超参数的梯度范数，如果超过阈值，等比例缩放

                # 梯度参数更新
                self.optimizer.step()

        # RND模型更新 - 使用最后的批次状态或全部状态
        self.update_rnd(batch_states)

        # 训练指标记录 
        self.policy_loss = policy_loss.item()
        self.value_loss = value_loss.item()
        self.loss = loss.item()
        self.entropy = entropy.mean().item()
        self.lr = self.optimizer.param_groups[0]['lr']

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0] # 在 value 列表末尾添加一个0，用于处理最后一个时间步
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t] 
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return returns, advantages