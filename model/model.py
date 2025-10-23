import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class RNDModel(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        # 目标网络(固定随机权重)
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # 预测网络(可训练)
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 冻结目标网络参数
        for param in self.target.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            target_features = self.target(x)
        predicted_features = self.predictor(x)
        return predicted_features, target_features


class Model(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_size=128):
        super().__init__()

        # User-defined network
        # 用户自定义网络
        # self.shared = nn.Sequential(
        #     nn.Linear(state_shape, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.LeakyReLU(),
        # )
        self.actor = nn.Sequential(
            nn.Linear(state_shape, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, action_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_shape, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.init_parameters()

    def init_parameters(self):
        # Initialize parameters
        # 参数初始化
        # for layer in self.shared:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)
        #         if layer.bias is not None:
        #             nn.init.zeros_(layer.bias)
        # nn.init.xavier_uniform_(self.actor.weight)
        # nn.init.zeros_(self.actor.bias)
        # nn.init.xavier_uniform_(self.critic.weight)
        # nn.init.zeros_(self.critic.bias)
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # feat = self.shared(x)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value