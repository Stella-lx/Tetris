import numpy as np
import os
from conf.conf import Config
from algorithm.algorithm import Algorithm
import torch
from env.env import Env
import numpy as np

class Agent():
    def __init__(self, device=None) -> None:

        # Initialize parameters
        # 参数初始化
        self.observation_size = Config.OBSERVATION_SHAPE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES
        self.algorithm = Algorithm(observation_size = Config.OBSERVATION_SHAPE, 
                                    action_size = Config.ACTION_SIZE, 
                                    gamma = Config.GAMMA, 
                                    learning_rate = Config.LEARNING_RATE, 
                                    lam=Config.LAMBDA,
                                    clip_coef=Config.CLIP_COEF,
                                    epochs=Config.EPOCHS,
                                    batch_size=Config.BATCH_SIZE,
                                    device = device)
        self.device = device if device else torch.device("cpu")

        # 暂时禁用预训练模型加载
        if Config.LOAD_MODEL_ID:
            print("Load pre-trained model.")
            self.__load_model(
                path="/data/projects/gorge_walk_v2/ckpt",
                id=Config.LOAD_MODEL_ID,
            )

        # super().__init__(agent_type, device, logger, monitor)

    def predict(self, observation):
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.device).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.algorithm.model(observation_tensor)  # 假设 model 返回 (logits, value)

            # 构建动作分布
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            act = action.item()

        return act, value.item(), logp.item()

    def exploit(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).to(self.device).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.algorithm.model(state)
            act_data = torch.argmax(logits, dim=-1).item()

        return act_data

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env, state):
        """
        [0:159]: 全局状态,16*10维
        [160: 160]: 智能体当前位置横坐标归一化
        [161: 161]: 智能体当前位置纵坐标归一化
        [162: 168]: 下一个方块类型
        [169: 175]: 当前方块类型
        [176: 179]: 当前方块旋转类型
        """

        # 特征#1: 全局状态,16*10维
        array_state = np.array(state)
        # print(f'env.state',state)
        global_state = array_state.flatten()
        normalized_global_state = (global_state / 8.0).astype(np.float32)

        # pos = [env.x, env.y]
        # # Feature #2: Current state of the agent (1-dimensional representation)
        # # 特征#2: 智能体当前 state (1维表示)
        # state = [int(pos[0] * 16 + pos[1] + 1) / (16 * 10)]  # Normalized position in the grid
        # # Feature #3: One-hot encoding of the agent's current position
        # # 特征#3: 智能体当前位置信息的 one-hot 编码
        # pos_row = [0] * 10
        # pos_row[pos[0]] = 1
        # pos_col = [0] * 16
        # pos_col[pos[1]] = 1

        # 特征#2: 智能体当前横坐标归一化
        normalized_x = [env.x / env.width]
        # print(f'env.x',env.x)

        # 特征#3: 智能体当前纵坐标归一化
        normalized_y = [env.y / (env.height + 4)]

        # 特征#4: 下一个方块类型
        next_block_type = [1 if i == env.next_block_index else 0 for i in range(7)]
        # print(f'env.next_block_index',env.next_block_index)

        # 特征#5: 当前方块类型
        current_block_type = [1 if i == env.block_index else 0 for i in range(7)]

        # 特征#6: 当前方块旋转类型
        current_rotation_state = [1 if i == env.rotation else 0 for i in range(4)]
        
        # 多个特征向量​​水平拼接，180维
        feature = np.concatenate(
            [
                normalized_global_state, #160维
                normalized_x, #1维
                normalized_y, # 1维
                next_block_type, # 7维
                current_block_type, # 7维
                current_rotation_state, # 4维
            ]
        ) 
        # print(feature.shape)
        return feature

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        if path is None:
            path = "models"

        # 确保目录存在
        import os
        os.makedirs(path, exist_ok=True)

        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        torch.save(self.algorithm.model.state_dict(), model_file_path)
        print(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        self.__load_model(path, id)

    def __load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        try:
            state_dict = torch.load(
                model_file_path,
                map_location=self.device
            )
            self.algorithm.model.load_state_dict(state_dict)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            print(f"File {model_file_path} not found")
