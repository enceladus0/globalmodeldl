import torch
import torch.nn as nn

# 定义本地模型
class LocalModel(nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

local_model = LocalModel()

# 从服务器接收全局模型并应用到本地模型
def receive_global_model(global_model_params):
    local_model.load_state_dict(global_model_params)
