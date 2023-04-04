import torch
import torch.nn as nn
import torch.optim as optim

# 定义全局模型
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

global_model = GlobalModel()

# 初始化全局模型参数
for param in global_model.parameters():
    param.data.uniform_(-1, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(global_model.parameters(), lr=0.01)

# 将全局模型下发给客户端
def send_global_model():
    return global_model.state_dict()
