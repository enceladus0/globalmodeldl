import socket
import torch
import pickle

# 从服务器接收全局模型并应用到本地模型
def receive_global_model():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('192.168.117.93', 8000))
        data = s.recv(4096)
        global_model_params = pickle.loads(data)
        local_model.load_state_dict(global_model_params)
        print('Received global model from server successfully!')

# 定义本地模型
class LocalModel(torch.nn.Module):
    def __init__(self):
        super(LocalModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

local_model = LocalModel()

# 从服务器接收全局模型并应用到本地模型
receive_global_model()


