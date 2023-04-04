import socket
import torch
import pickle

# 定义全局模型
class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

global_model = GlobalModel()

# 初始化全局模型参数
for param in global_model.parameters():
    param.data.uniform_(-1, 1)

# 将全局模型发送给客户端
def send_global_model():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('192.168.117.93', 8000))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            data = pickle.dumps(global_model.state_dict())
            conn.sendall(data)

