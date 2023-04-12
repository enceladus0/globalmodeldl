import socket
import pickle
import torch
import time

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

# 定义服务器端口和IP地址
SERVER_HOST = '192.168.238.93'
SERVER_PORT = 1234

# 尝试连接服务器
connected = False
while not connected:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((SERVER_HOST, SERVER_PORT))
            print(f'Connected to {SERVER_HOST}:{SERVER_PORT}...')
            connected = True
            # 接收来自服务器的全局模型
            data = s.recv(10240)
            state_dict = pickle.loads(data)
            global_model = GlobalModel()
            global_model.load_state_dict(state_dict)
            print("Received global model from server.")
            # 关闭连接
            s.shutdown(socket.SHUT_RDWR)
            s.close()
    except ConnectionRefusedError:
        print("Server not available, waiting...")
        time.sleep(10)

# 进行模型推理或训练等操作
# ...




