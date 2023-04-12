import socket
import pickle
import torch

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

# 定义服务器端口和IP地址
SERVER_HOST = '192.168.238.93'
SERVER_PORT = 8000

# 开始监听客户端连接
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((SERVER_HOST, SERVER_PORT))
    s.listen(1)
    print(f'Server listening on {SERVER_HOST}:{SERVER_PORT}...')

    while True:
        # 等待客户端连接
        conn, addr = s.accept()
        print(f'Connected by {addr}')

        # 将全局模型发送给客户端
        data = pickle.dumps(global_model.state_dict())
        conn.sendall(data)

        # 关闭连接
        conn.close()


