import socket
import pickle
import torch

# 定义服务器地址和端口号
SERVER_IP = '192.168.117.93'
SERVER_PORT = 8000

# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 定义客户端接收数据的缓冲区大小
BUFFER_SIZE = 4096

# 创建客户端套接字并连接到服务器
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))

# 接收服务器发送的数据
data = b''
while True:
    packet = client_socket.recv(BUFFER_SIZE)
    if not packet:
        break
    data += packet

# 反序列化数据并更新本地模型
model_dict = pickle.loads(data)
model = Net()
model.load_state_dict(model_dict)

# 显示提示信息
print("Client: Received global model from server successfully.")



