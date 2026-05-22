import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import numpy as np

# 1. Bộ nhớ đệm (Replay Buffer) - Lưu trữ trải nghiệm
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)), 
                torch.LongTensor(action), 
                torch.FloatTensor(reward), 
                torch.FloatTensor(np.array(next_state)), 
                torch.FloatTensor(done))

    def __len__(self):
        return len(self.buffer)

# 2. Mạng Q-Network (Như đã tối ưu ở prompt trước)
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    print("--- Đang khởi tạo Agent ---")
    state_dim = 2
    action_dim = 6
    
    model = QNet(state_dim, action_dim)
    
    # Bước này có thể hơi lâu ở lần đầu (khoảng 30-60s)
    print("Đang biên dịch mô hình (torch.compile)... Vui lòng đợi...")
    compiled_model = torch.compile(model)
    
    # Test thử 1 vòng lặp để kích hoạt biên dịch
    dummy_input = torch.randn(1, state_dim)
    output = compiled_model(dummy_input)
    
    print(f"--- Hoàn tất! ---")
    print(f"Kết quả test: {output.detach().numpy()}")
    
    # Test thử bộ nhớ
    memory = ReplayBuffer(1000)
    memory.add([0, 5], 1, -10, [1, 5], False)
    print(f"Bộ nhớ hiện tại: {len(memory)} trải nghiệm")