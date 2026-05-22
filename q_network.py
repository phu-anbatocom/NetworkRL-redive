import torch
import torch.nn as nn
import os

# Tối ưu hóa đa luồng cho CPU Intel
num_threads = os.cpu_count()
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)

class SimpleDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimpleDQN, self).__init__()
        # Sử dụng layer kích thước lũy thừa của 2 để tối ưu hóa vectorization
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Ép dùng CPU nhưng với đồ thị tính toán đã được biên dịch (Compiled Graph)
device = torch.device("cpu")
model = SimpleDQN(state_dim=2, action_dim=6).to(device)

# Sử dụng torch.compile với backend "inductor" - đây là chìa khóa để nhanh như GPU
model = torch.compile(model) 

print(f"--- Đang dùng CPU với {num_threads} luồng và oneDNN ---")