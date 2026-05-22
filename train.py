import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Import môi trường từ file network_env.py của bạn
from network_env import CurriculumNetworkEnv

# --- 1. BIẾN ĐỔI TRẠNG THÁI (One-hot Encoding) ---
def state_to_onehot(obs, num_nodes):
    current_node, dest_node = obs[0], obs[1]
    onehot = np.zeros(num_nodes * 2, dtype=np.float32)
    onehot[current_node] = 1.0
    onehot[num_nodes + dest_node] = 1.0
    return torch.FloatTensor(onehot).unsqueeze(0)

# --- 2. MẠNG NEURAL (Q-Net) ---
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

# --- 3. BỘ NHỚ ĐỆM (Replay Buffer) ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.cat(states), 
                torch.LongTensor(actions).unsqueeze(1), 
                torch.FloatTensor(rewards).unsqueeze(1), 
                torch.cat(next_states), 
                torch.FloatTensor(dones).unsqueeze(1))
    
    def __len__(self):
        return len(self.buffer)

# --- 4. TÁC NHÂN AI (DQNAgent) ---
class DQNAgent:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.state_dim = num_nodes * 2
        self.action_dim = num_nodes
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        self.policy_net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.lr = 1e-3
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_learning_rate(self, new_lr):
        """Hàm để hạ LR trong giai đoạn fine-tuning"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.lr = new_lr
        print(f"[*] Learning Rate đã được điều chỉnh thành: {self.lr}")

    def save_model(self, filepath="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"[*] Đã lưu mô hình tại: {filepath}")

    def load_model(self, filepath="dqn_model.pth"):
        if os.path.exists(filepath):
            self.policy_net.load_state_dict(torch.load(filepath))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = self.epsilon_min
            print(f"[*] Đã tải thành công mô hình từ: {filepath}")
        else:
            print("[!] Không tìm thấy file mô hình cũ, bắt đầu huấn luyện từ đầu.")

# --- 5. VÒNG LẶP HUẤN LUYỆN CHÍNH ---
if __name__ == "__main__":
    MAX_NODES = 20 # Cố định số lượng node để thiết kế Mạng Neural
    env = CurriculumNetworkEnv(max_nodes=MAX_NODES)
    agent = DQNAgent(MAX_NODES)
    
    num_episodes = 25500  # Tăng số vòng lặp để đủ thời gian học 3 level
    target_update_freq = 10
    rewards_history =[]
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN CURRICULUM LEARNING ---")
    start_time = time.time()
    
    for ep in range(1, num_episodes + 1):
        # LỊCH TRÌNH CHUYỂN CẤP (Curriculum Schedule)
        if ep == 1500:
            env.build_level(2); agent.epsilon = 0.5
        elif ep == 4500:
            env.build_level(3); agent.epsilon = 0.5
        elif ep == 9000:
            env.build_level(4); agent.epsilon = 0.5
        elif ep == 15000:
            env.build_level(5); agent.epsilon = 0.5
        # --- CƠ CHẾ FINE-TUNING GIAI ĐOẠN CUỐI (Ví dụ: 5000 episode cuối) ---
        if ep == 20000:
            print("\n[!!!] BẮT ĐẦU GIAI ĐOẠN FINE-TUNING [!!!]")
            agent.set_learning_rate(1e-4) # Hạ tốc độ học xuống 10 lần
            agent.epsilon = 0.05         # Giảm sự ngẫu nhiên xuống mức thấp
            agent.epsilon_decay = 0.999  # Giảm cực chậm epsilon
            # Tăng độ ưu tiên cho các trải nghiệm gần nhất trong Buffer (nếu cần)

        obs, _ = env.reset()
        state = state_to_onehot(obs, MAX_NODES)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_to_onehot(next_obs, MAX_NODES)
            
            agent.memory.add(state, action, reward, next_state, float(done))
            agent.train_step()
            
            state = next_state
            total_reward += reward
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            
        target_update_freq = 5 if ep > 20000 else 10
        if ep % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        rewards_history.append(total_reward)
        
        if ep % 250 == 0:
            print(f"Ep: {ep:4d}/{num_episodes} | Level: {env.level} | Reward: {total_reward:7.2f} | Epsilon: {agent.epsilon:.3f}")

    end_time = time.time()
    print(f"\n--- HOÀN TẤT HUẤN LUYỆN ---")
    print(f"Thời gian: {end_time - start_time:.2f} giây")
    agent.save_model("dqn_curriculum_model.pth")

    # --- VẼ ĐỒ THỊ ---
    print("\n[*] Đang tạo đồ thị...")
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, label='Reward', color='lightblue', alpha=0.5)
    
    window = 30
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards_history)), moving_avg, color='red', label='Trung bình trượt')
    
    # Vẽ đường ranh giới các Level
    plt.axvline(x=1500, color='green', linestyle='--', label='Bắt đầu Level 2')
    plt.axvline(x=4500, color='purple', linestyle='--', label='Bắt đầu Level 3')
    plt.axvline(x=9000, color='orange', linestyle='--', label='Bắt đầu Level 4')
    plt.axvline(x=15000, color='red', linestyle='--', label='Bắt đầu Level 5')

    plt.title('Curriculum Learning: DQN Agent qua 5 cấp độ mạng')
    plt.xlabel('Episodes')
    plt.ylabel('Tổng Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig("curriculum_curve.png", dpi=300, bbox_inches='tight')
    print("[*] Đã lưu đồ thị thành công tại: curriculum_curve.png")