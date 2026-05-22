import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class CurriculumNetworkEnv(gym.Env):
    def __init__(self, max_nodes=20):
        super().__init__()
        self.max_nodes = max_nodes
        
        # Action & Obs Space luôn cố định là max_nodes để Neural Network không bị lỗi
        self.action_space = spaces.Discrete(max_nodes)
        self.observation_space = spaces.MultiDiscrete([max_nodes, max_nodes])
        
        self.graph = nx.Graph()
        self.build_level(1) # Mặc định bắt đầu ở Level 1
        
    def build_level(self, level):
        self.level = level
        self.graph.clear()
        self.graph.add_nodes_from(range(self.max_nodes))
        edges = [] # Khởi tạo danh sách trống để tránh lỗi UnboundLocalError

        if level == 1:
            self.active_nodes = 6
            # Định dạng: (u, v, delay, bandwidth)
            edges = [(0,1,10,100), (1,2,10,100), (2,3,10,100), (3,4,10,100), (4,5,10,100), (0,2,50,1000)]
        
        elif level == 2:
            self.active_nodes = 9
            edges = [(i, i+1, 10, 100) for i in range(8)]
            edges += [(0,5,100,1000), (2,8,5,10), (5,8,80,1000), (7,8,10,100), (1,8,40,100)]
            
        elif level == 3:
            self.active_nodes = 12
            temp_g = nx.watts_strogatz_graph(12, k=4, p=0.2, seed=42)
            for u, v in temp_g.edges():
                d, bw = (5, 10) if np.random.random() > 0.3 else (80, 1000)
                edges.append((u, v, d, bw))

        elif level == 4:
            self.active_nodes = 15
            temp_g = nx.watts_strogatz_graph(15, k=6, p=0.3, seed=99)
            for u, v in temp_g.edges():
                d, bw = (5, 10) if np.random.random() > 0.3 else (80, 1000)
                edges.append((u, v, d, bw))

        else: # Level 5
            self.active_nodes = 20
            temp_g = nx.watts_strogatz_graph(20, k=8, p=0.4, seed=123)
            for u, v in temp_g.edges():
                # Tạo "cạm bẫy" cho OSPF: 
                # 20% xác định là đường trục (BW lớn, Delay cao)
                # 80% là đường tắt (BW thấp, Delay thấp)
                if np.random.random() > 0.8:
                    d, bw = 80, 1000 
                else:
                    d, bw = 5, 10
                edges.append((u, v, d, bw))

        # Nạp tất cả edges vào graph với đầy đủ thuộc tính
        for edge in edges:
            u, v, d, bw = edge
            self.graph.add_edge(u, v, delay=d, bandwidth=bw)
        
        print(f"\n[Environment] Đã chuyển sang LEVEL {level} (Nodes: {self.active_nodes})")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Chỉ chọn Nguồn và Đích trong phạm vi các node đang kích hoạt
        self.current_node = self.np_random.integers(0, self.active_nodes)
        self.dest_node = self.np_random.integers(0, self.active_nodes)
        while self.current_node == self.dest_node:
            self.dest_node = self.np_random.integers(0, self.active_nodes)
            
        self.visited_nodes = {self.current_node}
        return np.array([self.current_node, self.dest_node], dtype=np.int32), {}

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        
        if not self.graph.has_edge(self.current_node, action):
            reward = -50.0  # Phạt đi sai đường (hoặc đi vào node chưa kích hoạt)
        elif action in self.visited_nodes:
            reward = -100.0 # Phạt lặp vòng
            terminated = True
        else:
            delay = self.graph[self.current_node][action]['delay']
            reward = -delay 
            self.current_node = action
            self.visited_nodes.add(self.current_node)
            
            if self.current_node == self.dest_node:
                reward += 200.0
                terminated = True
                
        # Level càng cao, quãng đường cho phép càng dài
        max_steps = 8 if self.level == 1 else (15 if self.level == 2 else 25)
        if len(self.visited_nodes) > max_steps:
            truncated = True
            
        obs = np.array([self.current_node, self.dest_node], dtype=np.int32)
        return obs, reward, terminated, truncated, {"visited": list(self.visited_nodes)}