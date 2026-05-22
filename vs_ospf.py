import torch
import numpy as np
import networkx as nx
import time
from network_env import CurriculumNetworkEnv
from train import state_to_onehot, QNet # Import lại cấu hình cũ

def get_ospf_path(graph, start, end):
    """Mô phỏng OSPF: Tìm đường ngắn nhất dựa trên Cost = 1000/Bandwidth"""
    # Tính cost cho từng cạnh
    for u, v, d in graph.edges(data=True):
        graph[u][v]['ospf_cost'] = 1000 / d['bandwidth']
    
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight='ospf_cost')
        total_delay = sum(graph[path[i]][path[i+1]]['delay'] for i in range(len(path)-1))
        return path, total_delay
    except:
        return None, float('inf')

def run_test_head_to_head(num_tests=10000):
    MAX_NODES = 20
    env = CurriculumNetworkEnv(max_nodes=MAX_NODES)
    env.build_level(5) # Kiểm thử trên môi trường khó nhất
    
    # Load Agent đã train
    device = torch.device("cpu")
    model = QNet(MAX_NODES * 2, MAX_NODES).to(device)
    if os.path.exists("dqn_curriculum_model.pth"):
        model.load_state_dict(torch.load("dqn_curriculum_model.pth"))
        model.eval()
    else:
        print("Lỗi: Không tìm thấy file model!")
        return

    dqn_delays = []
    ospf_delays = []
    dqn_wins = 0
    ospf_wins = 0

    print(f"{'Test':<5} | {'Source->Dest':<15} | {'DQN Delay':<10} | {'OSPF Delay':<10} | {'Winner':<6}")
    print("-" * 60)

    for i in range(num_tests):
        obs, _ = env.reset()
        start_node, dest_node = obs[0], obs[1]
        
        # 1. DQN thực hiện định tuyến
        state = state_to_onehot(obs, MAX_NODES)
        curr_obs = obs
        dqn_path = [start_node]
        dqn_total_delay = 0
        step_count = 0
        done = False
        
        while not done and step_count < 25:
            with torch.no_grad():
                action = model(state).argmax(dim=1).item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Tính delay thực tế nếu bước đi hợp lệ
            if env.graph.has_edge(dqn_path[-1], action):
                dqn_total_delay += env.graph[dqn_path[-1]][action]['delay']
                dqn_path.append(action)
            else:
                dqn_total_delay += 200 # Phạt nặng nếu Agent chọn đường không tồn tại
                break
                
            state = state_to_onehot(next_obs, MAX_NODES)
            done = terminated or truncated
            step_count += 1

        # 2. OSPF thực hiện định tuyến
        ospf_path, ospf_total_delay = get_ospf_path(env.graph, start_node, dest_node)

        # 3. So sánh
        winner = "DQN" if dqn_total_delay < ospf_total_delay else "OSPF" if ospf_total_delay < dqn_total_delay else "Tie"
        if dqn_total_delay < ospf_total_delay: dqn_wins += 1
        elif dqn_total_delay > ospf_total_delay: ospf_wins += 1

        dqn_delays.append(dqn_total_delay)
        ospf_delays.append(ospf_total_delay)

        if i % 100 == 0: # Chỉ in một số kết quả tiêu biểu
            print(f"{i:<5} | {start_node:2d} -> {dest_node:2d}      | {dqn_total_delay:<10.1f} | {ospf_total_delay:<10.1f} | {winner}")

    # Tổng kết
    print("-" * 60)
    print(f"KẾT QUẢ SAU {num_tests} KỊCH BẢN:")
    print(f" - DQN Trung bình Delay: {np.mean(dqn_delays):.2f} ms")
    print(f" - OSPF Trung bình Delay: {np.mean(ospf_delays):.2f} ms")
    print(f" - Tỷ lệ DQN thắng: {(dqn_wins/num_tests)*100:.2f}%")
    print(f" - Tỷ lệ OSPF thắng: {(ospf_wins/num_tests)*100:.2f}%")

if __name__ == "__main__":
    import os
    run_test_head_to_head()