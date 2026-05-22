# Reinforcement Learning for Intelligent Network Routing (Non-CUDA Optimized)

Dự án nghiên cứu và triển khai thuật toán Học tăng cường (Reinforcement Learning - RL) để tối ưu hóa định tuyến mạng, tập trung vào việc giảm thiểu độ trễ (Latency) thay vì chỉ dựa vào băng thông tĩnh như các giao thức truyền thống (OSPF).

**Điểm đặc biệt:** Dự án được tối ưu hóa hoàn toàn cho kiến trúc **CPU Intel (Tiger Lake)** và **iGPU**, không phụ thuộc vào hệ sinh thái NVIDIA CUDA, hướng tới việc triển khai AI trên các thiết bị mạng đầu cuối (Edge Devices).

## 🚀 Tính năng nổi bật

- **Curriculum Learning (5 Levels):** Lộ trình huấn luyện Agent từ mạng đơn giản (6 nodes) đến mạng phức tạp (20 nodes - Watts-Strogatz Small World Topology) giúp tăng tốc độ hội tụ và khả năng tổng quát hóa.
- **Hardware-Agnostic Optimization:** Tận dụng tập lệnh **AVX-512 VNNI** và thư viện **Intel oneDNN** tích hợp trong PyTorch để đạt hiệu năng bare-metal trên CPU mà không cần GPU rời.
- **Double DQN (DDQN):** Sử dụng cơ chế Policy và Target Network để ổn định hóa giá trị Q-value, tránh hiện tượng ảo tưởng phần thưởng (overestimation).
- **Head-to-Head Benchmarking:** Hệ thống đánh giá đối đầu trực tiếp với giao thức OSPF (Dijkstra) dựa trên dữ liệu thực tế.

## 📊 Kết quả thực nghiệm

Mô hình đạt được sự hội tụ tối ưu tại ngưỡng **25,500 episodes**. Kết quả so sánh với giao thức OSPF truyền thống trong môi trường mạng phức tạp:

| Chỉ số | OSPF (Dijkstra) | DQN Agent (Ours) | Cải thiện |
| :--- | :--- | :--- | :--- |
| **Độ trễ trung bình** | 139.85 ms | **48.62 ms** | **~65%** |
| **Tỷ lệ thắng (Win Rate)** | 20.62% | **62.43%** | +41.81% |

*Lưu ý: DQN cho thấy khả năng triệt tiêu hoàn toàn các kịch bản "thảm họa độ trễ" mà OSPF thường mắc phải khi ưu tiên đường trục băng thông lớn.*

## 🛠 Cấu trúc dự án

- `network_env.py`: Môi trường giả lập mạng tùy biến dựa trên Gymnasium và NetworkX. Hỗ trợ 5 cấp độ Curriculum Learning.
- `train.py`: Mã nguồn huấn luyện Agent, tích hợp cơ chế Save/Load mô hình và vẽ đồ thị hội tụ.
- `vs_ospf.py`: Chương trình kiểm thử đối đầu (Head-to-Head) giữa AI và giao thức truyền thống.
- `q_network.py`: Script kiểm tra tính tương thích phần cứng và tối ưu hóa XPU/CPU.

## 💻 Cài đặt & Sử dụng

### Yêu cầu hệ thống
- OS: WSL2 (Ubuntu 22.04/24.04) hoặc Linux thuần.
- CPU: Intel Gen 11th+ (hỗ trợ AVX-512) để đạt hiệu năng tốt nhất.
- Python 3.10+

### Cài đặt môi trường
```bash
# Cài đặt PyTorch tối ưu cho Intel CPU/XPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện bổ trợ
pip install gymnasium networkx numpy matplotlib