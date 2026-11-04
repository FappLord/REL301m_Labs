# REL301m - Reinforcement Learning Labs (Interactive Physics Simulators)

Bộ 3 bài Lab môn **Reinforcement Learning (REL301m)** mô phỏng các bài toán điều khiển vật lý & cơ điện tử (Robotics / Control Engineering) có giao diện đồ họa tương tác thời gian thực (**Tkinter GUI**) và báo cáo thực nghiệm (**Jupyter Notebook**).

---

## Cấu trúc thư mục

```text
lab/
├── index.html                              # Giao diện Web Dashboard tổng hợp 3 Lab + Biểu đồ trực tiếp
├── PRESENTATION_SCRIPT.md                  # Kịch bản thuyết trình chi tiết cho 3 bài Lab
├── Lab1_Ball_And_Beam_QLearning/
│   ├── ball_and_beam_qlearning.py          # Ứng dụng GUI mô phỏng Thanh bập bênh cân bằng bóng
│   ├── Lab1_Ball_And_Beam_QLearning.ipynb  # Notebook báo cáo & thực nghiệm
│   └── ball_and_beam_result.png            # Biểu đồ kết quả huấn luyện
├── Lab2_Robotic_Arm_Reacher_ExpectedSarsa/
│   ├── robotic_arm_sarsa.py                # Ứng dụng GUI mô phỏng Cánh tay Robot 2 khớp gắp mục tiêu
│   ├── Lab2_Robotic_Arm_Reacher_ExpectedSarsa.ipynb
│   └── robotic_arm_result.png
└── Lab3_Mountain_Car_SARSA/
    ├── mountain_car_sarsa.py               # Ứng dụng GUI mô phỏng Xe lấy đà leo đồi
    ├── Lab3_Mountain_Car_SARSA.ipynb
    └── mountain_car_result.png
```

---

## 1. Chạy giao diện Web Dashboard tổng hợp (`index.html`)

Mở trực tiếp file `index.html` trên trình duyệt (Chrome / Edge / Firefox) để trải nghiệm cả 3 bài Lab trên cùng một giao diện kèm 4 biểu đồ thời gian thực (Learning Curve, Episode Return & Success Rate, Bản đồ chính sách $\pi(s)$, và Bản đồ nhiệt $V(s)$).

## 2. Hướng dẫn chạy các ứng dụng mô phỏng Python (Tkinter GUI)

Yêu cầu cài đặt: Python 3.8+ cùng `numpy` và `matplotlib` (`tkinter` có sẵn trong Python chuẩn).

```bash
# Chạy Lab 1: Thanh Bập Bênh Cân Bằng Bóng (Q-Learning)
python Lab1_Ball_And_Beam_QLearning/ball_and_beam_qlearning.py

# Chạy Lab 2: Cánh Tay Robot 2 Khớp Vươn Gắp Mục Tiêu (Expected SARSA)
python Lab2_Robotic_Arm_Reacher_ExpectedSarsa/robotic_arm_sarsa.py

# Chạy Lab 3: Xe Lấy Đà Leo Đồi (SARSA)
python Lab3_Mountain_Car_SARSA/mountain_car_sarsa.py
```

