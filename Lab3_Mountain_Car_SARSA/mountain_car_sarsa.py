# Lab 3: Mountain Car Control with SARSA - Interactive Visual Simulator (Tkinter GUI)
import math
import tkinter as tk
from tkinter import ttk
import numpy as np


# =====================================================================
# 1. ENVIRONMENT & SARSA REINFORCEMENT LEARNING AGENT
# =====================================================================
class MountainCarEnv:
    """
    1D Mountain Car Environment (Sutton & Barto, Example 10.1).
    - Continuous State: position x in [-1.2, 0.5], velocity v in [-0.07, 0.07]
    - Actions: 0 = Push Left (-1), 1 = Coast (0), 2 = Push Right (+1)
    - Reward: -1 on every step until reaching the hilltop goal (x >= 0.5)
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.min_pos, self.max_pos = -1.2, 0.5
        self.max_vel = 0.07
        self.goal_pos = 0.5
        self.num_actions = 3
        self.reset()

    def reset(self):
        self.pos = float(self.rng.uniform(-0.6, -0.4))
        self.vel = 0.0
        return (self.pos, self.vel)

    def step(self, action):
        force = action - 1  # {0, 1, 2} -> {-1, 0, +1}
        self.vel += 0.001 * force - 0.0025 * math.cos(3.0 * self.pos)
        self.vel = float(np.clip(self.vel, -self.max_vel, self.max_vel))
        self.pos += self.vel
        self.pos = float(np.clip(self.pos, self.min_pos, self.max_pos))

        if self.pos <= self.min_pos and self.vel < 0:
            self.vel = 0.0

        done = self.pos >= self.goal_pos
        reward = -1.0
        return (self.pos, self.vel), reward, done

    @staticmethod
    def hill_height(x):
        """Visual height function h(x) whose slope matches cos(3x)."""
        return math.sin(3.0 * x) * 0.45 + 0.55


class SarsaAgent:
    """Tabular SARSA Agent with State Discretization (Binning)."""
    def __init__(self, env, n_bins=20, alpha=0.2, gamma=0.99, epsilon=0.05):
        self.env = env
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.pos_bins = np.linspace(env.min_pos, env.max_pos, n_bins + 1)
        self.vel_bins = np.linspace(-env.max_vel, env.max_vel, n_bins + 1)
        self.reset_q()

    def reset_q(self):
        # Optimistic initialization (0.0 when step rewards are -1.0)
        self.Q = np.zeros((self.n_bins, self.n_bins, self.env.num_actions))
        self.episodes_trained = 0
        self.steps_history = []

    def discretize(self, state):
        pos, vel = state
        p_idx = int(np.clip(np.digitize(pos, self.pos_bins) - 1, 0, self.n_bins - 1))
        v_idx = int(np.clip(np.digitize(vel, self.vel_bins) - 1, 0, self.n_bins - 1))
        return (p_idx, v_idx)

    def select_action(self, state_idx, greedy=False):
        if not greedy and self.env.rng.random() < self.epsilon:
            return int(self.env.rng.integers(0, self.env.num_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s_idx, action, reward, next_s_idx, next_action, done):
        td_target = reward if done else reward + self.gamma * self.Q[next_s_idx + (next_action,)]
        self.Q[s_idx + (action,)] += self.alpha * (td_target - self.Q[s_idx + (action,)])

    def train_episodes_fast(self, num_episodes=500, max_steps=500):
        for _ in range(num_episodes):
            state = self.env.reset()
            s_idx = self.discretize(state)
            action = self.select_action(s_idx, greedy=False)
            steps = 0
            for _ in range(max_steps):
                next_state, reward, done = self.env.step(action)
                next_s_idx = self.discretize(next_state)
                next_action = self.select_action(next_s_idx, greedy=False)
                self.update(s_idx, action, reward, next_s_idx, next_action, done)
                s_idx, action = next_s_idx, next_action
                steps += 1
                if done:
                    break
            self.episodes_trained += 1
            self.steps_history.append(steps)


# =====================================================================
# 2. INTERACTIVE GUI WINDOW (TKINTER CANVAS ANIMATION)
# =====================================================================
class MountainCarVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lab 3: Mountain Car SARSA - Mô Phỏng Xe Học Leo Đồi")
        self.root.geometry("960x640")
        self.root.configure(bg="#1e293b")

        self.env = MountainCarEnv(seed=42)
        self.agent = SarsaAgent(self.env, n_bins=20, alpha=0.2, gamma=0.99, epsilon=0.05)

        # Simulation state
        self.mode = "LIVE_TRAIN"  # "LIVE_TRAIN" or "GREEDY_DEMO"
        self.paused = False
        self.steps_per_frame = 2
        self.max_ep_steps = 500
        self.best_steps = None

        self._build_ui()
        self._start_new_episode()
        self._animate()

    def _build_ui(self):
        # Top Info Bar
        top_frame = tk.Frame(self.root, bg="#0f172a", padx=15, pady=10)
        top_frame.pack(fill=tk.X)

        self.lbl_title = tk.Label(
            top_frame,
            text="MOUNTAIN CAR - THUẬT TOÁN SARSA (ON-POLICY TD CONTROL)",
            font=("Segoe UI", 13, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        )
        self.lbl_title.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            top_frame,
            text="Chế độ: Đang tự học trực tiếp (Live SARSA)",
            font=("Segoe UI", 11, "bold"),
            fg="#4ade80",
            bg="#0f172a",
        )
        self.lbl_status.pack(side=tk.RIGHT)

        # Main Canvas Area
        main_frame = tk.Frame(self.root, bg="#1e293b", padx=12, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_w = 640
        self.canvas_h = 440
        self.canvas = tk.Canvas(
            main_frame, width=self.canvas_w, height=self.canvas_h, bg="#0b1120", highlightthickness=1, highlightbackground="#334155"
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Dashboard Panel
        side_panel = tk.Frame(main_frame, width=280, bg="#0f172a", padx=14, pady=12)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side_panel.pack_propagate(False)

        tk.Label(side_panel, text="THÔNG SỐ TRẠNG THÁI", font=("Segoe UI", 11, "bold"), fg="#f8fafc", bg="#0f172a").pack(anchor="w", pady=(0, 8))

        self.lbl_ep = tk.Label(side_panel, text="Episodes đã học: 0", font=("Consolas", 11), fg="#e2e8f0", bg="#0f172a")
        self.lbl_ep.pack(anchor="w", pady=2)

        self.lbl_step = tk.Label(side_panel, text="Bước hiện tại:   0 / 500", font=("Consolas", 11), fg="#e2e8f0", bg="#0f172a")
        self.lbl_step.pack(anchor="w", pady=2)

        self.lbl_pos = tk.Label(side_panel, text="Vị trí (x):      -0.500", font=("Consolas", 11), fg="#38bdf8", bg="#0f172a")
        self.lbl_pos.pack(anchor="w", pady=2)

        self.lbl_vel = tk.Label(side_panel, text="Vận tốc (v):     +0.000", font=("Consolas", 11), fg="#fbbf24", bg="#0f172a")
        self.lbl_vel.pack(anchor="w", pady=2)

        self.lbl_act = tk.Label(side_panel, text="Hành động:       Thả trôi", font=("Consolas", 11, "bold"), fg="#a78bfa", bg="#0f172a")
        self.lbl_act.pack(anchor="w", pady=4)

        self.lbl_best = tk.Label(side_panel, text="Kỷ lục ít bước:  Chưa chạm đích", font=("Consolas", 10), fg="#4ade80", bg="#0f172a")
        self.lbl_best.pack(anchor="w", pady=4)

        # Mini Learning Curve Canvas
        tk.Label(side_panel, text="ĐƯỜNG CONG HỌC (SỐ BƯỚC/EP)", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#0f172a").pack(anchor="w", pady=(14, 4))
        self.chart_w, self.chart_h = 250, 140
        self.chart_canvas = tk.Canvas(side_panel, width=self.chart_w, height=self.chart_h, bg="#1e293b", highlightthickness=1, highlightbackground="#334155")
        self.chart_canvas.pack()

        # Speed slider
        tk.Label(side_panel, text="Tốc độ mô phỏng (Steps/Frame):", font=("Segoe UI", 9), fg="#cbd5e1", bg="#0f172a").pack(anchor="w", pady=(12, 2))
        self.speed_scale = tk.Scale(
            side_panel, from_=1, to=15, orient=tk.HORIZONTAL, bg="#0f172a", fg="#f8fafc", highlightthickness=0, command=self._on_speed_change
        )
        self.speed_scale.set(2)
        self.speed_scale.pack(fill=tk.X)

        # Bottom Control Buttons
        btn_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=10)
        btn_frame.pack(fill=tk.X)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "padx": 12, "pady": 6, "bd": 0, "cursor": "hand2"}

        tk.Button(
            btn_frame, text="⚡ Huấn Luyện Nhanh +300 Episodes", bg="#2563eb", fg="white", command=self.fast_train_300, **btn_style
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="🏆 Xem Xe Đã Học Leo Đồi (Greedy)", bg="#16a34a", fg="white", command=self.set_greedy_mode, **btn_style
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="🔄 Xem Xe Tự Học Trực Tiếp (Live Train)", bg="#d97706", fg="white", command=self.set_live_train_mode, **btn_style
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="⏯ Tạm Dừng / Tiếp Tục", bg="#475569", fg="white", command=self.toggle_pause, **btn_style
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="🗑 Reset Học Lại Từ Đầu", bg="#dc2626", fg="white", command=self.reset_all, **btn_style
        ).pack(side=tk.RIGHT, padx=5)

    def _on_speed_change(self, val):
        self.steps_per_frame = int(val)

    def _start_new_episode(self):
        self.state = self.env.reset()
        self.s_idx = self.agent.discretize(self.state)
        is_greedy = self.mode == "GREEDY_DEMO"
        self.action = self.agent.select_action(self.s_idx, greedy=is_greedy)
        self.current_steps = 0
        self.goal_banner_timer = 0

    def fast_train_300(self):
        self.agent.train_episodes_fast(num_episodes=300, max_steps=self.max_ep_steps)
        recent_min = min(self.agent.steps_history[-300:])
        if self.best_steps is None or recent_min < self.best_steps:
            self.best_steps = recent_min
        self.set_greedy_mode()
        self._draw_learning_curve()

    def set_greedy_mode(self):
        self.mode = "GREEDY_DEMO"
        self.lbl_status.config(text="Chế độ: Biểu diễn Chính sách Tối ưu (Greedy)", fg="#4ade80")
        self._start_new_episode()

    def set_live_train_mode(self):
        self.mode = "LIVE_TRAIN"
        self.lbl_status.config(text="Chế độ: Đang tự học trực tiếp (Live SARSA)", fg="#fbbf24")
        self._start_new_episode()

    def toggle_pause(self):
        self.paused = not self.paused

    def reset_all(self):
        self.agent.reset_q()
        self.best_steps = None
        self.lbl_best.config(text="Kỷ lục ít bước:  Chưa chạm đích")
        self.set_live_train_mode()
        self._draw_learning_curve()

    def _world_to_screen(self, x, y):
        w = self.canvas.winfo_width() or self.canvas_w
        h = self.canvas.winfo_height() or self.canvas_h
        margin_x = 45
        px = margin_x + (x - self.env.min_pos) / (self.env.max_pos - self.env.min_pos) * (w - 2 * margin_x)
        py = h - 55 - y * (h - 120)
        return px, py

    def _draw_scene(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or self.canvas_w
        h = self.canvas.winfo_height() or self.canvas_h

        # Draw sky grid lines
        for gx in np.linspace(-1.2, 0.5, 10):
            sx, _ = self._world_to_screen(gx, 0)
            self.canvas.create_line(sx, 20, sx, h - 30, fill="#1e293b", dash=(2, 4))
            self.canvas.create_text(sx, h - 18, text=f"{gx:.1f}", fill="#64748b", font=("Consolas", 8))

        # Draw Mountain Terrain Polygon
        xs = np.linspace(self.env.min_pos, self.env.max_pos, 120)
        poly_pts = []
        line_pts = []
        for x in xs:
            sx, sy = self._world_to_screen(x, self.env.hill_height(x))
            poly_pts.extend([sx, sy])
            line_pts.extend([sx, sy])
        poly_pts.extend([w - 45, h - 30, 45, h - 30])
        self.canvas.create_polygon(poly_pts, fill="#1e3a2f", outline="")
        self.canvas.create_line(line_pts, fill="#34d399", width=3)

        # Draw Goal Flag at x = 0.5
        gx, gy = self._world_to_screen(self.env.goal_pos, self.env.hill_height(self.env.goal_pos))
        self.canvas.create_line(gx, gy, gx, gy - 48, fill="#f8fafc", width=3)
        self.canvas.create_polygon(gx, gy - 48, gx + 28, gy - 38, gx, gy - 28, fill="#ef4444", outline="white")
        self.canvas.create_text(gx - 10, gy - 58, text="ĐÍCH (x=0.5)", fill="#f87171", font=("Segoe UI", 9, "bold"))

        # Draw Car aligned with Hill Slope
        car_x = self.state[0]
        car_y = self.env.hill_height(car_x)
        cx, cy = self._world_to_screen(car_x, car_y)

        # Compute screen-space slope angle
        dx = 0.02
        x1, y1 = self._world_to_screen(car_x - dx, self.env.hill_height(car_x - dx))
        x2, y2 = self._world_to_screen(car_x + dx, self.env.hill_height(car_x + dx))
        angle = math.atan2(y2 - y1, x2 - x1)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def rot(lx, ly):
            return (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)

        # Car chassis polygon (lifted slightly above track for wheels)
        body_coords = [rot(-22, -8), rot(22, -8), rot(20, -24), rot(-18, -24)]
        flat_body = [coord for pt in body_coords for coord in pt]
        self.canvas.create_polygon(flat_body, fill="#38bdf8", outline="#e0f2fe", width=2)

        # Cabin roof
        roof_coords = [rot(-12, -24), rot(12, -24), rot(7, -34), rot(-8, -34)]
        flat_roof = [coord for pt in roof_coords for coord in pt]
        self.canvas.create_polygon(flat_roof, fill="#0284c7", outline="#bae6fd", width=1)

        # Wheels
        for wx_local in (-13, 13):
            wx, wy = rot(wx_local, -6)
            r = 7
            self.canvas.create_oval(wx - r, wy - r, wx + r, wy + r, fill="#0f172a", outline="#94a3b8", width=2)

        # Draw Engine Thrust Flame / Direction Arrow
        if self.action == 0:  # Push Left
            fx, fy = rot(28, -15)
            self.canvas.create_polygon(rot(22, -12), rot(22, -18), (fx + 10, fy), fill="#f97316", outline="#fde047")
            self.canvas.create_text(cx, cy - 45, text="◀ ĐẨY TRÁI", fill="#fb923c", font=("Segoe UI", 9, "bold"))
        elif self.action == 2:  # Push Right
            fx, fy = rot(-28, -15)
            self.canvas.create_polygon(rot(-22, -12), rot(-22, -18), (fx - 10, fy), fill="#f97316", outline="#fde047")
            self.canvas.create_text(cx, cy - 45, text="ĐẨY PHẢI ▶", fill="#fb923c", font=("Segoe UI", 9, "bold"))

        # Banner when reaching goal
        if self.goal_banner_timer > 0:
            self.canvas.create_rectangle(w / 2 - 180, 40, w / 2 + 180, 85, fill="#16a34a", outline="white", width=2)
            self.canvas.create_text(
                w / 2, 62, text=f"🎉 CHẠM ĐỈNH ĐỒI SAU {self.last_win_steps} BƯỚC!", fill="white", font=("Segoe UI", 12, "bold")
            )

    def _draw_learning_curve(self):
        self.chart_canvas.delete("all")
        hist = self.agent.steps_history
        if not hist:
            self.chart_canvas.create_text(
                self.chart_w / 2, self.chart_h / 2, text="Chưa có dữ liệu Episode", fill="#64748b", font=("Segoe UI", 9)
            )
            return

        recent = hist[-150:]
        n = len(recent)
        pts = []
        for i, val in enumerate(recent):
            x = 10 + (i / max(1, n - 1)) * (self.chart_w - 20)
            y = self.chart_h - 12 - ((val - 80) / (self.max_ep_steps - 80)) * (self.chart_h - 24)
            y = max(8, min(self.chart_h - 8, y))
            pts.extend([x, y])

        if len(pts) >= 4:
            self.chart_canvas.create_line(pts, fill="#38bdf8", width=2)
        self.chart_canvas.create_text(
            self.chart_w / 2, 12, text=f"150 Ep gần nhất (TB: {np.mean(recent[-20:]):.0f} bước)", fill="#e2e8f0", font=("Segoe UI", 8)
        )

    def _animate(self):
        if not self.paused:
            if self.goal_banner_timer > 0:
                self.goal_banner_timer -= 1
                if self.goal_banner_timer == 0:
                    self._start_new_episode()
            else:
                for _ in range(self.steps_per_frame):
                    next_state, reward, done = self.env.step(self.action)
                    next_s_idx = self.agent.discretize(next_state)
                    is_greedy = self.mode == "GREEDY_DEMO"
                    next_action = self.agent.select_action(next_s_idx, greedy=is_greedy)

                    if self.mode == "LIVE_TRAIN":
                        self.agent.update(self.s_idx, self.action, reward, next_s_idx, next_action, done)

                    self.state = next_state
                    self.s_idx = next_s_idx
                    self.action = next_action
                    self.current_steps += 1

                    if done or self.current_steps >= self.max_ep_steps:
                        if self.mode == "LIVE_TRAIN":
                            self.agent.episodes_trained += 1
                            self.agent.steps_history.append(self.current_steps)
                            self._draw_learning_curve()

                        if done:
                            self.last_win_steps = self.current_steps
                            if self.best_steps is None or self.current_steps < self.best_steps:
                                self.best_steps = self.current_steps
                                self.lbl_best.config(text=f"Kỷ lục ít bước:  {self.best_steps} bước")
                            self.goal_banner_timer = 22
                        else:
                            self._start_new_episode()
                        break

        # Update Labels
        act_names = {0: "◀ Đẩy Trái (-1)", 1: "• Thả Trôi (0)", 2: "▶ Đẩy Phải (+1)"}
        self.lbl_ep.config(text=f"Episodes đã học: {self.agent.episodes_trained}")
        self.lbl_step.config(text=f"Bước hiện tại:   {self.current_steps} / {self.max_ep_steps}")
        self.lbl_pos.config(text=f"Vị trí (x):      {self.state[0]:+.3f}")
        self.lbl_vel.config(text=f"Vận tốc (v):     {self.state[1]:+.4f}")
        self.lbl_act.config(text=f"Hành động:       {act_names[self.action]}")

        self._draw_scene()
        self.root.after(20, self._animate)


if __name__ == "__main__":
    root = tk.Tk()
    app = MountainCarVisualizerApp(root)
    root.mainloop()
