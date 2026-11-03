# Lab 1: Ball & Beam Balancing Control with Q-Learning - Interactive Physics Simulator (Tkinter GUI)
import math
import tkinter as tk
import numpy as np


# =====================================================================
# 1. BALL & BEAM PHYSICS ENVIRONMENT & Q-LEARNING AGENT
# =====================================================================
class BallAndBeamEnv:
    """
    Ball and Beam Balancing Control System (Classic Control & Robotics Benchmark).
    - A steel ball rolls on a pivoting beam of length 2.0 m (x in [-1.0, +1.0]).
    - Continuous State: (x, v, theta)
      * x in [-1.0, +1.0] m: Ball position from the center pivot (Target: x = 0.0)
      * v in [-1.2, +1.2] m/s: Ball rolling velocity along the beam
      * theta in [-0.26, +0.26] rad (+/- 15 deg): Beam tilt angle (positive = right side down)
    - Actions:
      * 0: Tilt Beam Left  (d_theta = -0.025 rad)
      * 1: Hold Beam Angle (d_theta =  0.000 rad)
      * 2: Tilt Beam Right (d_theta = +0.025 rad)
    - Goal: Prevent the ball from rolling off the edges (|x| > 1.0) and stabilize it at x = 0.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.gravity = 9.81
        self.dt = 0.025
        self.max_x = 1.0
        self.max_v = 1.2
        self.max_theta = math.radians(15.0)
        self.d_theta = math.radians(1.6)
        self.num_actions = 3
        self.reset()

    def reset(self):
        # Start the ball away from the center with nonzero velocity so an untrained beam drops it
        side = -1.0 if self.rng.random() < 0.5 else 1.0
        self.x = float(side * self.rng.uniform(0.35, 0.70))
        self.v = float(side * self.rng.uniform(0.05, 0.25))
        self.theta = float(self.rng.uniform(-0.08, 0.08))
        return self.get_state()

    def get_state(self):
        return (self.x, self.v, self.theta)

    def step(self, action):
        # Update beam tilt angle
        if action == 0:
            self.theta -= self.d_theta
        elif action == 2:
            self.theta += self.d_theta
        self.theta = float(np.clip(self.theta, -self.max_theta, self.max_theta))

        # Solid sphere rolling down an inclined plane: a = (5/7) * g * sin(theta)
        accel = (5.0 / 7.0) * self.gravity * math.sin(self.theta)
        self.v = float(np.clip(0.992 * self.v + accel * self.dt, -self.max_v, self.max_v))
        self.x = float(self.x + self.v * self.dt)

        dropped = abs(self.x) > self.max_x

        if dropped:
            reward = -50.0
        else:
            # Desired tilt angle naturally opposes ball offset (x) and rolling speed (v)
            desired_theta = np.clip(-0.22 * self.x - 0.35 * self.v, -self.max_theta, self.max_theta)
            angle_err = abs(self.theta - desired_theta)
            reward = 2.0 - 1.5 * abs(self.x) - 0.8 * abs(self.v) - 6.0 * angle_err

        return self.get_state(), reward, dropped


class QLearningBallBeamAgent:
    """Tabular Q-Learning Agent with State Discretization."""
    def __init__(self, env, n_bins=11, alpha=0.25, gamma=0.98, epsilon=0.15):
        self.env = env
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Discretize composite desired-tilt error and current beam angle
        # Composite ball state s_ball = x + 1.5 * v captures both position and momentum
        self.ball_bins = np.linspace(-1.1, 1.1, n_bins - 1)
        self.theta_bins = np.linspace(-env.max_theta * 0.85, env.max_theta * 0.85, n_bins - 1)
        self.q_shape = (n_bins, n_bins, env.num_actions)
        self.reset_q()

    def reset_q(self):
        self.Q = np.zeros(self.q_shape)
        self.episodes_trained = 0
        self.steps_history = []

    def discretize(self, state):
        x, v, theta = state
        ball_state = x + 1.55 * v
        return (
            int(np.digitize(ball_state, self.ball_bins)),
            int(np.digitize(theta, self.theta_bins)),
        )

    def select_action(self, s_idx, greedy=False, eps=None):
        use_eps = self.epsilon if eps is None else eps
        if not greedy and self.env.rng.random() < use_eps:
            return int(self.env.rng.integers(0, self.env.num_actions))
        return int(np.argmax(self.Q[s_idx]))

    def update(self, s_idx, action, reward, next_s_idx, done, lr=None):
        step_lr = self.alpha if lr is None else lr
        best_next = 0.0 if done else float(np.max(self.Q[next_s_idx]))
        td_target = reward + self.gamma * best_next
        self.Q[s_idx + (action,)] += step_lr * (td_target - self.Q[s_idx + (action,)])

    def train_episodes_fast(self, num_episodes=350, max_steps=400):
        for _ in range(num_episodes):
            state = self.env.reset()
            s_idx = self.discretize(state)
            steps = 0
            eps = max(0.02, 0.25 * (0.99 ** self.episodes_trained))
            lr = max(0.05, 0.25 * (0.992 ** self.episodes_trained))
            for _ in range(max_steps):
                action = self.select_action(s_idx, greedy=False, eps=eps)
                next_state, reward, dropped = self.env.step(action)
                next_s_idx = self.discretize(next_state)
                self.update(s_idx, action, reward, next_s_idx, dropped, lr=lr)
                s_idx = next_s_idx
                steps += 1
                if dropped:
                    break
            self.episodes_trained += 1
            self.steps_history.append(steps)


# =====================================================================
# 2. INTERACTIVE GUI WINDOW (TKINTER CANVAS ANIMATION)
# =====================================================================
class BallAndBeamVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lab 1: Ball & Beam Q-Learning - Mô Phỏng Thanh Bập Bênh Cân Bằng Bóng")
        self.root.geometry("980x640")
        self.root.configure(bg="#1e293b")

        self.env = BallAndBeamEnv(seed=42)
        self.agent = QLearningBallBeamAgent(self.env)

        self.mode = "LIVE_TRAIN"
        self.paused = False
        self.steps_per_frame = 1
        self.max_ep_steps = 400
        self.best_steps = 0
        self.banner_timer = 0
        self.banner_text = ""
        self.banner_color = "#dc2626"

        self._build_ui()
        self._start_new_episode()
        self._animate()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#0f172a", padx=15, pady=10)
        top.pack(fill=tk.X)

        tk.Label(
            top,
            text="BALL & BEAM BALANCING - THUẬT TOÁN Q-LEARNING (OFF-POLICY TD CONTROL)",
            font=("Segoe UI", 13, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        ).pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            top, text="Chế độ: Đang tự học trực tiếp (Live Q-Learning)", font=("Segoe UI", 11, "bold"), fg="#fbbf24", bg="#0f172a"
        )
        self.lbl_status.pack(side=tk.RIGHT)

        main_frame = tk.Frame(self.root, bg="#1e293b", padx=12, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_w, self.canvas_h = 650, 440
        self.canvas = tk.Canvas(main_frame, width=self.canvas_w, height=self.canvas_h, bg="#0b1120", highlightthickness=1, highlightbackground="#334155")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right Dashboard
        side = tk.Frame(main_frame, width=285, bg="#0f172a", padx=14, pady=12)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        tk.Label(side, text="THÔNG SỐ BÓNG & THANH", font=("Segoe UI", 11, "bold"), fg="#f8fafc", bg="#0f172a").pack(anchor="w", pady=(0, 8))

        self.lbl_ep = tk.Label(side, text="Episodes đã học: 0", font=("Consolas", 11), fg="#e2e8f0", bg="#0f172a")
        self.lbl_ep.pack(anchor="w", pady=2)

        self.lbl_step = tk.Label(side, text="Bước giữ bóng:   0 / 400", font=("Consolas", 11), fg="#4ade80", bg="#0f172a")
        self.lbl_step.pack(anchor="w", pady=2)

        self.lbl_x = tk.Label(side, text="Vị trí bóng (x): +0.00 m", font=("Consolas", 11), fg="#38bdf8", bg="#0f172a")
        self.lbl_x.pack(anchor="w", pady=2)

        self.lbl_v = tk.Label(side, text="Vận tốc bóng(v): +0.00 m/s", font=("Consolas", 11), fg="#fbbf24", bg="#0f172a")
        self.lbl_v.pack(anchor="w", pady=2)

        self.lbl_theta = tk.Label(side, text="Góc nghiêng (θ): +0.0°", font=("Consolas", 11), fg="#e2e8f0", bg="#0f172a")
        self.lbl_theta.pack(anchor="w", pady=2)

        self.lbl_act = tk.Label(side, text="Điều khiển:      Giữ nguyên", font=("Consolas", 11, "bold"), fg="#a78bfa", bg="#0f172a")
        self.lbl_act.pack(anchor="w", pady=4)

        self.lbl_best = tk.Label(side, text="Kỷ lục lâu nhất: 0 / 400 bước", font=("Consolas", 10), fg="#4ade80", bg="#0f172a")
        self.lbl_best.pack(anchor="w", pady=4)

        tk.Label(side, text="ĐƯỜNG CONG HỌC (SỐ BƯỚC GIỮ BÓNG)", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#0f172a").pack(anchor="w", pady=(10, 4))
        self.chart_w, self.chart_h = 255, 135
        self.chart_canvas = tk.Canvas(side, width=self.chart_w, height=self.chart_h, bg="#1e293b", highlightthickness=1, highlightbackground="#334155")
        self.chart_canvas.pack()

        tk.Label(side, text="Tốc độ mô phỏng (Steps/Frame):", font=("Segoe UI", 9), fg="#cbd5e1", bg="#0f172a").pack(anchor="w", pady=(10, 2))
        self.speed_scale = tk.Scale(side, from_=1, to=10, orient=tk.HORIZONTAL, bg="#0f172a", fg="#f8fafc", highlightthickness=0, command=self._on_speed)
        self.speed_scale.set(1)
        self.speed_scale.pack(fill=tk.X)

        # Bottom Controls
        btn_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=10)
        btn_frame.pack(fill=tk.X)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "padx": 11, "pady": 6, "bd": 0, "cursor": "hand2"}

        tk.Button(
            btn_frame, text="⚡ Huấn Luyện Nhanh +350 Episodes", bg="#2563eb", fg="white", command=self.fast_train_350, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🏆 Xem Thanh Tự Cân Bằng Bóng (Greedy)", bg="#16a34a", fg="white", command=self.set_greedy_mode, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="💨 Hất Bóng Lăn Về Mép (Test Phản Xạ)", bg="#9333ea", fg="white", command=self.push_ball, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🔄 Xem Tự Học (Live Train)", bg="#d97706", fg="white", command=self.set_live_train_mode, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🗑 Reset Học Lại", bg="#dc2626", fg="white", command=self.reset_all, **btn_style
        ).pack(side=tk.RIGHT, padx=4)

    def _on_speed(self, val):
        self.steps_per_frame = int(val)

    def _start_new_episode(self):
        self.state = self.env.reset()
        self.s_idx = self.agent.discretize(self.state)
        self.current_steps = 0
        self.action = 1

    def fast_train_350(self):
        self.agent.train_episodes_fast(num_episodes=350, max_steps=self.max_ep_steps)
        self.best_steps = max(self.best_steps, max(self.agent.steps_history[-350:]))
        self.lbl_best.config(text=f"Kỷ lục lâu nhất: {self.best_steps} / 400 bước")
        self.set_greedy_mode()
        self._draw_learning_curve()

    def push_ball(self):
        # Give the ball a strong velocity kick toward one of the edges
        direction = -1.0 if self.env.x >= 0 else 1.0
        self.env.v = float(direction * 0.65)
        self.state = self.env.get_state()

    def set_greedy_mode(self):
        self.mode = "GREEDY_DEMO"
        self.lbl_status.config(text="Chế độ: Biểu diễn Cân Bằng Bóng Tối Ưu (Greedy)", fg="#4ade80")
        self._start_new_episode()

    def set_live_train_mode(self):
        self.mode = "LIVE_TRAIN"
        self.lbl_status.config(text="Chế độ: Đang tự học trực tiếp (Live Q-Learning)", fg="#fbbf24")
        self._start_new_episode()

    def reset_all(self):
        self.agent.reset_q()
        self.best_steps = 0
        self.lbl_best.config(text="Kỷ lục lâu nhất: 0 / 400 bước")
        self.set_live_train_mode()
        self._draw_learning_curve()

    def _draw_scene(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or self.canvas_w
        h = c.winfo_height() or self.canvas_h

        cx, cy = w / 2, h * 0.56
        beam_half_px = 240
        x, v, theta = self.state

        # Reference horizontal dashed line
        c.create_line(cx - 270, cy, cx + 270, cy, fill="#1e293b", dash=(4, 4), width=2)

        # Support pedestal triangle (Fulcrum)
        c.create_polygon(cx, cy + 6, cx - 36, cy + 95, cx + 36, cy + 95, fill="#334155", outline="#94a3b8", width=2)
        c.create_rectangle(cx - 70, cy + 95, cx + 70, cy + 110, fill="#1e293b", outline="#64748b", width=2)

        # Beam endpoints rotated by theta (positive theta = right side down)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        lx1, ly1 = cx - beam_half_px * cos_t, cy - beam_half_px * sin_t
        lx2, ly2 = cx + beam_half_px * cos_t, cy + beam_half_px * sin_t

        # Draw Beam Bar
        c.create_line(lx1, ly1, lx2, ly2, fill="#38bdf8", width=12, capstyle=tk.ROUND)

        # Center target zone marker on the beam
        c.create_line(cx - 18 * cos_t, cy - 18 * sin_t, cx + 18 * cos_t, cy + 18 * sin_t, fill="#22c55e", width=14)
        c.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill="#f8fafc", outline="#0f172a", width=2)
        c.create_text(cx, cy + 128, text="TRỤC XOAY & VÙNG ĐÍCH CÂN BẰNG (x = 0.0m)", fill="#4ade80", font=("Segoe UI", 9, "bold"))

        # Ball position along the tilted beam
        ball_r = 18
        dist_px = (x / self.env.max_x) * beam_half_px
        # Offset ball center perpendicular to beam surface
        nx, ny = sin_t, -cos_t
        bx = cx + dist_px * cos_t + nx * (ball_r + 6)
        by = cy + dist_px * sin_t + ny * (ball_r + 6)

        ball_color = "#4ade80" if abs(x) < 0.12 and abs(v) < 0.12 else ("#fbbf24" if abs(x) < 0.65 else "#ef4444")
        c.create_oval(bx - ball_r, by - ball_r, bx + ball_r, by + ball_r, fill=ball_color, outline="white", width=2)
        # Inner shine on ball
        c.create_oval(bx - 7, by - 7, bx - 1, by - 1, fill="white", outline="")

        # Velocity vector arrow on ball
        if abs(v) > 0.03:
            arrow_len = v * 75
            ax2 = bx + arrow_len * cos_t
            ay2 = by + arrow_len * sin_t
            c.create_line(bx, by, ax2, ay2, fill="#f97316", width=3, arrow=tk.LAST)

        # Banner
        if self.banner_timer > 0:
            c.create_rectangle(w / 2 - 205, 25, w / 2 + 205, 68, fill=self.banner_color, outline="white", width=2)
            c.create_text(w / 2, 46, text=self.banner_text, fill="white", font=("Segoe UI", 12, "bold"))

    def _draw_learning_curve(self):
        self.chart_canvas.delete("all")
        hist = self.agent.steps_history
        if not hist:
            self.chart_canvas.create_text(self.chart_w / 2, self.chart_h / 2, text="Chưa có dữ liệu Episode", fill="#64748b", font=("Segoe UI", 9))
            return

        recent = hist[-150:]
        n = len(recent)
        pts = []
        for i, val in enumerate(recent):
            px = 10 + (i / max(1, n - 1)) * (self.chart_w - 20)
            py = self.chart_h - 12 - (val / 400.0) * (self.chart_h - 26)
            pts.extend([px, py])
        if len(pts) >= 4:
            self.chart_canvas.create_line(pts, fill="#4ade80", width=2)
        self.chart_canvas.create_text(
            self.chart_w / 2, 12, text=f"150 Ep gần nhất (TB: {np.mean(recent[-20:]):.0f} / 400 bước)", fill="#e2e8f0", font=("Segoe UI", 8)
        )

    def _animate(self):
        if not self.paused:
            if self.banner_timer > 0:
                self.banner_timer -= 1
                if self.banner_timer == 0:
                    self._start_new_episode()
            else:
                for _ in range(self.steps_per_frame):
                    is_greedy = self.mode == "GREEDY_DEMO"
                    self.action = self.agent.select_action(self.s_idx, greedy=is_greedy)
                    next_state, reward, dropped = self.env.step(self.action)
                    next_s_idx = self.agent.discretize(next_state)

                    if self.mode == "LIVE_TRAIN":
                        self.agent.update(self.s_idx, self.action, reward, next_s_idx, dropped)

                    self.state = next_state
                    self.s_idx = next_s_idx
                    self.current_steps += 1

                    if dropped or self.current_steps >= self.max_ep_steps:
                        if self.mode == "LIVE_TRAIN":
                            self.agent.episodes_trained += 1
                            self.agent.steps_history.append(self.current_steps)
                            self._draw_learning_curve()

                        if self.current_steps > self.best_steps:
                            self.best_steps = self.current_steps
                            self.lbl_best.config(text=f"Kỷ lục lâu nhất: {self.best_steps} / 400 bước")

                        if not dropped and self.current_steps >= self.max_ep_steps:
                            self.banner_text = "🏆 HOÀN HẢO! CÂN BẰNG BÓNG ĐỦ 400/400 BƯỚC!"
                            self.banner_color = "#16a34a"
                            self.banner_timer = 20
                        else:
                            self.banner_text = f"💥 BÓNG LĂN RƠI KHỎI THANH Ở BƯỚC {self.current_steps}!"
                            self.banner_color = "#dc2626"
                            self.banner_timer = 12
                        break

        x, v, theta = self.state
        act_names = {0: "↙ Nghiêng Trái", 1: "• Giữ Nguyên", 2: "↘ Nghiêng Phải"}
        self.lbl_ep.config(text=f"Episodes đã học: {self.agent.episodes_trained}")
        self.lbl_step.config(text=f"Bước giữ bóng:   {self.current_steps} / 400")
        self.lbl_x.config(text=f"Vị trí bóng (x): {x:+.2f} m")
        self.lbl_v.config(text=f"Vận tốc bóng(v): {v:+.2f} m/s")
        self.lbl_theta.config(text=f"Góc nghiêng (θ): {math.degrees(theta):+.1f}°")
        self.lbl_act.config(text=f"Điều khiển:      {act_names[self.action]}")

        self._draw_scene()
        self.root.after(20, self._animate)


if __name__ == "__main__":
    root = tk.Tk()
    app = BallAndBeamVisualizerApp(root)
    root.mainloop()

