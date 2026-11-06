# Lab 2: 2-Joint Robotic Arm Target Reacher with Expected SARSA - Interactive Simulator (Tkinter GUI)
import math
import tkinter as tk
import numpy as np


# =====================================================================
# 1. 2-DOF ROBOTIC ARM ENVIRONMENT & EXPECTED SARSA AGENT
# =====================================================================
class RoboticArmEnv:
    """
    2-DOF Planar Robotic Arm Target Reacher Environment with Smooth Servo Dynamics.
    - Link lengths: L1 = 0.55 m (Upper Arm), L2 = 0.45 m (Forearm + Gripper).
    - Joint angles:
      * theta1 in [-pi, pi]:      Shoulder rotation angle
      * theta2 in [0.18, 2.70]:   Elbow flexion angle
    - Forward Kinematics:
      * Elbow:   (x1, y1) = (L1*cos(theta1), L1*sin(theta1))
      * Gripper: (xe, ye) = (x1 + L2*cos(theta1 + theta2), y1 + L2*sin(theta1 + theta2))
    - Actions (5 discrete motor commands):
      * 0: Rotate Shoulder Counter-Clockwise (+omega1)
      * 1: Rotate Shoulder Clockwise         (-omega1)
      * 2: Extend Elbow Outward              (-omega2)
      * 3: Flex Elbow Inward                 (+omega2)
      * 4: Fine-Lock / Hold Joints
    - Goal: Smoothly guide the Gripper (xe, ye) into the Target Orb (xt, yt) within radius <= 0.06 m.
    """
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.L1 = 0.55
        self.L2 = 0.45
        self.max_step = 0.055
        self.min_theta2 = 0.18
        self.max_theta2 = 2.70
        self.target_radius = 0.060
        self.num_actions = 5

        self.theta1 = 0.0
        self.theta2 = 1.2
        self.omega1 = 0.0
        self.omega2 = 0.0
        self.target_x = 0.5
        self.target_y = 0.5
        self.reset()

    @staticmethod
    def wrap_angle(angle):
        return float(((angle + math.pi) % (2.0 * math.pi)) - math.pi)

    def forward_kinematics(self):
        x1 = self.L1 * math.cos(self.theta1)
        y1 = self.L1 * math.sin(self.theta1)
        xe = x1 + self.L2 * math.cos(self.theta1 + self.theta2)
        ye = y1 + self.L2 * math.sin(self.theta1 + self.theta2)
        return (x1, y1), (xe, ye)

    def spawn_target(self, custom_xy=None):
        if custom_xy is not None:
            tx, ty = custom_xy
            r = math.hypot(tx, ty)
            if r < 0.22 or r > 0.94:
                ang = math.atan2(ty, tx)
                r = float(np.clip(r, 0.22, 0.94))
                tx, ty = r * math.cos(ang), r * math.sin(ang)
            self.target_x, self.target_y = float(tx), float(ty)
        else:
            r = float(self.rng.uniform(0.28, 0.90))
            ang = float(self.rng.uniform(-math.pi, math.pi))
            self.target_x = r * math.cos(ang)
            self.target_y = r * math.sin(ang)

    def reset(self, keep_arm_pose=False, custom_target=None):
        if not keep_arm_pose:
            self.theta1 = float(self.rng.uniform(-math.pi, math.pi))
            self.theta2 = float(self.rng.uniform(0.4, 2.3))
        self.omega1 = 0.0
        self.omega2 = 0.0
        self.spawn_target(custom_xy=custom_target)
        return self.get_state()

    def get_state(self):
        _, (xe, ye) = self.forward_kinematics()
        re = math.hypot(xe, ye)
        rt = math.hypot(self.target_x, self.target_y)
        phi_e = math.atan2(ye, xe)
        phi_t = math.atan2(self.target_y, self.target_x)

        delta_phi = self.wrap_angle(phi_t - phi_e)
        delta_r = rt - re
        dist = math.hypot(self.target_x - xe, self.target_y - ye)
        return (delta_phi, delta_r, dist)

    def step(self, action):
        delta_phi, delta_r, prev_dist = self.get_state()

        # Proportional velocity scaling prevents overshoot & chattering near zero error
        phi_scale = min(1.0, max(0.12, abs(delta_phi) / 0.16))
        r_scale = min(1.0, max(0.12, abs(delta_r) / 0.14))

        cmd_w1 = 0.0
        cmd_w2 = 0.0

        if action == 0:  # Shoulder CCW (+theta1)
            cmd_w1 = +self.max_step * phi_scale
            # Co-articulate elbow gently when radial error is also present for diagonal smoothness
            if abs(delta_r) > 0.03:
                cmd_w2 = -0.35 * np.sign(delta_r) * self.max_step * r_scale
        elif action == 1:  # Shoulder CW (-theta1)
            cmd_w1 = -self.max_step * phi_scale
            if abs(delta_r) > 0.03:
                cmd_w2 = -0.35 * np.sign(delta_r) * self.max_step * r_scale
        elif action == 2:  # Extend Elbow (-theta2 -> increases reach r)
            cmd_w2 = -self.max_step * r_scale
            # Compensate shoulder for elbow extension so gripper moves radially straight!
            cmd_w1 = +0.42 * self.max_step * r_scale
        elif action == 3:  # Flex Elbow (+theta2 -> decreases reach r)
            cmd_w2 = +self.max_step * r_scale
            cmd_w1 = -0.42 * self.max_step * r_scale
        else:  # Action 4: Fine-lock / damp velocity when very close
            cmd_w1 = 0.25 * np.clip(delta_phi, -self.max_step, self.max_step)
            cmd_w2 = -0.25 * np.clip(delta_r, -self.max_step, self.max_step)

        # Low-pass velocity filter for fluid industrial robot motion
        self.omega1 = 0.55 * self.omega1 + 0.45 * cmd_w1
        self.omega2 = 0.55 * self.omega2 + 0.45 * cmd_w2

        self.theta1 = self.wrap_angle(self.theta1 + self.omega1)
        self.theta2 = float(np.clip(self.theta2 + self.omega2, self.min_theta2, self.max_theta2))

        state = self.get_state()
        dist = state[2]
        reached = dist <= self.target_radius

        reward = (prev_dist - dist) * 28.0 - 0.12
        if action == 4 and dist < 0.12:
            reward += 0.5
        if reached:
            reward += 50.0

        return state, reward, reached


class ExpectedSarsaArmAgent:
    """Tabular Expected SARSA Agent for 2-DOF Robotic Arm Control."""
    def __init__(self, env, n_bins=11, alpha=0.25, gamma=0.96, epsilon=0.12):
        self.env = env
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Symmetric bins with a dedicated center deadzone bin [-0.035, +0.035]
        self.phi_bins = np.array([-0.80, -0.40, -0.18, -0.08, -0.03, 0.03, 0.08, 0.18, 0.40, 0.80])
        self.r_bins = np.array([-0.45, -0.25, -0.12, -0.05, -0.02, 0.02, 0.05, 0.12, 0.25, 0.45])
        self.reset_q()

    def reset_q(self):
        self.Q = np.zeros((self.n_bins, self.n_bins, self.env.num_actions))
        self.episodes_trained = 0
        self.steps_history = []

    def discretize(self, state):
        delta_phi, delta_r, _ = state
        p_idx = int(np.digitize(delta_phi, self.phi_bins))
        r_idx = int(np.digitize(delta_r, self.r_bins))
        return (p_idx, r_idx)

    def select_action(self, s_idx, greedy=False, eps=None):
        use_eps = self.epsilon if eps is None else eps
        if not greedy and self.env.rng.random() < use_eps:
            return int(self.env.rng.integers(0, self.env.num_actions))
        return int(np.argmax(self.Q[s_idx]))

    def update(self, s_idx, action, reward, next_s_idx, done, eps=None):
        """Expected SARSA update rule."""
        if done:
            expected_next_q = 0.0
        else:
            use_eps = self.epsilon if eps is None else eps
            q_next = self.Q[next_s_idx]
            best_a = int(np.argmax(q_next))
            probs = np.full(self.env.num_actions, use_eps / self.env.num_actions)
            probs[best_a] += 1.0 - use_eps
            expected_next_q = float(np.dot(probs, q_next))

        td_target = reward + self.gamma * expected_next_q
        self.Q[s_idx + (action,)] += self.alpha * (td_target - self.Q[s_idx + (action,)])

    def train_episodes_fast(self, num_episodes=300, max_steps=160):
        for _ in range(num_episodes):
            state = self.env.reset()
            s_idx = self.discretize(state)
            eps = max(0.02, 0.25 * (0.99 ** self.episodes_trained))
            steps = 0
            for _ in range(max_steps):
                action = self.select_action(s_idx, greedy=False, eps=eps)
                next_state, reward, reached = self.env.step(action)
                next_s_idx = self.discretize(next_state)
                self.update(s_idx, action, reward, next_s_idx, reached, eps=eps)
                s_idx = next_s_idx
                steps += 1
                if reached:
                    break
            self.episodes_trained += 1
            self.steps_history.append(steps)


# =====================================================================
# 2. INTERACTIVE GUI WINDOW (TKINTER ROBOTIC ARM WORKSPACE SIMULATOR)
# =====================================================================
class RoboticArmVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lab 2: 2-Joint Robotic Arm Expected SARSA - Mô Phỏng Cánh Tay Robot Gắp Mục Tiêu")
        self.root.geometry("980x650")
        self.root.configure(bg="#1e293b")

        self.env = RoboticArmEnv(seed=42)
        self.agent = ExpectedSarsaArmAgent(self.env)

        self.mode = "LIVE_TRAIN"
        self.paused = False
        self.steps_per_frame = 1
        self.max_ep_steps = 160
        self.targets_caught = 0
        self.banner_timer = 0
        self.gripper_trail = []

        self._build_ui()
        self._start_new_episode()
        self._animate()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#0f172a", padx=15, pady=10)
        top.pack(fill=tk.X)

        tk.Label(
            top,
            text="2-DOF ROBOTIC ARM REACHER - THUẬT TOÁN EXPECTED SARSA (CLICK CHUỘT ĐỂ ĐẶT MỤC TIÊU)",
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        ).pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            top, text="Chế độ: Đang tự học trực tiếp (Live Expected SARSA)", font=("Segoe UI", 11, "bold"), fg="#fbbf24", bg="#0f172a"
        )
        self.lbl_status.pack(side=tk.RIGHT)

        main_frame = tk.Frame(self.root, bg="#1e293b", padx=12, pady=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas_w, self.canvas_h = 650, 450
        self.canvas = tk.Canvas(
            main_frame, width=self.canvas_w, height=self.canvas_h, bg="#0b1120", highlightthickness=1, highlightbackground="#334155", cursor="crosshair"
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Right Dashboard
        side = tk.Frame(main_frame, width=285, bg="#0f172a", padx=14, pady=12)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        tk.Label(side, text="THÔNG SỐ CÁNH TAY ROBOT", font=("Segoe UI", 11, "bold"), fg="#f8fafc", bg="#0f172a").pack(anchor="w", pady=(0, 8))

        self.lbl_ep = tk.Label(side, text="Episodes đã học: 0", font=("Consolas", 11), fg="#e2e8f0", bg="#0f172a")
        self.lbl_ep.pack(anchor="w", pady=2)

        self.lbl_caught = tk.Label(side, text="Mục tiêu đã gắp: 0", font=("Consolas", 11, "bold"), fg="#4ade80", bg="#0f172a")
        self.lbl_caught.pack(anchor="w", pady=2)

        self.lbl_t1 = tk.Label(side, text="Khớp vai (θ1):   +0.0°", font=("Consolas", 11), fg="#38bdf8", bg="#0f172a")
        self.lbl_t1.pack(anchor="w", pady=2)

        self.lbl_t2 = tk.Label(side, text="Khớp khuỷu (θ2): +68.8°", font=("Consolas", 11), fg="#fbbf24", bg="#0f172a")
        self.lbl_t2.pack(anchor="w", pady=2)

        self.lbl_dist = tk.Label(side, text="Khoảng cách:     0.50 m", font=("Consolas", 11), fg="#e2e8f0", bg="#0f172a")
        self.lbl_dist.pack(anchor="w", pady=2)

        self.lbl_act = tk.Label(side, text="Lệnh mô-tơ:      Giữ nguyên", font=("Consolas", 10, "bold"), fg="#a78bfa", bg="#0f172a")
        self.lbl_act.pack(anchor="w", pady=4)

        tk.Label(side, text="💡 Mẹo: Click chuột trực tiếp lên màn hình bên trái để đặt vị trí mục tiêu mới!", font=("Segoe UI", 9, "italic"), fg="#38bdf8", bg="#0f172a", wraplength=250, justify=tk.LEFT).pack(anchor="w", pady=(4, 6))

        tk.Label(side, text="ĐƯỜNG CONG HỌC (SỐ BƯỚC GẮP TRÚNG)", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#0f172a").pack(anchor="w", pady=(6, 4))
        self.chart_w, self.chart_h = 255, 130
        self.chart_canvas = tk.Canvas(side, width=self.chart_w, height=self.chart_h, bg="#1e293b", highlightthickness=1, highlightbackground="#334155")
        self.chart_canvas.pack()

        tk.Label(side, text="Tốc độ mô phỏng (Steps/Frame):", font=("Segoe UI", 9), fg="#cbd5e1", bg="#0f172a").pack(anchor="w", pady=(8, 2))
        self.speed_scale = tk.Scale(side, from_=1, to=10, orient=tk.HORIZONTAL, bg="#0f172a", fg="#f8fafc", highlightthickness=0, command=self._on_speed)
        self.speed_scale.set(1)
        self.speed_scale.pack(fill=tk.X)

        # Bottom Controls
        btn_frame = tk.Frame(self.root, bg="#1e293b", padx=15, pady=10)
        btn_frame.pack(fill=tk.X)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "padx": 11, "pady": 6, "bd": 0, "cursor": "hand2"}

        tk.Button(
            btn_frame, text="⚡ Huấn Luyện Nhanh +300 Episodes", bg="#2563eb", fg="white", command=self.fast_train_300, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🏆 Xem Robot Tự Động Gắp Mục Tiêu (Greedy)", bg="#16a34a", fg="white", command=self.set_greedy_mode, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🎯 Hất Mục Tiêu Ra Mép", bg="#9333ea", fg="white", command=self._push_target_to_edge, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🔄 Xem Tự Học (Live Train)", bg="#d97706", fg="white", command=self.set_live_train_mode, **btn_style
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="🗑 Reset Học Lại", bg="#dc2626", fg="white", command=self.reset_all, **btn_style
        ).pack(side=tk.RIGHT, padx=4)

    def _on_speed(self, val):
        self.steps_per_frame = int(val)

    def _on_canvas_click(self, event):
        w = self.canvas.winfo_width() or self.canvas_w
        h = self.canvas.winfo_height() or self.canvas_h
        cx, cy = w / 2, h / 2
        scale = 195.0
        wx = (event.x - cx) / scale
        wy = (cy - event.y) / scale
        self._start_new_episode(keep_pose=True, custom_target=(wx, wy))

    def _push_target_to_edge(self):
        _, (xe, ye) = self.env.forward_kinematics()
        ang = math.atan2(ye, xe) + math.pi + float(self.env.rng.uniform(-0.55, 0.55))
        r = float(self.env.rng.uniform(0.80, 0.92))
        self._start_new_episode(keep_pose=True, custom_target=(r * math.cos(ang), r * math.sin(ang)))

    def _start_new_episode(self, keep_pose=True, custom_target=None):
        self.state = self.env.reset(keep_arm_pose=keep_pose, custom_target=custom_target)
        self.s_idx = self.agent.discretize(self.state)
        self.current_steps = 0
        self.action = 4
        self.banner_timer = 0
        self.gripper_trail.clear()

    def fast_train_300(self):
        self.agent.train_episodes_fast(num_episodes=300, max_steps=self.max_ep_steps)
        self.set_greedy_mode()
        self._draw_learning_curve()

    def set_greedy_mode(self):
        self.mode = "GREEDY_DEMO"
        self.lbl_status.config(text="Chế độ: Robot Tự Động Gắp Mục Tiêu (Greedy)", fg="#4ade80")
        self._start_new_episode(keep_pose=True)

    def set_live_train_mode(self):
        self.mode = "LIVE_TRAIN"
        self.lbl_status.config(text="Chế độ: Đang tự học trực tiếp (Live Expected SARSA)", fg="#fbbf24")
        self._start_new_episode(keep_pose=True)

    def reset_all(self):
        self.agent.reset_q()
        self.targets_caught = 0
        self.lbl_caught.config(text="Mục tiêu đã gắp: 0")
        self.set_live_train_mode()
        self._draw_learning_curve()

    def _world_to_screen(self, x, y):
        w = self.canvas.winfo_width() or self.canvas_w
        h = self.canvas.winfo_height() or self.canvas_h
        scale = 195.0
        return (w / 2 + x * scale, h / 2 - y * scale)

    def _draw_scene(self):
        c = self.canvas
        c.delete("all")
        w = self.canvas.winfo_width() or self.canvas_w
        h = self.canvas.winfo_height() or self.canvas_h
        cx, cy = w / 2, h / 2
        scale = 195.0

        r_max = (self.env.L1 + self.env.L2) * scale
        r_min = 0.22 * scale
        c.create_oval(cx - r_max, cy - r_max, cx + r_max, cy + r_max, outline="#1e293b", width=2, dash=(4, 4))
        c.create_oval(cx - r_min, cy - r_min, cx + r_min, cy + r_min, outline="#1e293b", width=1, dash=(2, 4))
        c.create_line(cx - r_max, cy, cx + r_max, cy, fill="#111827")
        c.create_line(cx, cy - r_max, cx, cy + r_max, fill="#111827")

        tx, ty = self._world_to_screen(self.env.target_x, self.env.target_y)
        tr = self.env.target_radius * scale
        c.create_oval(tx - tr - 5, ty - tr - 5, tx + tr + 5, ty + tr + 5, outline="#22c55e", width=1, dash=(2, 2))
        c.create_oval(tx - tr, ty - tr, tx + tr, ty + tr, fill="#16a34a", outline="#bbf7d0", width=2)
        c.create_text(tx, ty - tr - 14, text="MỤC TIÊU", fill="#4ade80", font=("Segoe UI", 9, "bold"))

        (x1, y1), (xe, ye) = self.env.forward_kinematics()
        sx1, sy1 = self._world_to_screen(x1, y1)
        sxe, sye = self._world_to_screen(xe, ye)

        self.gripper_trail.append((sxe, sye))
        if len(self.gripper_trail) > 45:
            self.gripper_trail.pop(0)
        if len(self.gripper_trail) >= 2:
            flat_t = [coord for pt in self.gripper_trail for coord in pt]
            c.create_line(flat_t, fill="#0284c7", width=2)

        c.create_line(cx, cy, sx1, sy1, fill="#38bdf8", width=12, capstyle=tk.ROUND)
        c.create_line(sx1, sy1, sxe, sye, fill="#fbbf24", width=9, capstyle=tk.ROUND)

        c.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, fill="#475569", outline="white", width=2)
        c.create_oval(sx1 - 9, sy1 - 9, sx1 + 9, sy1 + 9, fill="#0f172a", outline="#f8fafc", width=2)

        ang = self.env.theta1 + self.env.theta2
        for sign in (-1, 1):
            claw_a = ang + sign * 0.45
            gx1 = sxe + 14 * math.cos(claw_a)
            gy1 = sye - 14 * math.sin(claw_a)
            c.create_line(sxe, sye, gx1, gy1, fill="#f8fafc", width=3, capstyle=tk.ROUND)
        c.create_oval(sxe - 6, sye - 6, sxe + 6, sye + 6, fill="#ef4444", outline="white", width=2)

        if self.banner_timer > 0:
            c.create_rectangle(w / 2 - 190, 18, w / 2 + 190, 56, fill="#16a34a", outline="white", width=2)
            c.create_text(
                w / 2, 37, text=f"🎯 ĐÃ GẮP TRÚNG MỤC TIÊU SAU {self.last_win_steps} BƯỚC!",
                fill="white", font=("Segoe UI", 11, "bold")
            )

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
            py = self.chart_h - 12 - ((val - 5) / (self.max_ep_steps - 5)) * (self.chart_h - 26)
            py = max(8, min(self.chart_h - 8, py))
            pts.extend([px, py])
        if len(pts) >= 4:
            self.chart_canvas.create_line(pts, fill="#4ade80", width=2)
        self.chart_canvas.create_text(
            self.chart_w / 2, 12, text=f"150 Ep gần nhất (TB: {np.mean(recent[-20:]):.1f} bước)", fill="#e2e8f0", font=("Segoe UI", 8)
        )

    def _animate(self):
        if not self.paused:
            if self.banner_timer > 0:
                self.banner_timer -= 1
                if self.banner_timer == 0:
                    self._start_new_episode(keep_pose=True)
            else:
                for _ in range(self.steps_per_frame):
                    is_greedy = self.mode == "GREEDY_DEMO"
                    self.action = self.agent.select_action(self.s_idx, greedy=is_greedy)
                    next_state, reward, reached = self.env.step(self.action)
                    next_s_idx = self.agent.discretize(next_state)

                    if self.mode == "LIVE_TRAIN":
                        self.agent.update(self.s_idx, self.action, reward, next_s_idx, reached)

                    self.state = next_state
                    self.s_idx = next_s_idx
                    self.current_steps += 1

                    if reached or self.current_steps >= self.max_ep_steps:
                        if self.mode == "LIVE_TRAIN":
                            self.agent.episodes_trained += 1
                            self.agent.steps_history.append(self.current_steps)
                            self._draw_learning_curve()

                        if reached:
                            self.targets_caught += 1
                            self.last_win_steps = self.current_steps
                            self.lbl_caught.config(text=f"Mục tiêu đã gắp: {self.targets_caught}")
                            self.banner_timer = 16
                        else:
                            self._start_new_episode(keep_pose=True)
                        break

        act_names = {
            0: "↺ Xoay Vai Trái (+θ1)",
            1: "↻ Xoay Vai Phải (-θ1)",
            2: "↗ Duỗi Khuỷu (-θ2)",
            3: "↙ Gập Khuỷu (+θ2)",
            4: "⚖ Khóa Tinh Chỉnh",
        }
        self.lbl_ep.config(text=f"Episodes đã học: {self.agent.episodes_trained}")
        self.lbl_t1.config(text=f"Khớp vai (θ1):   {math.degrees(self.env.theta1):+.1f}°")
        self.lbl_t2.config(text=f"Khớp khuỷu (θ2): {math.degrees(self.env.theta2):+.1f}°")
        self.lbl_dist.config(text=f"Khoảng cách:     {self.state[2]:.2f} m")
        self.lbl_act.config(text=f"Lệnh mô-tơ:      {act_names[self.action]}")

        self._draw_scene()
        self.root.after(20, self._animate)


if __name__ == "__main__":
    root = tk.Tk()
    app = RoboticArmVisualizerApp(root)
    root.mainloop()
