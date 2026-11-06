# KỊCH BẢN THUYẾT TRÌNH BÁO CÁO LAB MÔN REL301m (REINFORCEMENT LEARNING)

> **Tài liệu trình chiếu chính:** Mở trực tiếp file **[index.html](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/index.html)** trên trình duyệt (hoặc chạy 3 ứng dụng Python GUI trong từng thư mục Lab).
>
> **3 câu hỏi trọng tâm xuyên suốt bài thuyết trình:**
> 1. **Hệ thống hoạt động như thế nào?** (Cơ chế vật lý, quy trình tương tác Agent – Environment theo chuẩn MDP: $S_t \to A_t \to R_{t+1}, S_{t+1}$, và cách rời rạc hóa không gian trạng thái liên tục).
> 2. **Sử dụng phương pháp RL nào?** (Thuật toán TD Control cụ thể, công thức toán học cập nhật bảng $Q(s,a)$, chiến lược khám phá $\varepsilon$-greedy / Optimistic Initialization, và lý do chọn thuật toán đó).
> 3. **Nội dung đối chiếu với chương trình học (`slides/`) như thế nào?** (Ánh xạ chính xác từng phần trong Lab với các bài giảng Chương 1, Chương 2 và Chương 3 của môn `REL301m`).

---

## PHẦN 1: KIỂM TRA & ĐỐI CHIẾU VỚI CHƯƠNG TRÌNH HỌC (`slides/`)

Toàn bộ 3 bài Lab được thiết kế **bám sát 100% nội dung lý thuyết trong thư mục `slides/` của môn REL301m**, cụ thể bao phủ từ Chương 1 (MDP & Bellman), Chương 2 (Temporal Difference Control) đến Chương 3 (Biểu diễn trạng thái liên tục & Khám phá):

| Nội dung trong 3 bài Lab | Bài giảng tương ứng trong thư mục `slides/` | Ý nghĩa áp dụng thực tế trong Lab |
| :--- | :--- | :--- |
| **Mô hình hóa MDP $(S, A, P, R, \gamma)$ & Mục tiêu tối đa hóa Return $G_t$** | • `1.4 Introduction to Markov Decision Processes .pptx`<br>• `1.5 Goal of Reinforcement Learning .pptx`<br>• `1.7 Policies and Value Functions.pptx`<br>• `1.8 Bellman Equations.pptx`<br>• `1.9 Optimality (Optimal Policies & Value Functions).pptx` | Định nghĩa không gian trạng thái $S$, tập hành động điều khiển mô-tơ $A$, hàm phần thưởng dẫn hướng $R$ (Reward Shaping / Sparse Penalty) và hệ số chiết khấu $\gamma \in [0.96, 0.99]$. |
| **Cân bằng Khám phá & Khai thác ($\varepsilon$-greedy)** | • `1.3 Exploration vs. Exploitation Tradeoff.pptx`<br>• `1.12 Generalized Policy Iteration.pptx` | Với xác suất $\varepsilon$ chọn hành động ngẫu nhiên (Explore) và $1-\varepsilon$ chọn hành động $\arg\max_a Q(s,a)$ (Exploit); giảm dần $\varepsilon$ theo số episode. |
| **HọcTD (Temporal Difference): Model-free, Online & Bootstrapping** | • `2.5 Introduction to Temporal Difference Learning.pptx`<br>• `2.6 Advantages of Temporal Difference.pptx` | Không cần biết trước phương trình đạo hàm của môi trường (**Model-free**); cập nhật giá trị $Q(s,a)$ ngay sau **mỗi bước thời gian** $\Delta t$ dựa trên **TD Target** và **TD Error** $\delta_t$ thay vì phải đợi hết episode như Monte Carlo. |
| **Lab 1: Thuật toán Q-Learning (Off-policy TD Control)** | • `2.8 Off-policy Temporal Difference Control Q-learning .pptx` | Học trực tiếp hàm giá trị hành động tối ưu $Q^*(s,a)$ thông qua toán tử $\max_{a'} Q(s',a')$ bất kể hành động khám phá tiếp theo là gì. |
| **Lab 2: Thuật toán Expected SARSA** | • `2.9 Expected Sarsa.pptx` | Thay thế giá trị mẫu đơn lẻ $Q(s',a')$ bằng **giá trị kỳ vọng** $\sum_{a'} \pi(a'|s') Q(s',a')$, giúp triệt tiêu phương sai (variance) do chọn hành động ngẫu nhiên trong điều khiển khớp tay robot. |
| **Lab 3: Thuật toán SARSA (On-policy TD Control)** | • `2.7 Temporal Difference for Control.pptx`<br>• `3.6 Episodic Sarsa with Function Approximation.pptx` | Học trên chính sách đang thực thi (On-policy) thông qua bộ 5 phần tử $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ cho bài toán Episodic Mountain Car. |
| **Rời rạc hóa không gian liên tục (State Aggregation / Binning)** | • `3.5 Feature Construction for Linear Methods.pptx` *(mục State Aggregation, Coarse/Tile Partitioning, Discrimination vs. Generalization)* | Chia không gian vật lý liên tục (vị trí, vận tốc, góc) thành lưới các ô rời rạc (Bins/Tiles). Cân bằng giữa **Discrimination** (chia mịn quanh vùng cân bằng $0$ để đứng êm) và **Generalization** (gom nhóm ở vùng biên để học nhanh). |
| **Khởi tạo lạc quan (Optimistic Initial Values)** | • `3.7 Exploration under Function Approximation.pptx` *(mục Optimistic Initial Values in the Tabular)* | Khởi tạo $Q_0(s,a) = 0.0$ trong khi phần thưởng mỗi bước ở Lab 3 là $R = -1.0$, buộc Agent tự động khám phá có hệ thống toàn bộ không gian trạng thái trước khi tìm thấy đỉnh đồi. |

---

## PHẦN 2: KỊCH BẢN THUYẾT TRÌNH CHI TIẾT (TRẢ LỜI 3 CÂU HỎI CHO TỪNG LAB)

### 🎤 MỞ ĐẦU (1 phút) — Giới thiệu trên Tab `📊 Tổng quan & So sánh` của `index.html`

> *"Em chào thầy ạ. Hôm nay em xin trình bày 3 bài Lab môn **Reinforcement Learning (REL301m)**. Thay vì dùng các bài toán lưới tĩnh đơn giản, em xây dựng **3 hệ thống điều khiển vật lý & cơ điện tử tương tác trực tiếp ở tốc độ 60 FPS**, được tích hợp chung vào một Dashboard trực quan kèm 4 biểu đồ thời gian thực.*
>
> *Cả 3 bài Lab đều thuộc nhóm **Model-free Temporal Difference (TD) Control** trong **Chương 2** kết hợp với kỹ thuật **State Aggregation / Binning** và **Exploration** trong **Chương 1 & Chương 3** của giáo trình. Với mỗi bài Lab, em xin trả lời rõ 2 câu hỏi cốt lõi: **(1) Hệ thống hoạt động như thế nào?** và **(2) Hệ thống sử dụng phương pháp RL nào?**"*

---

### ⚖️ BÀI LAB 1: THANH BẬP BÊNH CÂN BẰNG BÓNG (BALL & BEAM BALANCING)

*(Chuyển sang Tab **`⚖️ Lab 1 · Q-Learning`** trên [index.html](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/index.html) hoặc mở [ball_and_beam_qlearning.py](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/Lab1_Ball_And_Beam_QLearning/ball_and_beam_qlearning.py))*

#### 1. Hệ thống hoạt động như thế nào?
- **Cơ chế vật lý:** Một viên bi thép lăn trên thanh ngang dài $2.0\text{ m}$ ($x \in [-1.0, +1.0]\text{ m}$), thanh có thể nghiêng góc $\theta \in [-14^\circ, +14^\circ]$. Theo cơ học cổ điển, gia tốc lăn của viên bi tỷ lệ với $\sin\theta$:
  $$a = \frac{5}{7} g \sin\theta - \mu v$$
  Góc nghiêng thanh $\theta$ được điều khiển qua mô-tơ servo có bộ lọc quán tính (low-pass damping) để thanh nghiêng mượt mà, không bị giật cục.
- **Chu trình tương tác MDP ở mỗi bước thời gian ($\Delta t = 0.025\text{ s}$):**
  1. **Quan sát trạng thái ($S_t$):** Cảm biến đo vị trí bóng $x$, vận tốc bóng $v$, và góc nghiêng thanh $\theta$. Để chống trễ pha (overshoot), hệ thống kết hợp vị trí và vận tốc thành biến trạng thái dẫn hướng pha $(x + 1.6v)$ cùng góc $\theta$, sau đó rời rạc hóa (**State Aggregation** – *Slide 3.5*) thành lưới **$11 \times 11 = 121$ trạng thái** (với các ô chia siêu mịn quanh tâm $0$ tạo vùng "deadzone" chống rung).
  2. **Ra quyết định hành động ($A_t$):** Agent chọn 1 trong **3 hành động**:
     - $a = 0$: Nghiêng thanh sang trái (hãm bóng đang lao sang phải hoặc đẩy bóng về trái).
     - $a = 1$: Giữ thanh cân bằng về $0^\circ$.
     - $a = 2$: Nghiêng thanh sang phải.
  3. **Nhận phần thưởng ($R_{t+1}$):**
     - Nếu bóng rơi khỏi hai đầu thanh ($|x| > 1.0\text{ m}$): Phạt nặng **$-50$ điểm** và kết thúc episode.
     - Nếu bóng còn trên thanh: Thưởng dương khi bóng gần tâm, vận tốc nhỏ và thanh nằm ngang; thưởng thêm **$+0.6$** khi bóng nằm yên trong vùng tâm và chọn hành động giữ cân bằng.

#### 2. Sử dụng phương pháp RL nào?
- **Thuật toán:** **Q-Learning (Off-policy Temporal Difference Control)** — *bám sát bài giảng `2.8 Off-policy Temporal Difference Control Q-learning .pptx`*.
- **Công thức cập nhật sau mỗi bước:**
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ \underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')}_{\text{TD Target}} - Q(S_t, A_t) \Big]$$
  Với hệ số chiết khấu $\gamma = 0.98$, tốc độ học $\alpha$ giảm dần từ $0.25 \to 0.05$, và chiến lược khám phá $\varepsilon$-greedy ($\varepsilon: 0.25 \to 0.02$, *Slide 1.3*).
- **Tại sao chọn Q-Learning cho Lab 1?**
  - Bài toán giữ thăng bằng đòi hỏi một **chính sách tối ưu chuẩn xác tuyệt đối** $\pi^*(s) = \arg\max_a Q^*(s,a)$ tại vùng cân bằng để thanh không bị lệch.
  - Vì Q-Learning là **Off-policy** (*Slide 2.8, trang 8–12*), trong lúc tập luyện dù Agent có đôi lúc chọn hành động ngẫu nhiên ($\varepsilon$-greedy) làm rơi bóng, công thức cập nhật vẫn lấy $\max_{a'} Q(S_{t+1}, a')$ — tức là học theo quỹ đạo thăng bằng tốt nhất có thể mà không bị nhiễu bởi các bước khám phá sai lầm.

#### 3. Thao tác Demo trực tiếp trên `index.html`:
- Bấm **`⚡ Huấn luyện +350 ep`** $\to$ chỉ vào **Biểu đồ 1 (Đường cong học)** tăng vọt lên mức tối đa **400/400 bước** (100% thành công).
- Chỉ vào **Biểu đồ 3 (Bản đồ chính sách $\pi(s)$)**: Giải thích vùng màu xanh lá ở chính giữa là vùng giữ thanh phẳng, hai bên là vùng nghiêng trái/phải đối xứng để phanh quán tính.
- Bấm **`💨 Hất bóng về mép`** nhiều lần để chứng minh Agent nghiêng thanh đón lõng, phanh bóng lại trước mép vực rồi đưa bóng về đúng vạch $0.000\text{ m}$ và đứng im phẳng lặng.

---

### 🦾 BÀI LAB 2: CÁNH TAY ROBOT 2 KHỚP GẮP MỤC TIÊU (2-DOF ROBOTIC ARM REACHER)

*(Chuyển sang Tab **`🦾 Lab 2 · Expected SARSA`** trên [index.html](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/index.html) hoặc mở [robotic_arm_sarsa.py](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/Lab2_Robotic_Arm_Reacher_ExpectedSarsa/robotic_arm_sarsa.py))*

#### 1. Hệ thống hoạt động như thế nào?
- **Cơ chế vật lý (Động học thuận 2-DOF):** Cánh tay robot hoạt động trên mặt phẳng gồm khớp vai $\theta_1$ (khâu $L_1 = 0.55\text{ m}$) và khớp khuỷu $\theta_2$ (khâu $L_2 = 0.45\text{ m}$). Tọa độ đầu kẹp (end-effector) được tính bằng phương trình động học thuận:
  $$\begin{cases} x_e = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2) \\ y_e = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2) \end{cases}$$
- **Biểu diễn trạng thái tương đối trong hệ tọa độ cực ($S_t$):**
  - Thay vì dùng tọa độ tuyệt đối khiến cánh tay chỉ biết gắp 1 điểm cố định, hệ thống đo **sai số góc hướng $\Delta\phi = \phi_{\text{target}} - \phi_{\text{ee}}$** và **sai số bán kính vươn $\Delta r = r_{\text{target}} - r_{\text{ee}}$** từ đầu kẹp tới mục tiêu.
  - Nhờ biểu diễn sai số tương đối này (**Feature Construction & Generalization** – *Slide 3.5*), chỉ với lưới **$11 \times 11 = 121$ trạng thái**, Agent học một lần nhưng **gắp được mọi vị trí mục tiêu bất kỳ** trong vùng làm việc!
- **Tập hành động ($A_t$ – 5 lệnh điều khiển mô-tơ khớp):**
  - $a = 0, 1$: Xoay khớp vai sang trái / sang phải (kèm bù khuỷu tự động).
  - $a = 2, 3$: Duỗi khuỷu tay vươn xa / Gập khuỷu tay thu gần.
  - $a = 4$: Khóa tinh chỉnh đồng thời cả 2 khớp khi đã áp sát mục tiêu.
- **Hàm phần thưởng ($R_{t+1}$):** Thưởng theo độ giảm khoảng cách $28.0 \times (d_{t-1} - d_t) - 0.12$ mỗi bước, và thưởng lớn **$+50$ điểm** ngay khi đầu kẹp chạm trúng mục tiêu ($d_t \le 0.055\text{ m}$).

#### 2. Sử dụng phương pháp RL nào?
- **Thuật toán:** **Expected SARSA** — *bám sát bài giảng `2.9 Expected Sarsa.pptx` và `3.6 Episodic Sarsa with Function Approximation.pptx` (trang 16–18)*.
- **Công thức cập nhật sau mỗi bước:**
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$
  Trong đó phân phối xác suất $\pi(a'|S_{t+1})$ của chính sách $\varepsilon$-greedy là:
  $$\pi(a'|S_{t+1}) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|A|} & \text{nếu } a' = \arg\max_b Q(S_{t+1}, b) \\ \frac{\varepsilon}{|A|} & \text{nếu } a' \neq \arg\max_b Q(S_{t+1}, b) \end{cases}$$
- **Tại sao chọn Expected SARSA cho Lab 2?**
  - Theo đúng **Slide 2.9 (trang 8, 9, 15 – Stability & Variance Reduction)**: Cánh tay robot có tới 5 hành động điều khiển 2 khớp liên kết phi tuyến. Nếu dùng SARSA thông thường, việc bốc thăm ngẫu nhiên 1 hành động $A_{t+1}$ sẽ tạo ra phương sai (variance) lớn làm quỹ đạo cánh tay bị giật.
  - **Expected SARSA** lấy trung bình có trọng số qua tất cả 5 hành động ở trạng thái kế tiếp, giúp loại bỏ hoàn toàn nhiễu ngẫu nhiên trong bước cập nhật, cho phép dùng tốc độ học cao ($\alpha = 0.25$) mà hàm giá trị $V(s)$ vẫn hội tụ cực kỳ mượt và ổn định.

#### 3. Thao tác Demo trực tiếp trên `index.html`:
- Bấm **`⚡ Huấn luyện +300 ep`** $\to$ chỉ vào **Biểu đồ 1**: Số bước trung bình để gắp trúng mục tiêu giảm nhanh xuống chỉ còn **~30–40 bước**, tỉ lệ thành công đạt **100%**.
- **Click chuột trực tiếp vào nhiều điểm khác nhau** trên màn hình mô phỏng để đặt mục tiêu mới ngay theo yêu cầu của thầy — cánh tay sẽ lập tức vươn ra gắp trúng 100% với đường cong quỹ đạo mượt mà.

---

### 🚗 BÀI LAB 3: XE LẤY ĐÀ LEO ĐỒI (MOUNTAIN CAR CONTROL)

*(Chuyển sang Tab **`🚗 Lab 3 · SARSA`** trên [index.html](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/index.html) hoặc mở [mountain_car_sarsa.py](file:///c:/PhatVo_Important/Codes/My_Codes/Academy/REL301m/doc/Lab-Student/lab/Lab3_Mountain_Car_SARSA/mountain_car_sarsa.py))*

#### 1. Hệ thống hoạt động như thế nào?
- **Cơ chế vật lý (Under-actuated System):** Chiếc xe nằm dưới đáy thung lũng ($x \in [-0.6, -0.4]$) và cần leo lên lá cờ trên đỉnh đồi bên phải ($x_{\text{goal}} = 0.5$). Phương trình cập nhật vận tốc và vị trí ở mỗi bước:
  $$\begin{cases} v_{t+1} = \text{clip}\big(v_t + 0.001(\text{action} - 1) - 0.0025\cos(3x_t),\; -0.07,\; +0.07\big) \\ x_{t+1} = \text{clip}\big(x_t + v_{t+1},\; -1.2,\; +0.5\big) \end{cases}$$
  Điểm mấu chốt: Lực đẩy cực đại của động cơ ($0.001$) **nhỏ hơn 2.5 lần** so với lực cản trọng trường ở sườn dốc ($0.0025\cos(3x)$). Nếu chỉ nhấn ga tiến thẳng sang phải, xe chắc chắn bị tụt dốc.
- **Không gian trạng thái & Hành động:**
  - **Trạng thái ($S_t$):** Cặp biến liên tục $(x_t, v_t)$ được chia lưới (**Binning / Tile Partitioning** – *Slide 3.5*) thành bảng Q kích thước **$16 \times 16 \times 3$** (hoặc $20 \times 20 \times 3$ trong bản Python).
  - **Hành động ($A_t$):** $0 =$ Đẩy trái (Lùi), $1 =$ Thả trôi (Coast), $2 =$ Đẩy phải (Tiến).
  - **Phần thưởng ($R_{t+1}$):** Phạt **$-1$ điểm** ở mọi bước thời gian cho tới khi chạm đỉnh đồi ($x \ge 0.5$). Không hề có phần thưởng gợi ý giữa chừng (**Sparse / Delayed Reward**).

#### 2. Sử dụng phương pháp RL nào?
- **Thuật toán:** **SARSA (On-policy Temporal Difference Control)** kết hợp **Optimistic Initial Values** — *bám sát bài giảng `2.7 Temporal Difference for Control.pptx`, `3.6 Episodic Sarsa with Function Approximation.pptx` và `3.7 Exploration under Function Approximation.pptx`*.
- **Công thức cập nhật dựa trên bộ 5 phần tử $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:**
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \Big]$$
  Với $\alpha = 0.22$, $\gamma = 0.99$, $\varepsilon = 0.05 \to 0.01$.
- **Kỹ thuật Khởi tạo lạc quan (Optimistic Initial Values – *Slide 3.7, trang 3–4*):**
  - Vì mỗi bước bị phạt $R = -1$, giá trị thực của mọi trạng thái đều âm ($V^*(s) \approx -150$).
  - Khi khởi tạo bảng $Q_0(s,a) = 0.0$, bất kỳ ô trạng thái nào vừa được đi qua sẽ bị giảm giá trị xuống âm (ví dụ $-0.22$), khiến các ô chưa thăm (vẫn bằng $0.0$) trở nên hấp dẫn hơn. Nhờ vậy, Agent tự động khám phá có hệ thống toàn bộ lòng chảo $(x, v)$ và nhanh chóng phát hiện ra chiến lược **lùi lên dốc trái lấy thế năng rồi phóng vọt lên đỉnh đồi phải**.

#### 3. Thao tác Demo trực tiếp trên `index.html`:
- Bấm **`⚡ Huấn luyện +500 ep`** $\to$ chỉ vào **Biểu đồ 3 (Bản đồ chính sách $\pi(s)$)** và **Biểu đồ 4 (Hàm giá trị $V(s)$)**:
  - Nửa dưới bản đồ ($v < 0$, xe đang lùi): Phủ màu cam (**Đẩy trái**) để lấy thêm đà.
  - Nửa trên bản đồ ($v > 0$, xe đang lao tới): Phủ màu xanh dương (**Đẩy phải**) để cộng hưởng quán tính vượt đỉnh.
  - Bản đồ nhiệt $V(s)$ hiện rõ hình xoắn ốc từ tâm đáy thung lũng vươn ra vùng sáng vàng ở góc phải ($x = 0.5$).

---

## PHẦN 3: BẢNG TỔNG KẾT SO SÁNH 3 BÀI LAB (DÙNG KHI KẾT LUẬN & HỎI ĐÁP)

| Tiêu chí | Lab 1: Ball & Beam | Lab 2: 2-DOF Robotic Arm | Lab 3: Mountain Car |
| :--- | :--- | :--- | :--- |
| **Bài toán vật lý** | Cân bằng bóng trên thanh bập bênh | Điều khiển cánh tay robot 2 khớp gắp vật | Xe công suất yếu lấy đà vượt dốc |
| **Phương pháp RL** | **Q-Learning** (Off-policy TD) | **Expected SARSA** | **SARSA** (On-policy TD) |
| **Bài giảng trong `slides/`** | `2.8 Off-policy TD Control Q-learning` | `2.9 Expected Sarsa` | `2.7 TD for Control` & `3.6`, `3.7` |
| **TD Target $Y_t$** | $R + \gamma \max_{a'} Q(S', a')$ | $R + \gamma \sum_{a'} \pi(a'|S') Q(S', a')$ | $R + \gamma Q(S', A')$ |
| **Biểu diễn trạng thái** | $(x + 1.6v,\; \theta) \to 11 \times 11$ bins | Sai số cực $(\Delta\phi,\; \Delta r) \to 11 \times 11$ bins | $(x,\; v) \to 16 \times 16$ (Web) / $20 \times 20$ (Py) |
| **Cơ chế khám phá** | $\varepsilon$-greedy giảm dần ($0.25 \to 0.02$) | $\varepsilon$-greedy giảm dần ($0.25 \to 0.02$) | $\varepsilon$-greedy + **Optimistic Init ($Q_0 = 0$)** |
| **Kết quả sau huấn luyện** | Giữ bóng 400/400 bước (100%), đứng êm | Gắp trúng 100% mọi vị trí sau ~35 bước | Lên đỉnh 100% sau ~150–165 bước |
