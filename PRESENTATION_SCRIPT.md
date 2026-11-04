# KỊCH BẢN THUYẾT TRÌNH & DEMO 3 BÀI LAB REINFORCEMENT LEARNING (REL301m)

Tài liệu này bao gồm **lời dẫn thuyết trình**, **kịch bản bấm nút Demo trực tiếp trên cửa sổ GUI**, **giải thích công thức & code ngắn gọn**, và **bộ câu hỏi Q&A phản biện của giảng viên** cho cả 3 bài Lab.

---

## PHẦN MỞ ĐẦU CHUNG (30 giây)

> **Lời nói:**
> "Chào thầy và các bạn. Hôm nay em xin trình bày 3 bài Lab thực hành môn Reinforcement Learning. Thay vì làm các bài toán trên lưới tĩnh (GridWorld), em xây dựng **3 mô phỏng điều khiển hệ thống vật lý và cơ điện tử (Physics & Robotics Control)** với không gian trạng thái liên tục, sử dụng phương pháp **Rời rạc hóa trạng thái (State Discretization / Binning)** kết hợp với 3 thuật toán **Temporal-Difference (TD) Control** cốt lõi của môn học:
> 1. **Lab 1:** Hệ thống Thanh Bập Bênh Cân Bằng Bóng (**Ball & Beam**) sử dụng **Q-Learning** *(Off-policy TD Control)*.
> 2. **Lab 2:** Cánh Tay Robot 2 Khớp Vươn Gắp Mục Tiêu (**2-DOF Robotic Arm Reacher**) sử dụng **Expected SARSA**.
> 3. **Lab 3:** Xe Lấy Đà Vượt Dốc (**Mountain Car**) sử dụng **SARSA** *(On-policy TD Control)*."

---

## BÀI LAB 1: HỆ THỐNG THANH BẬP BÊNH CÂN BẰNG BÓNG (BALL & BEAM – Q-LEARNING)

### 1. Chạy ứng dụng mở màn hình Demo
```powershell
python "lab/Lab1_Ball_And_Beam_QLearning/ball_and_beam_qlearning.py"
```

### 2. Kịch bản nói & thao tác trên cửa sổ GUI

* **Bước 1 – Giới thiệu bài toán & Trạng thái chưa học (Untrained):**
  * *(Chỉ vào màn hình khi mới bật lên, bóng đang lăn và bị rơi)*:
  * **Lời nói:** "Ở bài Lab 1, chúng ta có một thanh bập bênh dài $2\text{ m}$ có trục xoay ở chính giữa, bên trên là một quả bóng thép chịu gia tốc trọng trường $a = \frac{5}{7}g\sin(\theta)$.
    - **State (Trạng thái):** Gồm vị trí bóng $x \in [-1, 1]\text{ m}$, vận tốc lăn $v$, và góc nghiêng hiện tại của thanh $\theta \in [-15^\circ, +15^\circ]$. Để giảm số chiều bảng $Q$, em tổng hợp trạng thái động lực của bóng thành biến kết hợp $s_{\text{ball}} = x + 1.55v$ (vừa phản ánh độ lệch tâm, vừa dự đoán quán tính lăn) cùng với góc nghiêng $\theta$, chia thành lưới $11 \times 11$ ô.
    - **Action (Hành động):** Có 3 hành động rời rạc: `0: Nghiêng trái`, `1: Giữ nguyên`, `2: Nghiêng phải`.
    - **Reward (Phần thưởng):** Phạt nặng `-50` nếu bóng rơi khỏi mép thanh ($|x| > 1.0\text{ m}$); ở mỗi bước trên thanh, thưởng điểm dương khi bóng tiến gần tâm $x=0$, vận tốc $v \approx 0$ và góc nghiêng thanh phù hợp hướng hãm phanh.
    - Như thầy thấy trên màn hình ở chế độ `Live Train` ban đầu, khi bảng $Q(s,a)$ còn bằng 0, thanh nghiêng ngẫu nhiên làm quả bóng lăn tuột khỏi mép và rơi xuống vực chỉ sau vài chục bước."

* **Bước 2 – Bấm huấn luyện nhanh & Biểu diễn kết quả:**
  * *(Bấm nút **`⚡ Huấn Luyện Nhanh +350 Episodes`**)*:
  * **Lời nói:** "Bây giờ em bấm huấn luyện nhanh 350 episodes bằng thuật toán **Q-Learning** (Off-policy TD Control), cập nhật theo công thức:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
    Sau 350 episodes, thầy có thể thấy ở biểu đồ góc phải dưới, đường cong học đã vọt lên mức tối đa **400/400 bước**. Trên màn hình mô phỏng, dù quả bóng xuất phát ở sát mép trái hay mép phải, thanh bập bênh tự động nghiêng ngược chiều để **hãm phanh quán tính lăn của bóng**, dẫn bóng lăn từ từ về đúng tâm trục xoay $x = 0.0\text{ m}$ rồi tự cân bằng ngang để giữ bóng đứng yên màu xanh lá."

* **Bước 3 – Tương tác gây nhiễu (Điểm nhấn ăn điểm):**
  * *(Đợi bóng đang nằm im màu xanh ở giữa thanh, bấm nút **`💨 Hất Bóng Lăn Về Mép (Test Phản Xạ)`** 2–3 lần)*:
  * **Lời nói:** "Để chứng minh Agent thực sự hiểu quy luật vật lý chứ không học vẹt, em bấm nút **Hất bóng lăn về mép** để tác động một vận tốc ngoại lực lớn đẩy văng quả bóng. Ngay lập tức, thanh bập bênh nghiêng đón đỡ, phanh tốc độ bóng trước khi chạm mép vực và đưa bóng quay về tâm cân bằng."

---

## BÀI LAB 2: CÁNH TAY ROBOT 2 KHỚP GẮP MỤC TIÊU (2-DOF ROBOTIC ARM – EXPECTED SARSA)

### 1. Chạy ứng dụng mở màn hình Demo
```powershell
python "lab/Lab2_Robotic_Arm_Reacher_ExpectedSarsa/robotic_arm_sarsa.py"
```

### 2. Kịch bản nói & thao tác trên cửa sổ GUI

* **Bước 1 – Giới thiệu bài toán & Động học Robot:**
  * *(Chỉ vào cánh tay 2 đoạn xanh – vàng và quả cầu mục tiêu màu xanh lá)*:
  * **Lời nói:** "Ở bài Lab 2, em giải bài toán điều khiển động học cánh tay Robot phẳng 2 bậc tự do (2-DOF Planar Manipulator):
    - Cánh tay gồm **Khớp vai ($\theta_1$)** dài $L_1 = 0.55\text{ m}$ và **Khớp khuỷu ($\theta_2$)** dài $L_2 = 0.45\text{ m}$. Vị trí đầu gắp (Gripper) được tính qua phương trình động học thuận (Forward Kinematics):
      $$x_e = L_1\cos\theta_1 + L_2\cos(\theta_1 + \theta_2), \quad y_e = L_1\sin\theta_1 + L_2\sin(\theta_1 + \theta_2)$$
    - **State:** Thay vì lưu tọa độ tuyệt đối khiến không gian trạng thái bị nổ tổ hợp, em biểu diễn trạng thái theo **sai số tọa độ cực tương đối** giữa đầu gắp và mục tiêu: $(\Delta \phi, \Delta r)$, trong đó $\Delta \phi$ là góc lệch hướng và $\Delta r$ là độ lệch bán kính vươn, chia thành lưới $11 \times 11$ trạng thái.
    - **Action:** Có 5 lệnh điều khiển mô-tơ khớp: `Xoay vai trái/phải`, `Duỗi/Gập khớp khuỷu`, và `Giữ nguyên`.
    - **Reward:** Thưởng theo độ giảm khoảng cách tiến về mục tiêu $25 \cdot (d_{t-1} - d_t) - 0.15$, và thưởng lớn `+50` khi đầu gắp chạm trúng tâm quả cầu mục tiêu ($d \le 0.065\text{ m}$)."

* **Bước 2 – Bấm huấn luyện bằng Expected SARSA:**
  * *(Bấm nút **`⚡ Huấn Luyện Nhanh +300 Episodes`**)*:
  * **Lời nói:** "Ở bài này em sử dụng thuật toán **Expected SARSA**. Khác với Q-Learning dùng $\max_{a'} Q(s', a')$ dễ bị phương sai cao hay SARSA lấy mẫu đơn lẻ, Expected SARSA tính **kỳ vọng toán học** trên toàn bộ phân phối xác suất $\pi(a'|s')$:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \sum_{a'} \pi(a'|s') Q(s', a') - Q(s, a) \right]$$
    Sau khi huấn luyện 300 episodes, số bước trung bình để gắp trúng mục tiêu giảm xuống chỉ còn khoảng **30–35 bước**. Cánh tay phối hợp nhịp nhàng vừa xoay vai vừa gập/duỗi khuỷu để đi thẳng tới mục tiêu."

* **Bước 3 – Click chuột trực tiếp trên màn hình (Điểm nhấn ăn điểm):**
  * *(Dùng chuột **Click trực tiếp vào 4–5 vị trí khác nhau** quanh vòng tròn trên màn hình mô phỏng)*:
  * **Lời nói:** "Em có thiết kế tính năng tương tác trực tiếp: khi em **click chuột vào bất kỳ tọa độ nào** trong vùng làm việc $360^\circ$, quả cầu mục tiêu sẽ nhảy tới vị trí chuột vừa click và cánh tay Robot lập tức điều khiển 2 khớp vươn tới gắp trúng đích 100%."

---

## BÀI LAB 3: XE LẤY ĐÀ VƯỢT DỐC (MOUNTAIN CAR – SARSA)

### 1. Chạy ứng dụng mở màn hình Demo
```powershell
python "lab/Lab3_Mountain_Car_SARSA/mountain_car_sarsa.py"
```

### 2. Kịch bản nói & thao tác trên cửa sổ GUI

* **Bước 1 – Giới thiệu bài toán & Tại sao xe bị kẹt dưới dốc:**
  * *(Chỉ vào chiếc xe đang chạy qua lại dưới đáy thung lũng lúc mới mở)*:
  * **Lời nói:** "Ở bài Lab 3 là bài toán kinh điển **Mountain Car** (Example 10.1 trong sách Sutton & Barto):
    - Chiếc xe bắt đầu ở đáy thung lũng ($x \approx -0.5$) và cần leo lên lá cờ đích trên đỉnh đồi phải ($x = 0.5$).
    - Điểm đặc biệt của hệ vật lý này là lực đẩy của động cơ ($0.001$) nhỏ hơn thành phần trọng lực dốc ($0.0025\cos(3x)$). Do đó nếu xe chỉ cố nhấn ga chạy sang phải, xe sẽ bị tụt dốc. Xe buộc phải học chiến lược **đi ngược chiều mục tiêu**: lùi lên sườn đồi bên trái để tích lũy thế năng rồi mới phóng vọt qua đỉnh đồi bên phải.
    - **State:** Vị trí $x \in [-1.2, 0.5]$ và vận tốc $v \in [-0.07, 0.07]$ được rời rạc hóa thành lưới $20 \times 20$ ô.
    - **Action:** `0: Đẩy trái (-1)`, `1: Thả trôi (0)`, `2: Đẩy phải (+1)`.
    - **Reward:** Phạt `-1` ở mỗi bước thời gian để ép Agent tìm cách chạm đỉnh đồi với ít bước nhất. Đồng thời bảng $Q$ được khởi tạo bằng $0$ (**Optimistic Initial Values** – cao hơn mức phạt `-1`), giúp thúc đẩy Agent tự động khám phá các vùng trạng thái mới."

* **Bước 2 – Bấm huấn luyện SARSA & Quan sát xe lấy đà:**
  * *(Bấm nút **`⚡ Huấn Luyện Nhanh +300 Episodes`**, có thể bấm 2 lần lên 600 episodes để đường đi cực mượt)*:
  * **Lời nói:** "Em huấn luyện Agent bằng thuật toán **SARSA (On-policy TD Control)** cập nhật theo bộ 5 phần tử $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$:
    $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
    Như thầy thấy trên màn hình ở chế độ `Greedy`, sau khi học xong, chiếc xe chủ động bật động cơ **Đẩy Trái (`◀`)** leo ngược lên đồi trái để lấy đà, sau đó đảo chiều sang **Đẩy Phải (`▶`)** lao một mạch lên chạm lá cờ đích chỉ trong khoảng **150–190 bước** (so với 500 bước bị kẹt lúc chưa học)."

---

## PHẦN Q&A: CÁC CÂU HỎI GIẢNG VIÊN HAY HỎI & CÁCH TRẢ LỜI

### Câu 1: Tại sao cả 3 bài toán đều có trạng thái là số thực liên tục (Continuous State) mà em vẫn dùng bảng Q-Table (Tabular TD) được?
* **Trả lời:**
  "Dạ em áp dụng kỹ thuật **State Discretization (State Aggregation / Binning)** mà môn học giới thiệu ở Module 3 (tương tự bài Lab 3.4 State Aggregation). Cụ thể em dùng hàm `np.digitize()` để chia không gian liên tục thành các khoảng lưới rời rạc (bins). Cách này giúp giữ thuật toán cực kỳ nhẹ, chạy huấn luyện vài trăm episodes chỉ mất chưa tới 0.3 giây bằng `numpy` thuần mà không cần GPU hay mạng Neural phức tạp."

### Câu 2: Sự khác nhau cốt lõi giữa 3 thuật toán Q-Learning (Lab 1), Expected SARSA (Lab 2) và SARSA (Lab 3) trong bài của em là gì?
* **Trả lời:**
  * **SARSA (Lab 3):** Là **On-policy** — cập nhật giá trị dựa trên chính hành động thực tế tiếp theo $A_{t+1}$ được chọn từ chính sách $\epsilon$-greedy: $R + \gamma Q(S', A')$.
  * **Q-Learning (Lab 1):** Là **Off-policy** — cập nhật trực tiếp theo hành động tham lam lớn nhất $\max_{a'} Q(S', a')$ bất kể bước tiếp theo Agent có khám phá ngẫu nhiên hay không.
  * **Expected SARSA (Lab 2):** Tính giá trị trung bình kỳ vọng theo xác suất của chính sách hiện tại $\sum_{a'} \pi(a'|S') Q(S', a')$, giúp loại bỏ nhiễu ngẫu nhiên khi chọn $A'$ của SARSA và ổn định hơn khi có nhiều hành động (như 5 hành động của cánh tay Robot).

### Câu 3: Ở bài Lab 3 (Mountain Car), tại sao khởi tạo bảng `Q = np.zeros(...)` lại gọi là Optimistic Initialization (Khởi tạo lạc quan)?
* **Trả lời:**
  "Dạ vì ở bài Mountain Car, mỗi bước thời gian chưa tới đích đều nhận phần thưởng âm `reward = -1`. Do đó giá trị thực tế $Q^*(s, a)$ luôn là số âm (ví dụ $-150$). Khi khởi tạo $Q_0(s,a) = 0$, giá trị $0$ lớn hơn rất nhiều so với giá trị thực, nên những ô trạng thái chưa được thăm (vẫn có giá trị $\approx 0$) sẽ luôn hấp dẫn hơn các ô đã thăm (đã bị cập nhật giảm xuống âm), buộc chiếc xe tự động khám phá rộng khắp thung lũng để tìm ra đường lên đồi."

