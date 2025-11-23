

## 1. 依赖环境 (Prerequisites)

本项目依赖以下开源库，请确保已正确安装：

### 基础工具
*   **C++17** 编译器 (GCC 7+ / Clang 6+)
*   **CMake** 3.10+

### 第三方库
1.  **PCL (Point Cloud Library)** >= 1.8
    *   用于 3D 点云渲染和可视化。
    *   `sudo apt install libpcl-dev`
2.  **OpenCV** >= 3.4
    *   用于图像处理（即使不加载图片也需要链接）。
    *   `sudo apt install libopencv-dev`
3.  **Eigen3**
    *   用于矩阵运算和卡尔曼滤波。
    *   `sudo apt install libeigen-dev`
4.  **XGBoost** (C API)
    *   用于加载模型进行目标推理。
    *   需编译安装或通过包管理器安装 `libxgboost-dev`。
5.  **nlohmann_json**
    *   现代 C++ JSON 解析库。
    *   `sudo apt install nlohmann-json3-dev`

---

## 2. 编译构建 (Build)

```bash
mkdir build
cd build
cmake ..
make -j4
```

编译成功后，会在 `build` 目录下生成可执行文件 `radar_viewer`。

---

## 3. 配置文件 (config.json)

项目运行强依赖于 JSON 配置文件。以下是关键参数说明：

### 路径配置
*   `radar_csv`: 雷达数据 CSV 文件路径（**必须**）。
*   `models_dir`: 存放 XGBoost 模型 (`xgboost_model.json`, `scaler.json`, `RFE.json`) 的目录（**必须**）。
*   `lidar_dir`: 激光雷达 `.bin` 文件目录（可选，无则设为 `""`）。
*   `cam_dir`: 相机图片目录（可选，无则设为 `""`）。
*   `radar_extrinsic` / `lidar_extrinsic`: 外参 JSON 文件（可选，无则设为 `""`）。

### 检测参数
*   `prob_thres`: (0.0~1.0) 判定为障碍物的概率阈值。建议 `0.5` - `0.6`。
*   `max_misses`: (Int) 目标丢失多少帧后才删除轨迹。建议 `6`。

### 🛡️ 软车速估算 (Soft Ego Velocity) - **核心**
当没有 CAN 总线车速时，开启此功能可大幅减少误报。

*   `enable_soft_ego_est`: `true` 开启，`false` 关闭。
*   `static_speed_thres`: (m/s) 绝对速度小于此值的物体被视为背景。**建议 `0.6`** 以保护慢速行人。
*   `ego_rcs_min`: 参与车速估算的背景点最小强度。**建议 `-10.0`** (包含弱背景)。
*   `ego_min_candidates`: 最少需要多少个点才启动估算。**建议 `2`**。

---

## 4. 运行方法 (Usage)

### 基础运行
```bash
./radar_viewer --config ../config/config.json
```

### 键盘快捷键 (运行时控制)
*   `Space`: 暂停 / 继续播放。
*   `Right` / `d`: 下一帧。
*   `Left` / `a`: 上一帧。
*   `e`: 开启 / 关闭 CSV 结果导出。
*   `s`: 截图 (保存为 `viewer_capture.png`)。
*   `.` (句号): 增加 `prob_thres` 阈值 (+0.02)。
*   `,` (逗号): 减小 `prob_thres` 阈值 (-0.02)。
*   `Esc`: 退出程序。

---

## 5. 可视化颜色说明

| 颜色 | 状态 | 含义 |
| :--- | :--- | :--- |
| 🔴 **红色** | **Detected / Latched** | **危险**。XGBoost 判定为障碍物，或处于锁定保持期。 |
| 🟢 **绿色** | **Undetected / Filtered** | **安全**。XGBoost 判定为非障碍物，或被软车速逻辑判定为静止背景。 |
| ⚪ **白色** | Lidar Point Cloud | 激光雷达点云（仅作参考）。 |
| ⬜ **灰色框** | ROI Zone | 检测区域边界。 |

*注：如果在 HUD 信息中看到 `SoftEgo: VALID`，说明软车速估算正在工作，原本红色的静止护栏会自动变绿。*

---

## 6. 常见问题 (FAQ)

**Q1: 为什么画面中只有绿点，没有红点？**
*   检查 `prob_thres` 是否设得太高。
*   检查 `static_speed_thres` 是否设得太高（导致慢速目标被过滤）。尝试设为 `0.5`。

**Q2: 为什么满屏都是红点（误报严重）？**
*   这通常是因为**软车速估算未生效**。
*   检查 `enable_soft_ego_est` 是否为 `true`。
*   检查 `ego_rcs_min` 是否够低（建议 `-10.0`），否则找不到背景点无法启动估算。

**Q3: 报错 `Cannot open extrinsic` 或 `image`？**
*   这是正常的警告。如果没有外参文件或图片，直接在 JSON 里把对应路径设为 `""` 空字符串即可消除警告。

---

## 7. 目录结构示例

```
project_root/
├── build/
│   └── radar_viewer       # 编译出的可执行文件
├── config/
│   └── config.json        # 配置文件
├── data/
│   ├── radar.csv          # 数据源
│   └── lidar_frames/      # (可选)
├── models/
│   ├── xgboost_model.json # 模型文件
│   ├── scaler.json
│   └── RFE.json
├── main.cpp
└── CMakeLists.txt
```
