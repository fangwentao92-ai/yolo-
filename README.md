# YOLO 可视化工具

基于 Ultralytics YOLO11 的零代码识别工具：打开窗口点按钮，即可完成图片识别、视频识别和摄像头实时检测，支持**物体检测**与**骨架姿态**两种模式。同时提供 Gradio 网页版，可一键部署到云端。

## 功能特性

- 两种识别模式：物体检测（80 类目标画框）/ 骨架姿态（17 个关键点）
- 图片识别：弹窗选图，结果即时显示并保存
- 视频识别：逐帧推理，另存为带标注的新视频
- 摄像头实时检测：画面内嵌显示，支持 DirectShow 后端
- 置信度滑块：实时调整识别门槛
- 网页版：Gradio 界面，支持上传图片/视频与网页摄像头拍照，可部署到 Render

## 项目结构

```
yolo-gui/
├── yolo_gui.py          # 桌面版主程序（Tkinter）
├── webapp.py            # 网页版入口（Gradio，可部署 Render）
├── yolo11n.pt           # 物体检测权重
├── yolo11n-pose.pt      # 骨架姿态权重
├── download_weights.py  # 权重下载脚本（镜像优先，可选）
├── requirements.txt     # 网页版/部署依赖
├── 启动界面.bat          # Windows 桌面启动器
├── .python-version
├── LICENSE
└── README.md
```

## 快速开始

### 1. 环境准备

建议 Python 3.10 ~ 3.12。

PyTorch 按显卡选择安装：

```bash
# 普通 CUDA 显卡
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# RTX 50 系列（Blackwell，如 5060/5070/5080/5090）需要 CUDA 12.8+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 纯 CPU
pip install torch torchvision
```

### 2. 权重文件

仓库已包含两个权重（共约 12 MB），克隆即可用。如需重新下载：

```bash
python download_weights.py
```

脚本优先使用 hf-mirror 镜像，官方 GitHub release 作为备用。

### 3. 桌面版（Tkinter）

```bash
pip install ultralytics opencv-python pillow numpy
python yolo_gui.py
```

Windows 用户也可双击 `启动界面.bat`。

### 4. 网页版（Gradio）

```bash
pip install -r requirements.txt
python webapp.py
```

然后打开 `http://127.0.0.1:7860`。

## 桌面版功能说明

| 区域 | 作用 |
|---|---|
| 识别模式 | 物体检测 / 骨架姿态，单选切换 |
| 置信度滑块 | 拖动调整识别门槛，值越高越严格 |
| 选择图片识别 | 选图后结果显示在窗口并自动保存 |
| 选择视频识别 | 逐帧识别后另存为新视频 |
| 打开摄像头 | 实时画面内嵌显示，点"停止"结束 |
| 打开结果文件夹 | 查看所有识别结果 |
| 日志区 | 显示状态、目标数量和保存路径 |

结果保存于 `results/`（图片 `img_时间戳.jpg`，视频 `video_时间戳.mp4`）。

## 部署到 Render

1. 把本仓库推到 GitHub
2. Dashboard → **New + → Web Service** → 关联该仓库
3. 填写：
   - **Build Command**
     ```
     pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
     ```
   - **Start Command**：`python webapp.py`
   - **实例规格**：至少 2 GB 内存（512 MB 免费实例跑 torch 容易 OOM）
4. 部署完成后访问 `https://<服务名>.onrender.com`

可选环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_VIDEO_FRAMES` | 150 | 单次视频最多处理帧数 |
| `MAX_WIDTH` | 960 | 推理前最大宽度，降低 CPU 压力 |
| `PORT` | 7860 | 平台自动注入 |

注意：云端无 GPU，视频处理较慢；容器磁盘临时，重启后 `outputs/` 丢失；免费实例空闲 15 分钟后休眠。

## 常见问题

1. **摄像头打不开**：确认未被微信、QQ、会议软件占用；程序已使用 DirectShow 后端（默认 MSMF 在部分机器上抓帧失败）
2. **切换模式后第一次变慢**：正常，新模型需要加载
3. **视频处理不流畅**：正常，逐帧推理，进度看日志
4. **窗口画面不完整**：显示用缩放适配，保存结果保持原始分辨率

## License

[MIT](LICENSE)
