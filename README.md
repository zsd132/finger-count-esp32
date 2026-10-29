# 手指计数：摄像头 + YOLO + ESP32 OLED

用电脑摄像头识别你伸出的手指数量（0~5），实时显示在屏幕上；可选通过 USB 串口把数字发给 **ESP32**，在 **0.96 寸 OLED** 上显示，并用 **GPIO17** 输出一个开关量。

![demo](docs/demo.jpg)

<p>
  <img alt="python" src="https://img.shields.io/badge/python-3.10--3.12-blue">
  <img alt="platform" src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey">
  <img alt="esp32" src="https://img.shields.io/badge/MCU-ESP32-red">
</p>

---

## 目录

- [0. 这个仓库能做什么](#0-这个仓库能做什么)
- [1. 准备清单](#1-准备清单)
- [2. 从零复现（macOS / Linux）](#2-从零复现macos--linux)
- [3. Windows 复现](#3-windows-复现)
- [4. 使用说明](#4-使用说明)
- [5. 接 ESP32 + OLED](#5-接-esp32--oled)
- [6. 常见问题](#6-常见问题)
- [7. 项目结构](#7-项目结构)
- [8. 它是怎么识别的](#8-它是怎么识别的)
- [9. 换权重 / 自己训练](#9-换权重--自己训练)
- [10. 性能](#10-性能)
- [11. 文档与致谢](#11-文档与致谢)

---

## 0. 这个仓库能做什么

- **0~5 手指计数**：YOLO11n-pose 手部 21 关键点，或 MediaPipe 关键点；
- **两段式推理**：先检测手，再裁出手部放大跑第二遍，显著提升小目标精度；
- **判指**：几何规则（四指 + 拇指单独规则），MediaPipe 模式下还会用学习型分类器；
- **防抖**：9 帧多数投票 + 0.5 秒保持 + 丢失 0.6 秒保持，数字不乱跳；
- **屏幕 HUD**：骨架高亮、大数字、左右手标签、FPS；
- **串口**：`FC <总数> <手数> [数字[:L|R]]`，断线自动重连；
- **ESP32**：OLED 显示大数字 + L/R 徽标，GPIO17 在数字为 5 时输出高电平。

> 仓库**不包含**第三方 YOLO 权重和 HaGRID 数据（许可证/体积原因）。
> 第一次运行前执行 `python tools/download_models.py`，脚本会自动下载并校验。

---

## 1. 准备清单

### 硬件

| 必需 | 说明 |
|---|---|
| 电脑 | macOS / Linux / Windows |
| 摄像头 | 笔记本内置即可，或用 USB 摄像头 |

| 可选（显示到单片机） | 说明 |
|---|---|
| ESP32 开发板 | 经典 ESP32-WROOM-32 即可（本项目在 ESP32-D0WD-V3 上验证） |
| 0.96 寸 OLED | SSD1306，128×64，I2C，4 针，地址 0x3C |
| USB 数据线 | 能给 ESP32 供电并支持串口（不是纯充电线） |

### 软件

| 软件 | 版本 / 说明 |
|---|---|
| Python | **3.10 / 3.11 / 3.12**（3.13 可能没有 torch 2.2.2 轮子） |
| Git | 任意新版本 |
| PlatformIO | 只在烧 ESP32 固件时需要；`firmware/flash.sh` 会自动装 |

---

## 2. 从零复现（macOS / Linux）

> 下面命令默认在终端里执行。macOS 用 `python3`；Linux 如果 `python3` 指向 3.13，请显式用 `python3.12`。

### 2.1 克隆仓库

```bash
git clone https://github.com/zsd132/finger-count-esp32.git
cd finger-count-esp32
```

### 2.2 创建虚拟环境并安装依赖

```bash
python3 --version          # 确认是 3.10 ~ 3.12
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

依赖是固定版本的（Intel Mac / 通用平台兼容）：

```
torch 2.2.2 / torchvision 0.17.2
mediapipe 0.10.21
numpy 1.26.4
opencv-contrib-python 4.10.0.84
ultralytics 8.3.40
pyserial 3.5
```

> Apple Silicon 也能用这套版本；如果 `mediapipe` 装不上，先确认 Python 是 3.11 或 3.12。

### 2.3 下载手部关键点权重

```bash
python tools/download_models.py
```

预期输出：

```
model OK: .../models/yolo11n-pose-hands.pt (6.3 MB)
mediapipe hand model OK
```

脚本会走 GitHub API 下载并校验 SHA-256，不依赖被墙的 `github.com` 网页。
如果仍然失败，手动下载
[`best.pt`](https://github.com/chrismuntean/YOLO11n-pose-hands/blob/main/runs/pose/train/weights/best.pt)
保存为 `models/yolo11n-pose-hands.pt`。

### 2.4 运行

```bash
python main.py
```

对着镜头比 0~5，窗口左上角显示数字。

- 首次运行 macOS 会请求摄像头权限，**必须允许**；
- 拒绝过就去 `系统设置 → 隐私与安全性 → 摄像头` 勾选 Terminal / VS Code，然后完全退出重开。

### 2.5 验证安装（不接摄像头也能测）

```bash
# 19 个单元测试
python -m unittest discover -s tests -v

# 跑一张示例图，期望输出 counts=[5] total=5
python main.py --source samples/px_diva-plavalaguna-6937816.jpg --headless
```

成功后会在 `out/` 生成标注图。

### 2.6 一键脚本（可选）

macOS / Linux 可以直接：

```bash
chmod +x run.sh
./run.sh
```

`run.sh` 会自动完成：建 venv → 装依赖 → 下模型 → 启动。

macOS 也可以双击 `启动.command`（窗口不会自动关，日志在 `out/launcher.log`）。

---

## 3. Windows 复现

`run.sh` / `flash.sh` 是 bash 脚本；Windows 用下面的命令即可（PowerShell）：

```powershell
git clone https://github.com/zsd132/finger-count-esp32.git
cd finger-count-esp32

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -r requirements.txt
python tools\download_models.py
python main.py
```

- 摄像头一般是 `--source 0`，多摄像头试 `--source 1`；
- 串口是 `COM3` / `COM4` 这种，用 `python main.py --list-serial` 查；
- 没有 `run.sh`、`启动.command`，用上面的命令；
- 固件烧录建议装 VS Code 的 **PlatformIO IDE** 扩展（见 5.2）。

---

## 4. 使用说明

### 4.1 常用命令

```bash
# 摄像头 + 屏幕
python main.py

# 摄像头 + 串口 + 左右手图标（推荐）
python main.py --serial auto --backend mediapipe

# 只用 YOLO
python main.py --serial auto --backend yolo

# 跑图片 / 视频
python main.py --source samples/px_diva-plavalaguna-6937816.jpg --headless
python main.py --source video.mp4 --headless --record out/demo.mp4

# 列出串口
python main.py --list-serial
```

### 4.2 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--source` | `0` | 摄像头编号 / 视频 / 图片 |
| `--backend` | `auto` | `auto` = YOLO 为主，某帧没检到手时自动用 MediaPipe 补；也可强制 `yolo` / `mediapipe` |
| `--model` | `models/yolo11n-pose-hands.pt` | 换权重改这里 |
| `--imgsz` | `640` | YOLO 第一段检测尺寸 |
| `--conf` | `0.35` | 手部检测阈值，手远/暗可降到 `0.2` |
| `--max-hands` | `2` | 最多几只手 |
| `--hold` | `0.5` | 新数字稳定多少秒才显示/发送（0 关闭） |
| `--no-refine` | 关 | 关闭第二段裁剪放大（快一倍，精度下降） |
| `--no-mirror` | 关 | 不镜像 |
| `--serial` | 空 | 串口路径，或 `auto` |
| `--serial-baud` | `115200` | 要和单片机一致 |
| `--serial-hz` | `30` | 串口最大发送频率 |
| `--headless` | 关 | 不弹窗（跑文件/服务器） |
| `--record` | 空 | 录制标注后的视频 |
| `--dump-dir` | 空 | 每隔 N 帧存调试图 |
| `--list-serial` | - | 列出串口后退出 |

### 4.3 按键

| 键 | 作用 |
|---|---|
| `q` / `Esc` | 退出 |
| `s` | 截图到 `out/snapshot_*.jpg` |
| `d` | 保存调试图 + 关键点 JSON（数字不对时用） |
| `m` | 切换镜像 |

### 4.4 两个引擎怎么选

| | YOLO | MediaPipe |
|---|---|---|
| 速度（Intel CPU） | ~5 FPS（两段式） | ~20–30 FPS |
| 左右手 | ❌ 不输出 | ✅ L/R |
| 判指 | 几何规则 | 几何规则 + 学习型分类器 |
| 适用 | 想用 YOLO / 自定义权重 | 日常使用、要 L/R 图标 |

**建议日常用 `--backend mediapipe`**；想研究 YOLO 或换自己的权重用 `--backend yolo`。

---

## 5. 接 ESP32 + OLED

### 5.1 接线

| OLED | ESP32（DevKit / WROOM-32） |
|---|---|
| VCC | **3V3** |
| GND | **GND** |
| SCL | **GPIO22** |
| SDA | **GPIO21** |

> 看模块丝印，不要按排列顺序猜。ESP32-C3/S3 默认 I2C 不同（常见 SDA=8、SCL=9），
> 需要同时改 `firmware/src/main.cpp` 顶部的 `I2C_SDA` / `I2C_SCL` 和
> `platformio.ini` 里的 `board`。

### 5.2 烧录固件

**方式 A：一条命令（macOS / Linux）**

```bash
cd firmware
chmod +x flash.sh
./flash.sh
```

脚本会：

1. 自动找 USB 串口（也可以 `./flash.sh /dev/cu.usbserial-1420` 指定）；
2. 检查 PlatformIO，没有就装到项目 `.venv`；
3. 第一次会下载 ESP32 工具链（几百 MB，几分钟）；
4. 以 115200 波特率烧录。

> 烧录失败时：关掉 Arduino IDE / 串口监视器 / 正在运行的本项目；按住 BOOT 再点上传；
> 换一根能传数据的 USB 线。

**方式 B：VS Code 的 PlatformIO 扩展**

1. VS Code 扩展商店搜 **PlatformIO IDE** 安装；
2. 打开 `firmware/` 文件夹；
3. 点底部 **Upload**（→ 图标）；
4. 需要看日志点 **Serial Monitor**（115200）。

**Windows**：用方式 B；或者 `pip install platformio` 后：

```powershell
cd firmware
pio run -t upload --upload-port COM3
```

### 5.3 运行

```bash
cd ..            # 回到项目根目录
python main.py --serial auto --backend mediapipe
```

- 屏幕左下角出现 `SER` 表示串口已连接；
- OLED 显示大数字，右上角显示 L / R / ?；
- 电脑端 3 秒没发数据，OLED 左上角变 `NO LINK`。

串口设备名：

| 系统 | 名称 |
|---|---|
| macOS | `/dev/cu.usbserial-*`、`/dev/cu.usbmodem*` |
| Linux | `/dev/ttyUSB*`、`/dev/ttyACM*` |
| Windows | `COM3`、`COM4` … |

用 `--serial auto` 就不用手改；换 USB 口后程序也会自动重新查找。

### 5.4 串口协议

ASCII、`115200 8N1`、一行一帧、`\n` 结尾：

```
FC <总数> <手数> [<手指数>[:L|R] ...]
```

| 场景 | 发送内容 |
|---|---|
| 连接成功 | `FC READY` |
| 没检测到手 | `FC 0 0` |
| 右手 3 根 | `FC 3 1 3:R` |
| 两只手 2+3 根 | `FC 5 2 2:L 3:R` |
| 左右手未知（YOLO） | `FC 4 1 4` |

- 只在数字变化时发送，另外每秒补一帧心跳；
- 最大 30 帧/秒；
- 断线每 2 秒重连。

### 5.5 GPIO17 输出

| 手指数量 | GPIO17 |
|---|---|
| **5** | **高电平 3.3V** |
| 0~4 | 低电平 0V |
| 上电/复位 | 低电平 |
| 串口断线 | 保持最后状态 |

接线示例：

```
LED：  GPIO17 ── 330Ω ── LED ── GND
继电器：GPIO17 ── 模块 IN；模块 VCC/GND 另接；共地
```

> GPIO 是 3.3V、约 12mA，**不能直接驱动继电器线圈/电机**，要经三极管/光耦/模块。
> 驱动继电器、电磁锁、电机、舵机、220V 负载的完整方法见
> [`docs/完整技术文档.md`](docs/完整技术文档.md) 第 12 章。

### 5.6 OLED 布局

```
┌────────────────────────┐
│ FINGERS           [L]  │
│          5             │
└────────────────────────┘
```

---

## 6. 常见问题

### Q1. `pip install` 卡住 / 装不上 torch 或 mediapipe

- 确认 Python 是 **3.10~3.12**（`python3 --version`）；3.13 换 3.12；
- 国内网络慢可以先换源：
  `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
- Apple Silicon 装不上 mediapipe 时，优先用 Python 3.11。

### Q2. 模型下载失败

`python tools/download_models.py` 走 `api.github.com`；如果失败：

1. 浏览器打开
   <https://github.com/chrismuntean/YOLO11n-pose-hands/blob/main/runs/pose/train/weights/best.pt>
2. 下载后保存为 `models/yolo11n-pose-hands.pt`；
3. 重新运行 `python main.py`。

### Q3. MediaPipe 首次运行报错 / 下载模型失败

MediaPipe 需要访问 `storage.googleapis.com`。如果被墙：

```bash
python main.py --backend yolo
```

YOLO 模式不依赖 MediaPipe（只是没有 L/R 图标）。

### Q4. 摄像头打不开 / `not authorized`

- macOS：`系统设置 → 隐私与安全性 → 摄像头`，勾选 Terminal / VS Code / PyCharm，然后**完全退出重开**；
- 确认没有被 Zoom / 微信 / OBS 占用；
- 多摄像头试 `--source 1`。

### Q5. 串口打不开 / `Resource busy`

- 关掉 Arduino IDE 的串口监视器、其他串口工具（串口独占）；
- macOS 用 `/dev/cu.*`，不要 `/dev/tty.*`；
- `python main.py --list-serial` 确认名字；
- 换 USB 口后名字可能变，用 `--serial auto`。

### Q6. OLED 不亮 / 花屏

- 检查 3V3、GND、SCL=22、SDA=21；
- 少数模块地址是 0x3D：在 `firmware/src/main.cpp` 里
  `display.setI2CAddress(0x3D << 1);`
- 花屏/偏移可能是 SH1106 屏：把构造函数换成
  `U8G2_SH1106_128X64_NONAME_F_HW_I2C`；
- 杜邦线长时把 `Wire.setClock(400000L)` 改成 `100000L`。

### Q7. OLED 显示 `?` 而不是 L/R

YOLO 权重不输出左右手。用 `--backend mediapipe`。

### Q8. 数字变化慢 / 一直不变

- 这是 `--hold 0.5` 的防抖设计，改 `--hold 0.2` 或 `0`；
- 数字一直 0：手离镜头太远/太暗，或 `--conf` 太高（试 `0.2`）；
- 按 `d` 保存调试图，看 `out/debug_*.json` 里的角度和概率。

### Q9. 固件上传失败

- 关掉占用串口的程序；
- 按 BOOT 再点上传；
- 上传速度已设为 115200（`platformio.ini`），不要改回 921600；
- 换 USB 线/口。

### Q10. 仓库里为什么没有 `models/*.pt` 和 `data/basic_export.csv`

- 权重是第三方模型、原仓库没有明确许可证，公开仓库不直接分发；`download_models.py` 会下载；
- `basic_export.csv` 是 HaGRID 派生数据，只在重训判指分类器时需要，`data/README.md` 有下载地址。

---

## 7. 项目结构

```
finger-count-esp32/
├── main.py                 # 入口
├── run.sh                  # 一键：建 venv / 装依赖 / 下模型 / 启动
├── 启动.command             # macOS 双击启动
├── requirements.txt
│
├── src/
│   ├── app.py              # 主循环、参数、摄像头/文件/串口
│   ├── backends.py         # YOLO / MediaPipe / auto 回退 + 两段式精修
│   ├── counting.py         # 21 关键点 → 每指伸/屈
│   ├── features.py         # 每指 10 维归一化特征
│   ├── finger_model.py     # 学习型 5 指分类器
│   ├── tracking.py         # 多手跟踪 + 多数投票 + 0.5s 保持
│   ├── serial_link.py      # 串口协议、心跳、重连
│   └── draw.py             # 骨架 / 数字面板 / 状态条
│
├── models/
│   └── finger_lr.npz       # 判指分类器（YOLO 权重需下载）
│
├── firmware/               # ESP32 工程（PlatformIO）
│   ├── platformio.ini
│   ├── src/main.cpp        # 串口解析 + OLED + GPIO17
│   ├── flash.sh            # 一键编译烧录
│   └── README.md
│
├── arduino/                # 通用 Arduino 接收示例
├── tools/
│   ├── download_models.py
│   ├── eval_images.py
│   └── train_finger_model.py
├── data/README.md          # 训练数据下载说明
├── tests/                  # 19 个单元测试
├── samples/                # 示例图片
├── docs/
│   ├── demo.jpg
│   └── 完整技术文档.md
└── out/                    # 运行时输出（git 忽略）
```

---

## 8. 它是怎么识别的

1. **读帧 + 镜像**：摄像头 1280×720，水平翻转；
2. **检测 + 21 关键点**：YOLO11n-pose（两段式：整帧检测 → 裁手放大再推理）或 MediaPipe；
3. **判指**：
   - 四指：PIP/DIP 关节角度 ≥135°，且指尖沿掌轴比 PIP 更远；
   - 拇指：MCP/IP 角度 ≥125°，且指尖离食指根 ≥1.0 掌宽或横向伸出 ≥0.9 掌宽；
   - MediaPipe 模式还可用 5 个逻辑回归分类器（HaGRID 数据训练，计数 86.9%、拇指 92.3%）；
4. **防抖**：9 帧多数投票 → 0.5 秒保持 → 丢失 0.6 秒保持；
5. **输出**：屏幕 HUD + 串口 `FC ...` → ESP32 → OLED + GPIO17。

完整公式、阈值、原理、协议细节见
👉 **[docs/完整技术文档.md](docs/完整技术文档.md)**。

---

## 9. 换权重 / 自己训练

### 换权重

```bash
python main.py --model models/my_hand_pose.pt
```

新权重必须：`task=pose`、`kpt_shape=[21,3]`、21 个关键点顺序与 MediaPipe 一致。
检查：

```bash
python - <<'PY'
from ultralytics import YOLO
m = YOLO("models/my_hand_pose.pt")
print(m.task, m.names, m.model.kpt_shape)
PY
```

### 用自己的数据训练

- **路线 A：微调 YOLO 关键点模型**（采集 → 标 21 点 → YOLO pose 格式 → 训练 → 替换）；
- **路线 B：只训练判指分类器**（只要 0~5 类别标签，几分钟搞定，推荐先做）。

两条路线的完整步骤、标注格式、`data.yaml`、训练命令、常见坑：
👉 **技术文档第 7、8 章**。

---

## 10. 性能

Intel Mac、纯 CPU 实测：

| 模式 | 单帧 | FPS |
|---|---:|---:|
| YOLO 两段式 | ~200 ms | ~5 |
| YOLO `--no-refine` | ~95 ms | ~10 |
| MediaPipe | ~30–50 ms | ~20–30 |

体感延迟主要来自 `--hold 0.5` 的防抖，不是模型。

---

## 11. 文档与致谢

- 完整技术文档：[`docs/完整技术文档.md`](docs/完整技术文档.md)（1400+ 行：原理、协议、换权重、训练、GPIO 驱动外部设备）
- 手部关键点权重：[chrismuntean/YOLO11n-pose-hands](https://github.com/chrismuntean/YOLO11n-pose-hands)
  （训练数据为 *Hand Keypoint Dataset 26K*；仓库未声明许可证，**仅供学习/个人演示，商用前自行确认授权**）
- 判指分类器训练数据：[HaGRID 关键点导出](https://huggingface.co/datasets/GestureDetectionConnoisseurs/preprocessed_hand_skeletons)
- 示例图片：[cj-mills/pexel-hand-gesture-test-images](https://huggingface.co/datasets/cj-mills/pexel-hand-gesture-test-images)（Pexels）

本项目代码可自由修改使用。
