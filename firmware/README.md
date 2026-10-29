# ESP32 固件：手指计数 OLED 显示

电脑端（本项目根目录）负责识别，通过 USB 串口把结果发过来：

```
FC <总数> <手数> [<手指数>[:L|R] ...]
```

例如 `FC 3 1 3:R`、`FC 5 2 2:L 3:R`。右上角会画 L/R 小徽标；
没有左右手信息时（YOLO 引擎）发 `FC 3 1 3`，徽标显示 `?`。

## 接线

| OLED | ESP32（DevKit / WROOM-32） |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SCL | GPIO22 |
| SDA | GPIO21 |

> ESP32-C3：SDA=8、SCL=9；ESP32-S3：SDA=8、SCL=9。
> 同时把 `platformio.ini` 的 `board` 改成 `esp32-c3-devkitm-1` / `esp32-s3-devkitc-1`，
> 并修改 `src/main.cpp` 顶部的 `I2C_SDA` / `I2C_SCL`。

## 烧录（不需要 Arduino IDE）

### 方式 A：一条命令（推荐）

插好 ESP32，然后：

```bash
cd ~/Desktop/yolo/firmware
./flash.sh
```

脚本会自动找串口、编译、上传。第一次会下载 ESP32 工具链（几百 MB，几分钟）。

指定串口：

```bash
./flash.sh /dev/cu.usbserial-1420
```

### 方式 B：VS Code + PlatformIO 扩展

1. VS Code 扩展商店搜索 **PlatformIO IDE** 安装；
2. 打开 `firmware` 这个文件夹（不是整个 yolo 目录）；
3. 等右下角初始化完成，点底部的 **Upload**（→ 图标）；
4. 需要看日志就点 **Serial Monitor**（115200）。

## 运行电脑端

```bash
cd ~/Desktop/yolo
./run.sh --serial /dev/cu.usbserial-1420
```

OLED 显示 0~5；电脑端 3 秒没发数据，屏幕右上角显示 `NO LINK`。

## GPIO 输出（数字 5 → 高电平）

固件会把 **GPIO17** 当作一个开关量输出：

| 手指数量 | GPIO17 |
|---|---|
| **5** | **高电平（3.3V）** |
| 0~4 | 低电平（0V） |
| 上电/复位瞬间 | 低电平 |

更新时机是电脑端发来一帧稳定后的数字（所以要比屏幕上的数字晚约 0.5 秒）。
串口断线时引脚**保持最后一次状态**，不会自动变低。

接线示例：

```
LED：  GPIO17 ── 330Ω ── LED ── GND
继电器：GPIO17 ── 继电器模块 IN（模块 VCC/GND 另接，3.3V 逻辑）
```

> GPIO 只能输出 3.3V / 约 12mA，**不要直接驱动继电器线圈或大电流负载**，
> 用三极管/光耦/继电器模块。模块和 ESP32 要共地。
> 想换引脚或改逻辑，改 `src/main.cpp` 顶部的 `OUTPUT_PIN` 和
> `digitalWrite(OUTPUT_PIN, (total == 5) ? HIGH : LOW);`。

> 想驱动继电器、电磁锁、电机、舵机、220V 负载，或多路输出/选引脚，
> 见 [`docs/完整技术文档.md`](../docs/完整技术文档.md) 第 12 章。

## 常见问题

- **OLED 不亮**：检查 3V3/GND、SDA/SCL 是否接反；少数模块地址是 0x3D，可在 `main.cpp` 里 `display.setI2CAddress(0x3D << 1);`。
- **画面整体偏移/花屏**：可能是 SH1106 而不是 SSD1306，把构造函数换成 `U8G2_SH1106_128X64_NONAME_F_HW_I2C`。
- **杜邦线较长时花屏**：把 `Wire.setClock(400000L)` 改成 `100000L`。
- **上传失败 “Failed to connect”**：按住 BOOT 再点上传，或换一根能传数据的 USB 线；确认串口没被其他程序占用。
- **中文乱码/不显示**：本固件只用英文和数字，避免字库问题。
- **数字一直是 `--`**：电脑端没连上，检查 `./run.sh --serial ...` 是否打印 `[serial] 已连接`。
