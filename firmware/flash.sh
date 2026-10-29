#!/usr/bin/env bash
# 编译并烧录 ESP32 固件（不需要 Arduino IDE）。
# 用法：./flash.sh [串口]      不传串口则自动找 USB 串口。
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

export PLATFORMIO_CORE_DIR="${PLATFORMIO_CORE_DIR:-$ROOT/.platformio}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
PIO="$ROOT/.venv/bin/pio"

PORT="${1:-}"
if [ -z "$PORT" ]; then
  for p in /dev/cu.usbserial* /dev/cu.wchusbserial* /dev/cu.SLAB_USBtoUART* /dev/cu.usbmodem*; do
    if [ -e "$p" ]; then PORT="$p"; break; fi
  done
fi
if [ -z "$PORT" ]; then
  echo "没有找到 ESP32 串口。请插好 USB 线，然后运行："
  echo "  ../.venv/bin/python ../main.py --list-serial"
  exit 1
fi

echo "使用串口：$PORT"
echo "提示：如果上传失败，先关掉 Arduino IDE / 串口监视器 / 本项目的识别程序。"
"$PIO" run -t upload --upload-port "$PORT"
echo
echo "烧录完成。用串口监视器看输出："
echo "  $PIO device monitor --port $PORT --baud 115200"
