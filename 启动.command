#!/bin/bash
# 双击启动。窗口不会自动关闭，出错会停住并留下日志。
cd "$(dirname "$0")" || {
  echo "无法进入项目目录：$(dirname "$0")"
  echo "按回车键关闭…"; read -r; exit 1
}

mkdir -p out
LOG="out/launcher.log"
echo "启动中…（日志：$LOG）"
echo "首次运行会请求摄像头权限，请在弹窗中点“允许”。"
echo

./run.sh "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo
if [ "$status" -ne 0 ]; then
  echo "===== 运行失败（退出码 $status）====="
  echo "完整日志：$(pwd)/$LOG"
  echo "把上面的报错发给我，或查看 README 的“常见问题”。"
else
  echo "程序已退出。日志：$(pwd)/$LOG"
fi
echo
echo "按回车键关闭这个窗口…"
read -r
