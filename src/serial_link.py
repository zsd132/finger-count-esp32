"""Send the finger count to a microcontroller over a serial port.

Wire protocol (ASCII, 115200 8N1, one frame per line):

    FC <total> <n_hands> [count_1[:L|R] ...]\\n

Examples
--------
    FC 0 0                 no hand visible
    FC 3 1 3:R             one right hand, three fingers
    FC 5 2 2:L 3:R         two hands, 2 + 3 fingers
    FC 4 1 4               one hand, side unknown (YOLO backend)

Frames are only sent when the value changes, plus a heartbeat every second,
so an Arduino never has to deal with a flood of data.  The parser on the MCU
side can simply use ``sscanf(line, "FC %d %d", &total, &n)``.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

FRAME_PREFIX = "FC"


def list_serial_ports() -> List[Tuple[str, str]]:
    from serial.tools import list_ports

    return [(p.device, p.description or "") for p in list_ports.comports()]


def autodetect_port() -> Optional[str]:
    ports = list_serial_ports()
    preferred = ("usbmodem", "usbserial", "SLAB_USBtoUART", "wchusbserial", "cu.SLAB")
    for device, _ in ports:
        if any(key in device for key in preferred):
            return device
    return ports[0][0] if len(ports) == 1 else None


class SerialLink:
    def __init__(
        self,
        port: str,
        *,
        baud: int = 115200,
        max_hz: float = 30.0,
        heartbeat: float = 1.0,
        reset_wait: float = 1.5,
        reconnect_interval: float = 2.0,
        verbose: bool = True,
    ) -> None:
        self.port = port          # what the user asked for ("auto" stays "auto")
        self.resolved_port: Optional[str] = None
        self.baud = int(baud)
        self.min_interval = 1.0 / max(1.0, float(max_hz))
        self.heartbeat = float(heartbeat)
        self.reset_wait = float(reset_wait)
        self.reconnect_interval = float(reconnect_interval)
        self.verbose = verbose

        self._ser = None
        self._last_value: Optional[Tuple] = None
        self._last_sent = 0.0
        self._last_attempt = 0.0
        self._last_error: Optional[str] = None
        self.frames_sent = 0

        self._open(initial=True)

    # ------------------------------------------------------------------ port
    def _open(self, initial: bool = False) -> bool:
        import serial

        self._last_attempt = time.perf_counter()
        port = self.port
        if port == "auto":
            port = autodetect_port()
            if not port:
                self._report("找不到串口设备（可用 --list-serial 查看）")
                return False
        self.resolved_port = port

        try:
            if "://" in port:
                self._ser = serial.serial_for_url(
                    port, baudrate=self.baud, timeout=0, write_timeout=0.2
                )
            else:
                self._ser = serial.Serial(
                    port,
                    baudrate=self.baud,
                    timeout=0,
                    write_timeout=0.2,
                )
        except Exception as exc:
            self._report(f"串口打开失败（{port}）：{exc}")
            self._ser = None
            return False

        if self.verbose:
            print(f"[serial] 已连接 {port} @ {self.baud} 8N1", flush=True)
        # Opening a real Arduino toggles DTR and reboots it; give it time.
        if initial and self.reset_wait > 0 and "://" not in port:
            if self.verbose:
                print(f"[serial] 等待单片机启动 {self.reset_wait:.1f}s …", flush=True)
            time.sleep(self.reset_wait)
        self._write(f"{FRAME_PREFIX} READY\n")
        return True

    @property
    def connected(self) -> bool:
        return self._ser is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _report(self, message: str) -> None:
        # Only print a given error once per connection attempt.
        if self.verbose and message != self._last_error:
            print(f"[serial] {message}", flush=True)
        self._last_error = message

    # ------------------------------------------------------------------ send
    def send(
        self,
        total: int,
        counts: Sequence[int],
        sides: Sequence[Optional[str]] | None = None,
        fps: float = 0.0,
    ) -> bool:
        counts = tuple(int(c) for c in counts)
        sides = tuple(sides or ())
        value = (int(total), counts, sides)

        now = time.perf_counter()
        changed = value != self._last_value
        due_heartbeat = (now - self._last_sent) >= self.heartbeat
        if not changed and not due_heartbeat:
            return False
        if now - self._last_sent < self.min_interval:
            return False

        if self._ser is None:
            if now - self._last_attempt >= self.reconnect_interval:
                self._open()
            if self._ser is None:
                return False

        frame = self._format(total, counts, sides)
        try:
            self._write(frame)
        except Exception as exc:  # unplugged, port busy, ...
            self._report(f"串口写入失败：{exc}")
            self._close_quietly()

        self._last_value = value
        self._last_sent = now
        return True

    @staticmethod
    def _format(total: int, counts: Sequence[int], sides: Sequence[Optional[str]] = ()) -> str:
        parts = [FRAME_PREFIX, str(int(total)), str(len(counts))]
        for i, count in enumerate(counts):
            side = sides[i] if i < len(sides) else None
            if isinstance(side, str) and side.upper() in ("L", "R"):
                parts.append(f"{int(count)}:{side.upper()}")
            else:
                parts.append(str(int(count)))
        return " ".join(parts) + "\n"

    def _write(self, text: str) -> None:
        if self._ser is None:
            raise IOError("serial port is not open")
        self._ser.write(text.encode("ascii"))
        self._ser.flush()
        if text.startswith(FRAME_PREFIX + " ") and text.strip() != f"{FRAME_PREFIX} READY":
            self.frames_sent += 1

    # ----------------------------------------------------------------- close
    def _close_quietly(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def close(self) -> None:
        self._close_quietly()


def describe_ports() -> None:
    ports = list_serial_ports()
    if not ports:
        print("没有发现串口设备。请确认单片机已插上，并安装了对应的 USB 驱动。")
        return
    print("可用串口：")
    for device, desc in ports:
        print(f"  {device:35s} {desc}")
    print("\nmacOS 一般选 /dev/cu.xxx（不要选 /dev/tty.xxx，可能会卡住等 DCD）。")
