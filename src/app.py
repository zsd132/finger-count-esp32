"""Real-time finger counting demo.

Usage
-----
    python main.py                      # default camera, YOLO backend
    python main.py --backend mediapipe  # MediaPipe fallback
    python main.py --source path/to.mp4 --headless --record out/demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .backends import create_backend
from .counting import analyze_hand
from .draw import HUD
from .finger_model import load_default_classifier
from .serial_link import SerialLink, describe_ports
from .tracking import HandTracker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "out"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Count raised fingers from a webcam with YOLO hand keypoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0", help="camera index, video file or image file")
    p.add_argument("--backend", default="auto", choices=["auto", "yolo", "mediapipe"])
    p.add_argument("--model", default=str(PROJECT_ROOT / "models" / "yolo11n-pose-hands.pt"))
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    p.add_argument("--conf", type=float, default=0.35, help="hand detection confidence")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--hold", type=float, default=0.5, help="seconds a new count must stay stable before it is shown/sent")
    p.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    p.add_argument("--no-refine", action="store_true", help="disable the second YOLO crop pass (faster, less precise)")
    p.add_argument("--headless", action="store_true", help="no window (for tests / files)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = endless)")
    p.add_argument("--record", default="", help="write the annotated result to this video path")
    p.add_argument("--dump-dir", default="", help="save annotated frames here (debug)")
    p.add_argument("--dump-every", type=int, default=30)
    p.add_argument("--no-hud", action="store_true", help="draw only the landmarks")
    p.add_argument("--serial", default="", metavar="PORT", help="send counts to a microcontroller (e.g. /dev/cu.usbmodem14201, or 'auto')")
    p.add_argument("--serial-baud", type=int, default=115200)
    p.add_argument("--serial-hz", type=float, default=30.0, help="max frames per second on the wire")
    p.add_argument("--list-serial", action="store_true", help="list serial ports and exit")
    return p.parse_args(argv)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    print(
        f"[info] 正在打开摄像头 {index}（首次运行 macOS 会弹权限框，请点“允许”）…",
        flush=True,
    )
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    cap = None
    for api in backends:
        try:
            c = cv2.VideoCapture(index, api)
        except Exception:
            continue
        if c is not None and c.isOpened():
            cap = c
            break
        c.release()
    if cap is None:
        raise RuntimeError(
            "Cannot open the camera.\n"
            "  - Close other apps using the camera (Zoom / WeChat / OBS ...).\n"
            "  - macOS: System Settings > Privacy & Security > Camera, allow your terminal / IDE.\n"
            "  - Run once from the same app you will use (Terminal or PyCharm)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(
        f"[info] 摄像头已打开：{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        flush=True,
    )
    return cap


def _save_debug(canvas, tracks, backend_name: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = OUTPUT_DIR / f"debug_{stamp}.jpg"
    json_path = OUTPUT_DIR / f"debug_{stamp}.json"
    cv2.imwrite(str(img_path), canvas)
    payload = {
        "backend": backend_name,
        "hands": [
            {
                "count": int(t.count),
                "extended": list(t.analysis.extended) if t.analysis else None,
                "probs": list(t.analysis.probs) if t.analysis else None,
                "bend": list(t.analysis.bend) if t.analysis else None,
                "points": [[round(float(x), 2), round(float(y), 2)] for x, y in t.points],
                "bbox": [round(float(v), 2) for v in t.bbox],
            }
            for t in tracks
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[debug] 已保存 {img_path.name} / {json_path.name}（可发给开发者定位问题）")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

    if args.list_serial:
        describe_ports()
        return 0

    source_str = str(args.source)
    source_path = Path(source_str).expanduser()
    from_file = not source_str.isdigit() and source_path.exists()
    if not source_str.isdigit() and not source_path.exists():
        print(f"[error] source not found: {source_path}", file=sys.stderr)
        return 2
    image_mode = from_file and is_image(source_path)

    print("[info] 正在加载手部模型（首次约 3~10 秒）…", flush=True)
    backend = create_backend(
        args.backend,
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        max_hands=args.max_hands,
        refine=not args.no_refine,
    )
    print(f"[info] backend = {backend.name}", flush=True)
    classifier = load_default_classifier() if backend.name == "mediapipe" else None
    print(
        "[info] 手指判定：学习型分类器" if classifier else "[info] 手指判定：几何规则",
        flush=True,
    )

    cap = None
    frame = None
    if image_mode:
        frame = cv2.imread(str(source_path))
        if frame is None:
            print(f"[error] cannot read image {source_path}", file=sys.stderr)
            backend.close()
            return 2
        camera_fps = 30.0
    else:
        if from_file:
            cap = cv2.VideoCapture(str(source_path))
            camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            try:
                cap = open_camera(int(source_str), args.camera_width, args.camera_height)
            except RuntimeError as exc:
                print(f"[error] {exc}", file=sys.stderr)
                backend.close()
                return 2
            camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if camera_fps <= 1 or camera_fps > 240:
                camera_fps = 30.0
        if cap is None or not cap.isOpened():
            print(f"[error] cannot open source {source_str}", file=sys.stderr)
            backend.close()
            return 2

    mirror = not args.no_mirror and not from_file
    show = (not args.headless) and not image_mode

    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.dump_dir:
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)

    writer = None
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)

    link = None
    if args.serial:
        try:
            link = SerialLink(args.serial, baud=args.serial_baud, max_hz=args.serial_hz)
        except Exception as exc:
            print(f"[warn] 串口不可用（{exc}），继续运行但不发送", file=sys.stderr)
            link = None

    tracker = None
    hud = None
    last_total = 0
    last_counts: list[int] = []
    last_sides: list = []
    last_seen_t = 0.0
    fps = 0.0
    last_t = time.perf_counter()
    frame_idx = 0
    saved = 0
    annotated_frame = None

    if show:
        cv2.namedWindow("Finger Count - YOLO", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Finger Count - YOLO", 1100, 660)

    try:
        while True:
            if image_mode:
                ok = frame_idx == 0
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            if tracker is None:
                tracker = HandTracker(frame.shape[1], hold_seconds=args.hold)
                hud = HUD(frame.shape[1], frame.shape[0])

            hands = backend.detect(frame)
            analyses = [
                analyze_hand(h.points, h.scores, classifier=classifier) for h in hands
            ]
            tracks = tracker.update(hands, analyses)

            now = time.perf_counter()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst = 1.0 / dt
            fps = inst if fps == 0 else (0.88 * fps + 0.12 * inst)

            canvas = frame.copy()
            # Hold the last value for a moment when detection drops out, so a
            # single missed frame does not flash 0 on the OLED.
            if tracks:
                total = sum(t.count for t in tracks)
                counts = [t.count for t in tracks]
                sides = [
                    "L" if t.handedness == "Left" else "R" if t.handedness == "Right" else None
                    for t in tracks
                ]
                last_total, last_counts, last_sides, last_seen_t = total, counts, sides, now
            elif last_total > 0 and (now - last_seen_t) < max(0.6, args.hold):
                total, counts, sides = last_total, last_counts, last_sides
            else:
                total, counts, sides = 0, [], []

            if link is not None:
                link.send(total, counts, sides, fps)

            if not args.no_hud and hud is not None:
                hud.render(
                    canvas,
                    tracks,
                    fps=fps,
                    backend=backend.name,
                    serial_on=bool(link and link.connected),
                    total=total,
                    hands=len(counts),
                )
            annotated_frame = canvas

            if writer is None and args.record:
                Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    args.record, fourcc, camera_fps, (canvas.shape[1], canvas.shape[0])
                )
            if writer is not None:
                writer.write(canvas)

            if args.dump_dir and frame_idx % max(1, args.dump_every) == 0:
                path = Path(args.dump_dir) / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(path), canvas)
                saved += 1

            if show:
                cv2.imshow("Finger Count - YOLO", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    snap = OUTPUT_DIR / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(snap), canvas)
                    print(f"[info] snapshot -> {snap}")
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("d"):
                    _save_debug(canvas, tracks, backend.name)
            else:
                total = sum(t.count for t in tracks)
                if frame_idx % 30 == 0:
                    print(
                        f"frame {frame_idx:4d}  hands={len(tracks)}  "
                        f"counts={[t.count for t in tracks]}  total={total}"
                    )

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        backend.close()
        if link is not None:
            link.close()
            print(f"[serial] 共发送 {link.frames_sent} 帧", flush=True)
        if show:
            cv2.destroyAllWindows()

    if image_mode and annotated_frame is not None:
        out = OUTPUT_DIR / f"{source_path.stem}_annotated.jpg"
        cv2.imwrite(str(out), annotated_frame)
        print(f"[info] annotated image -> {out}")
    if args.dump_dir:
        print(f"[info] saved {saved} debug frame(s) to {args.dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
