"""YOLO 可视化工具 —— 网页版入口（Gradio）。

桌面版 yolo_gui.py 保持不动，本文件复用同一批权重，供 Render 等平台部署。
本地运行：python webapp.py  然后打开 http://127.0.0.1:7860
"""

import os
import time

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "detect": ("yolo11n.pt", "物体检测"),
    "pose": ("yolo11n-pose.pt", "骨架姿态"),
}

# 服务器没有 GPU，torch 会装 CPU 版；本地有卡时自动用 GPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 视频逐帧推理在 CPU 上很慢，限制处理帧数，避免请求卡死
MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "150"))
MAX_WIDTH = int(os.environ.get("MAX_WIDTH", "960"))

_models = {}


def get_model(key):
    """按需加载模型，第一次识别会慢一点。"""
    if key not in _models:
        _models[key] = YOLO(os.path.join(BASE, MODELS[key][0]))
    return _models[key]


def summarize(result):
    """把检测结果总结成一句话，显示在界面上。"""
    names = result.names
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes):
        counts = {}
        for c in boxes.cls.tolist():
            name = names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        parts = ["%s x%d" % (k, v) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
        return "检测到 %d 个目标：%s" % (len(boxes), "，".join(parts[:8]))

    kpts = getattr(result, "keypoints", None)
    if kpts is not None and len(kpts):
        return "检测到 %d 个人体骨架" % len(kpts)

    return "未检测到目标"


def predict_image(img, mode, conf):
    if img is None:
        return None, "请先上传图片或拍照"
    # Gradio 给的是 RGB，Ultralytics 按 OpenCV 习惯接收 BGR
    bgr = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    result = get_model(mode).predict(
        source=bgr, conf=float(conf), device=DEVICE, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return annotated, summarize(result)


def _make_writer(path, size, fps):
    """优先用 imageio-ffmpeg 编 H.264（浏览器能直接播放），失败则退回 OpenCV。"""
    try:
        import imageio_ffmpeg
        gen = imageio_ffmpeg.write_frames(
            path, size=size, fps=fps, codec="libx264",
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            macro_block_size=1, ffmpeg_log_level="error")
        gen.send(None)
        return "ffmpeg", gen
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return "cv2", cv2.VideoWriter(path, fourcc, fps, size)


def _write_frame(writer, rgb):
    kind, obj = writer
    if kind == "ffmpeg":
        obj.send(rgb)
    else:
        obj.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _close_writer(writer):
    kind, obj = writer
    if kind == "ffmpeg":
        obj.close()
    else:
        obj.release()


def predict_video(video_path, mode, conf, progress=gr.Progress()):
    if not video_path:
        return None, "请先选择视频文件"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "视频无法打开"

    model = get_model(mode)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    # 缩小到 MAX_WIDTH 以内，宽高取偶数，H.264 的 yuv420p 需要
    scale = min(1.0, MAX_WIDTH / src_w) if src_w else 1.0
    width = max(2, int(src_w * scale) // 2 * 2)
    height = max(2, int(src_h * scale) // 2 * 2)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    target = min(total_frames, MAX_VIDEO_FRAMES) if total_frames else MAX_VIDEO_FRAMES

    out_path = os.path.join(OUT_DIR, "web_video_%s.mp4" % time.strftime("%Y%m%d_%H%M%S"))

    writer = None
    n = 0
    last_msg = "未检测到目标"
    try:
        writer = _make_writer(out_path, (width, height), fps)
        while n < MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            result = model.predict(
                source=frame, conf=float(conf), device=DEVICE, verbose=False)[0]
            _write_frame(writer, cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
            n += 1
            last_msg = summarize(result)
            progress(min(n / max(target, 1), 1.0), desc="已处理 %d 帧" % n)
        _close_writer(writer)
        writer = None
    finally:
        cap.release()
        if writer is not None:
            _close_writer(writer)

    if n == 0:
        return None, "没有读到任何帧"

    msg = "处理完成：%d 帧，最后一帧%s。结果已转成 MP4，可直接播放或下载。" % (n, last_msg)
    if total_frames > MAX_VIDEO_FRAMES:
        msg += "（视频较长，只处理了前 %d 帧）" % MAX_VIDEO_FRAMES
    return out_path, msg


with gr.Blocks(title="YOLO 可视化工具（网页版）") as demo:
    gr.Markdown(
        "## YOLO 可视化工具（网页版）\n"
        "上传图片、拍照或选一段短视频，识别结果直接显示在页面里。"
        "首次识别需要加载模型，会慢几秒。")

    with gr.Row():
        mode = gr.Radio(
            choices=[(MODELS[k][1], k) for k in MODELS],
            value="detect", label="识别模式")
        conf = gr.Slider(0.1, 0.9, value=0.25, step=0.01, label="置信度阈值")

    with gr.Tab("图片识别"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="上传图片 / 网页摄像头拍照",
                                  type="numpy", sources=["upload", "webcam"], height=420)
                img_btn = gr.Button("开始识别", variant="primary")
            with gr.Column():
                img_out = gr.Image(label="识别结果", type="numpy",
                                   height=420, interactive=False)
                img_msg = gr.Textbox(label="结果说明", lines=2)
        img_btn.click(predict_image, [img_in, mode, conf], [img_out, img_msg])

    with gr.Tab("视频识别"):
        gr.Markdown("CPU 环境下逐帧处理较慢，单次最多处理 %d 帧，超出部分会被截断。"
                    % MAX_VIDEO_FRAMES)
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="上传视频", height=360)
                video_btn = gr.Button("开始识别", variant="primary")
            with gr.Column():
                video_out = gr.Video(label="识别结果", height=360, interactive=False)
                video_msg = gr.Textbox(label="结果说明", lines=2)
        video_btn.click(predict_video, [video_in, mode, conf], [video_out, video_msg])

    gr.Markdown("> 部署在云端时磁盘是临时的，重启后 `outputs/` 里的文件会丢失，"
                "请及时下载结果。")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0", server_port=port, show_error=True)
