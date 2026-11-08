import os
import time
import threading
import traceback

import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

MODELS = {
    'detect': ('yolo11n.pt', '物体检测'),
    'pose': ('yolo11n-pose.pt', '骨架姿态'),
}


class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title('YOLO 可视化工具')
        self.root.geometry('1020x800')
        self.root.minsize(900, 700)

        self.mode = tk.StringVar(value='detect')
        self.conf = tk.DoubleVar(value=0.25)
        self.models = {}
        self.running = False
        self.worker = None
        self.photo = None

        self._build_ui()
        self.log('程序已启动  GPU: %s' % (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到GPU，将使用CPU'))

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')

        ttk.Label(top, text='识别模式:').pack(side='left')
        for key, (_, label) in MODELS.items():
            ttk.Radiobutton(top, text=label, value=key,
                            variable=self.mode).pack(side='left', padx=6)

        ttk.Label(top, text='置信度:').pack(side='left', padx=(20, 4))
        scale = ttk.Scale(top, from_=0.1, to=0.9, length=140,
                          variable=self.conf, command=self._conf_changed)
        scale.pack(side='left')
        self.conf_label = ttk.Label(top, text='0.25', width=5)
        self.conf_label.pack(side='left')

        btns = ttk.Frame(self.root, padding=(8, 0))
        btns.pack(fill='x')
        self.btn_image = ttk.Button(btns, text='选择图片识别', command=self.pick_image)
        self.btn_image.pack(side='left', padx=4)
        self.btn_video = ttk.Button(btns, text='选择视频识别', command=self.pick_video)
        self.btn_video.pack(side='left', padx=4)
        self.btn_camera = ttk.Button(btns, text='打开摄像头', command=self.start_camera)
        self.btn_camera.pack(side='left', padx=4)
        self.btn_stop = ttk.Button(btns, text='停止', command=self.stop, state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        ttk.Button(btns, text='打开结果文件夹', command=self.open_results).pack(side='right', padx=4)

        self.canvas = ttk.Label(self.root, anchor='center', background='#1e1e1e')
        self.canvas.pack(fill='both', expand=True, padx=8, pady=8)

        log_frame = ttk.Frame(self.root)
        log_frame.pack(fill='both', padx=8, pady=(0, 8))
        self.logbox = tk.Text(log_frame, height=9, wrap='word',
                              background='#252526', foreground='#d4d4d4', font=('Consolas', 10))
        self.logbox.pack(side='left', fill='both', expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.logbox.yview)
        scroll.pack(side='right', fill='y')
        self.logbox.config(yscrollcommand=scroll.set)

        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def _conf_changed(self, _=None):
        self.conf_label.config(text='%.2f' % self.conf.get())

    # ---------------- helpers ----------------
    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.logbox.insert('end', '[%s] %s\n' % (ts, msg))
        self.logbox.see('end')

    def get_model(self):
        key = self.mode.get()
        if key not in self.models:
            weight = os.path.join(BASE, MODELS[key][0])
            self.log('正在加载模型 %s ...' % MODELS[key][0])
            self.models[key] = YOLO(weight)
        return self.models[key]

    def set_busy(self, busy):
        state = 'disabled' if busy else 'normal'
        for b in (self.btn_image, self.btn_video, self.btn_camera):
            b.config(state=state)
        self.btn_stop.config(state='normal' if busy else 'disabled')

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        max_w, max_h = 960, 520
        img.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(image=self.photo)

    def open_results(self):
        os.startfile(RESULTS)

    # ---------------- actions ----------------
    def pick_image(self):
        path = filedialog.askopenfilename(
            title='选择图片',
            filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp *.webp'), ('所有文件', '*.*')])
        if path:
            self._start(self._worker_image, path)

    def pick_video(self):
        path = filedialog.askopenfilename(
            title='选择视频',
            filetypes=[('视频文件', '*.mp4 *.avi *.mov *.mkv *.wmv'), ('所有文件', '*.*')])
        if path:
            self._start(self._worker_video, path)

    def start_camera(self):
        self._start(self._worker_camera)

    def _start(self, fn, *args):
        if self.running:
            return
        self.running = True
        self.set_busy(True)
        self.worker = threading.Thread(target=fn, args=args, daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.log('正在停止...')

    # ---------------- workers ----------------
    def _worker_image(self, path):
        try:
            model = self.get_model()
            result = model.predict(source=path, device=0,
                                   conf=self.conf.get(), verbose=False)[0]
            frame = result.plot()
            out = os.path.join(RESULTS, 'img_%s.jpg' % time.strftime('%Y%m%d_%H%M%S'))
            cv2.imwrite(out, frame)
            self.root.after(0, self.show_frame, frame)
            self.root.after(0, self.log, '识别完成: %d 个目标 -> %s' % (len(result.boxes), out))
        except Exception:
            self.root.after(0, self.log, '出错: %s' % traceback.format_exc())
        finally:
            self.running = False
            self.root.after(0, self.set_busy, False)

    def _worker_video(self, path):
        cap = None
        writer = None
        try:
            model = self.get_model()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError('视频无法打开: %s' % path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = os.path.join(RESULTS, 'video_%s.mp4' % time.strftime('%Y%m%d_%H%M%S'))
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            n = 0
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(source=frame, device=0,
                                       conf=self.conf.get(), verbose=False)[0]
                annotated = result.plot()
                writer.write(annotated)
                n += 1
                if n % 10 == 0:
                    self.root.after(0, self.show_frame, annotated)
                    self.root.after(0, self.log, '已处理 %d 帧...' % n)

            if self.running:
                self.root.after(0, self.log, '视频处理完成，共 %d 帧 -> %s' % (n, out))
            else:
                self.root.after(0, self.log, '已手动停止，已处理部分保存于 -> %s' % out)
        except Exception:
            self.root.after(0, self.log, '出错: %s' % traceback.format_exc())
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            self.running = False
            self.root.after(0, self.set_busy, False)

    def _worker_camera(self):
        cap = None
        try:
            model = self.get_model()
            self.root.after(0, self.log, '摄像头已打开，按"停止"结束')
            # CAP_DSHOW: DirectShow backend; the default MSMF backend fails to
            # grab frames on this machine, so open the camera explicitly.
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError('摄像头无法打开，请检查是否被其他软件占用')
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(source=frame, device=0,
                                       conf=self.conf.get(), verbose=False)[0]
                self.root.after(0, self.show_frame, result.plot())
        except Exception:
            self.root.after(0, self.log, '出错: %s' % traceback.format_exc())
        finally:
            if cap is not None:
                cap.release()
            self.running = False
            self.root.after(0, self.set_busy, False)
            self.root.after(0, self.log, '摄像头已关闭')

    def on_close(self):
        self.running = False
        if self.worker and self.worker.is_alive():
            self.root.after(300, self.on_close)
            return
        self.root.destroy()


if __name__ == '__main__':
    app = tk.Tk()
    YoloApp(app)
    app.mainloop()
