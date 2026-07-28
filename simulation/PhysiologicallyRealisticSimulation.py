# coding:utf-8
# @Time    : 2026/4/20
# @Author  : XiaoYuan
# @FileName: PhysiologicallyRealisticSimulation.py
# @description: Realistic Simulation System with Adjustable Physiological Parameters


import os
import cv2
import time
import warnings
import threading
import numpy as np
import tkinter as tk
from datetime import datetime
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


class RealisticSimulationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Realistic simulation system")

        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        self.root.resizable(True, True)

        self.canvas_w, self.canvas_h = screen_w, screen_h

        # Define Initialization Parameters
        self.original = None
        self.display = None
        self.photo = None
        self.ref_tk = None
        self.running = False
        self.thread = None
        self.ref_zero = None
        self.frame_count = 0
        self.save_animation_frames = False
        self.animation_save_path = None

        # Respiratory and Heartbeat frequency parameters (Hz)
        self.heart_hz = tk.DoubleVar(value=2.0)
        self.resp_hz = tk.DoubleVar(value=0.5)
        # Amplitude
        self.arterial_amp = tk.IntVar(value=3)
        self.resp_amp = tk.IntVar(value=3)

        # Other parameters
        self.amp_decay = tk.DoubleVar(value=2.0)
        self.heart_x = tk.IntVar(value=400)
        self.heart_y = tk.IntVar(value=300)
        self.max_frames = tk.IntVar(value=2000)
        self.capture_fps = tk.IntVar(value=200)
        self.heart_major = tk.IntVar(value=18)
        self.heart_minor = tk.IntVar(value=14)
        self.block_x_neg = tk.IntVar(value=-99)
        self.block_x_pos = tk.IntVar(value=99)
        self.block_y_neg = tk.IntVar(value=-78)
        self.block_y_pos = tk.IntVar(value=79)
        self.block_major = tk.IntVar(value=30)
        self.block_minor = tk.IntVar(value=30)

        self.offset_x = 0
        self.offset_y = 0
        self.img_width = 0
        self.img_height = 0
        self.orig_img_width = 0
        self.orig_img_height = 0

        self.last_block_rect = None
        self.phase_accum = 0.0
        self.resp_phase_accum = 0.0
        self.last_phase_time = None

        self.build_ui()
        self.bind_parameter_callbacks()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def bind_parameter_callbacks(self):
        params = [
            self.heart_hz, self.arterial_amp, self.resp_amp, self.amp_decay,
            self.heart_x, self.heart_y, self.heart_major, self.heart_minor,
            self.block_x_neg, self.block_x_pos, self.block_y_neg, self.block_y_pos,
            self.block_major, self.block_minor, self.resp_hz
        ]
        for var in params:
            var.trace_add('write', self.on_param_change)

    def on_param_change(self, *args):
        if self.original is None:
            return
        self.update_ref_zero()
        self.last_block_rect = None
        if not self.running and self.ref_zero is not None:
            frame_with_block, _ = self.make_frame_pair()
            if frame_with_block is not None:
                self.display = frame_with_block
                self.update_show()

    def build_ui(self):
        top_control = ttk.Frame(self.root, padding="5")
        top_control.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(top_control)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="select image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="preview", command=self.start).pack(side=tk.LEFT, padx=(15, 2))
        ttk.Button(row1, text="pause", command=self.stop).pack(side=tk.LEFT, padx=2)

        save_frame = ttk.LabelFrame(row1, text="generate data offline", padding=3)
        save_frame.pack(side=tk.LEFT, padx=15, pady=2)
        ttk.Button(save_frame, text="select folder generate", command=self.toggle_save_animation).pack(side=tk.LEFT, padx=2)
        ttk.Label(save_frame, text="acquisition(FPS):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(save_frame, from_=1, to=1000, textvariable=self.capture_fps, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(save_frame, text="frame count:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Spinbox(save_frame, from_=1, to=50000, textvariable=self.max_frames, width=6).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(top_control)
        row2.pack(fill=tk.X, pady=4)

        # physiological frequency
        freq_frame = ttk.LabelFrame(row2, text="physiological frequency", padding=2)
        freq_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Label(freq_frame, text="heartbeat frequency(Hz):").pack(side=tk.LEFT, padx=2)
        tk.Scale(freq_frame, from_=0.1, to=100.0, variable=self.heart_hz,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=0.1).pack(side=tk.LEFT, padx=2)
        ttk.Label(freq_frame, textvariable=self.heart_hz, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(freq_frame, text="respiratory frequency(Hz):").pack(side=tk.LEFT, padx=5)
        tk.Scale(freq_frame, from_=0.1, to=80.0, variable=self.resp_hz,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=0.1).pack(side=tk.LEFT, padx=5)
        ttk.Label(freq_frame, textvariable=self.resp_hz, width=4).pack(side=tk.LEFT, padx=5)

        # motion amplitude
        motion_frame = ttk.LabelFrame(row2, text="motion amplitude", padding=5)
        motion_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Label(motion_frame, text="heartbeat amplitude(px):").pack(side=tk.LEFT, padx=2)
        tk.Scale(motion_frame, from_=0, to=20, variable=self.arterial_amp,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(motion_frame, textvariable=self.arterial_amp, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(motion_frame, text="respiratory amplitude(px):").pack(side=tk.LEFT, padx=(10, 2))
        tk.Scale(motion_frame, from_=0, to=20, variable=self.resp_amp,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(motion_frame, textvariable=self.resp_amp, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(motion_frame, text="attenuate range:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Scale(motion_frame, from_=0.0, to=20.0, variable=self.amp_decay,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=0.1).pack(side=tk.LEFT, padx=2)
        ttk.Label(motion_frame, textvariable=self.amp_decay, width=4).pack(side=tk.LEFT, padx=2)

        # arterial morphology
        heart_row = ttk.Frame(top_control)
        heart_row.pack(fill=tk.X, pady=4)
        heart_frame = ttk.LabelFrame(heart_row, text="arterial position and dimensions", padding=5)
        heart_frame.pack(fill=tk.X)
        ttk.Label(heart_frame, text="X:").pack(side=tk.LEFT, padx=2)
        tk.Scale(heart_frame, from_=0, to=800, variable=self.heart_x,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, textvariable=self.heart_x, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, text="Y:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Scale(heart_frame, from_=0, to=600, variable=self.heart_y,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, textvariable=self.heart_y, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, text="X axis:").pack(side=tk.LEFT, padx=(15, 2))
        tk.Scale(heart_frame, from_=5, to=50, variable=self.heart_major,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, textvariable=self.heart_major, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, text="Y axis:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Scale(heart_frame, from_=5, to=50, variable=self.heart_minor,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(heart_frame, textvariable=self.heart_minor, width=3).pack(side=tk.LEFT, padx=2)

        # tissue occluder
        block_row = ttk.Frame(top_control)
        block_row.pack(fill=tk.X, pady=4)
        block_frame = ttk.LabelFrame(block_row, text="tissue occluder parameters", padding=5)
        block_frame.pack(fill=tk.X)

        offset_sub = ttk.Frame(block_frame)
        offset_sub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(offset_sub, text="offset range:").pack(side=tk.LEFT, padx=5)
        ttk.Label(offset_sub, text="left").pack(side=tk.LEFT)
        tk.Scale(offset_sub, from_=-200, to=0, variable=self.block_x_neg,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT)
        ttk.Label(offset_sub, textvariable=self.block_x_neg, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(offset_sub, text="right").pack(side=tk.LEFT)
        tk.Scale(offset_sub, from_=0, to=200, variable=self.block_x_pos,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT)
        ttk.Label(offset_sub, textvariable=self.block_x_pos, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(offset_sub, text="upper").pack(side=tk.LEFT, padx=(10, 0))
        tk.Scale(offset_sub, from_=-200, to=40, variable=self.block_y_neg,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT)
        ttk.Label(offset_sub, textvariable=self.block_y_neg, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(offset_sub, text="lower").pack(side=tk.LEFT)
        tk.Scale(offset_sub, from_=0, to=200, variable=self.block_y_pos,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT)
        ttk.Label(offset_sub, textvariable=self.block_y_pos, width=4).pack(side=tk.LEFT, padx=2)

        size_sub = ttk.Frame(block_frame)
        size_sub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(size_sub, text="dimensions:").pack(side=tk.LEFT, padx=5)
        ttk.Label(size_sub, text="major axis:").pack(side=tk.LEFT)
        tk.Scale(size_sub, from_=5, to=100, variable=self.block_major,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_sub, textvariable=self.block_major, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_sub, text="minor axis:").pack(side=tk.LEFT, padx=(10, 2))
        tk.Scale(size_sub, from_=5, to=100, variable=self.block_minor,
                 orient=tk.HORIZONTAL, length=100, showvalue=0, resolution=1).pack(side=tk.LEFT, padx=2)
        ttk.Label(size_sub, textvariable=self.block_minor, width=3).pack(side=tk.LEFT, padx=2)

        image_frame = ttk.Frame(self.root)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.label = ttk.Label(image_frame, relief=tk.FLAT)
        self.label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ref_label = ttk.Label(image_frame, relief=tk.FLAT)
        self.ref_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.stat = tk.StringVar(value="ready")
        ttk.Label(status_frame, textvariable=self.stat, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.save_status = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.save_status, relief=tk.SUNKEN, foreground="blue").pack(
            side=tk.LEFT, fill=tk.X, expand=True)

    def extract_512x512_region(self, image):
        if image is None:
            return None
        center_x, center_y = self.canvas_w // 2, self.canvas_h // 2
        x1 = max(0, center_x - 256)
        y1 = max(0, center_y - 256)
        x2 = min(self.canvas_w, center_x + 256)
        y2 = min(self.canvas_h, center_y + 256)
        region = image[y1:y2, x1:x2]
        if region.shape[0] != 512 or region.shape[1] != 512:
            region = cv2.resize(region, (512, 512), interpolation=cv2.INTER_AREA)
        return region

    def toggle_save_animation(self):
        if self.save_animation_frames:
            self.save_animation_frames = False
            self.save_status.set("animation frame saving stopped")
            self.frame_count = 0
        else:
            save_dir = filedialog.askdirectory(title="select folder to Save Animation Frames")
            if not save_dir:
                return
            max_frames_value = self.max_frames.get()
            if max_frames_value <= 0:
                messagebox.showwarning("warning", "The number of saved frames must be greater than 0.")
                return

            if self.original is None:
                messagebox.showwarning("warning", "please load an image first.")
                return

            self.animation_save_path = save_dir
            self.save_animation_frames = True
            self.frame_count = 0
            self.save_status.set(f"generating {max_frames_value} frames at maximum speed, please wait...")

            info_file = os.path.join(save_dir, "simulate_message.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"====== Animation Frames & Parameter Settings Snapshot ======\n")
                f.write(f"saving time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"image size: 512x512 px\n\n")
                f.write(f"--- basic acquisition parameters ---\n")
                f.write(f"total frames: {max_frames_value}\n")
                f.write(f"acquisition: {self.capture_fps.get()} FPS (dt = {1.0 / self.capture_fps.get():.4f}s)\n\n")
                f.write(f"--- physiological parameters ---\n")
                f.write(f"arterial frequency: {self.heart_hz.get()} Hz (sin)\n")
                f.write(f"arterial pulsation amplitude: {self.arterial_amp.get()} px\n")
                f.write(f"respiratory frequency: {self.resp_hz.get()} Hz\n")
                f.write(f"respiratory amplitude: {self.resp_amp.get()} 像素\n")
                f.write(f"tissue range: {self.amp_decay.get()}\n\n")
                f.write(f"--- arterial morphology and position ---\n")
                f.write(f"center coordinates (X, Y): ({self.heart_x.get()}, {self.heart_y.get()})\n")
                f.write(f"size (major axis, minor axis): ({self.heart_major.get()}, {self.heart_minor.get()})\n\n")
                f.write(f"--- tissue occluder ---\n")
                f.write(f"horizontal random range: [{self.block_x_neg.get()}, {self.block_x_pos.get()}]\n")
                f.write(f"vertical random range: [{self.block_y_neg.get()}, {self.block_y_pos.get()}]\n")
                f.write(f"size (major axis, minor axis): ({self.block_major.get()}, {self.block_minor.get()})\n")

            if not self.running:
                self.start()

    def save_animation_frames_pair(self, frame_with_block, frame_without_block):
        if not self.save_animation_frames or self.animation_save_path is None:
            return
        try:
            max_frames_value = self.max_frames.get()
            if self.frame_count >= max_frames_value:
                self.save_animation_frames = False
                self.save_status.set(f"The maximum frame limit ({max_frames_value}) has been reached. Saving process terminated.")
                self.root.after(5000, lambda: self.save_status.set(""))
                return

            cropped_with_block = self.extract_512x512_region(frame_with_block)
            cropped_without_block = self.extract_512x512_region(frame_without_block)

            blocked_dir = os.path.join(self.animation_save_path, "occluded_data")
            unblocked_dir = os.path.join(self.animation_save_path, "Unoccluded_data")
            os.makedirs(blocked_dir, exist_ok=True)
            os.makedirs(unblocked_dir, exist_ok=True)
            filename = f"{self.frame_count:05d}.jpg"
            cv2.imwrite(os.path.join(blocked_dir, filename), cropped_with_block)
            cv2.imwrite(os.path.join(unblocked_dir, filename), cropped_without_block)

            self.frame_count += 1
            if self.frame_count % 10 == 0 or self.frame_count == max_frames_value:
                progress = (self.frame_count / max_frames_value) * 100
                self.save_status.set(f"Frames Generated: {self.frame_count}/{max_frames_value} ({progress:.1f}%), Offline Fast Mode.")
            if self.frame_count >= max_frames_value:
                self.save_animation_frames = False
                self.save_status.set(f"Dataset generation completed, normal preview restored.")
        except Exception as e:
            print(f"Error saving animation frame: {str(e)}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                pil_img = Image.open(path).convert('L')
                img = np.array(pil_img)
            self.orig_img_height, self.orig_img_width = img.shape[:2]
            h_img, w_img = img.shape[:2]
            canvas = np.full((self.canvas_h, self.canvas_w), 0, dtype=np.uint8)
            y0 = (self.canvas_h - h_img) // 2
            x0 = (self.canvas_w - w_img) // 2
            if h_img <= self.canvas_h and w_img <= self.canvas_w:
                canvas[y0:y0 + h_img, x0:x0 + w_img] = img
            else:
                scale = min(self.canvas_w / w_img, self.canvas_h / h_img)
                new_w, new_h = int(w_img * scale), int(h_img * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                y0 = (self.canvas_h - new_h) // 2
                x0 = (self.canvas_w - new_w) // 2
                canvas[y0:y0 + new_h, x0:x0 + new_w] = img_resized
                self.orig_img_width, self.orig_img_height = new_w, new_h
            self.original = canvas.copy()
            self.display = canvas.copy()
            self.offset_x, self.offset_y = x0, y0
            self.img_width = self.orig_img_width
            self.img_height = self.orig_img_height
            self.ref_zero = self.make_zero_frame()
            self.update_show()
            self.stat.set(f"Loaded: {os.path.basename(path)}")
        except Exception as e:
            self.stat.set(f"Load Failed: {str(e)}")

    def make_zero_frame(self):
        if self.original is None:
            return None
        return self.original.copy()

    def get_ref_image_with_heart(self):
        if self.original is None:
            return None
        img = self.original.copy()
        h, w = img.shape[:2]
        center = np.array([self.heart_x.get() + self.offset_x, self.heart_y.get() + self.offset_y])
        center[0] = np.clip(center[0], 0, w - 1)
        center[1] = np.clip(center[1], 0, h - 1)

        cv2.ellipse(img, (int(center[0]), int(center[1])),
                    (self.heart_major.get(), self.heart_minor.get()),
                    0, 0, 360, 200, -1, lineType=cv2.LINE_AA)
        return img

    def update_ref_zero(self):
        if self.original is None:
            return
        self.ref_zero = self.make_zero_frame()
        if self.ref_zero is not None:
            ref_display = self.get_ref_image_with_heart()
            if ref_display is None:
                return
            ref_roi = ref_display[
                      self.offset_y:self.offset_y + self.img_height,
                      self.offset_x:self.offset_x + self.img_width]
            h, w = ref_roi.shape[:2]
            max_side = 600
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                ref_resized = cv2.resize(ref_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                ref_resized = ref_roi

            margin = 5
            display_h, display_w = ref_resized.shape[0] + 2 * margin, ref_resized.shape[1] + 2 * margin
            ref_show = np.full((display_h, display_w), 0, dtype=np.uint8)
            ref_show[margin:margin + ref_resized.shape[0], margin:margin + ref_resized.shape[1]] = ref_resized
            self.ref_tk = ImageTk.PhotoImage(image=Image.fromarray(ref_show))
            self.ref_label.config(image=self.ref_tk)

    def make_frame_pair(self):
        if self.ref_zero is None:
            return None, None
        try:
            h, w = self.ref_zero.shape[:2]

            if self.save_animation_frames:
                dt = 1.0 / self.capture_fps.get()
                self.last_phase_time = time.time()
            else:
                current_time = time.time()
                if self.last_phase_time is None or not self.running:
                    dt = 0.0
                else:
                    dt = current_time - self.last_phase_time
                self.last_phase_time = current_time

            # 动脉相位
            self.phase_accum += 2 * np.pi * self.heart_hz.get() * dt
            self.phase_accum %= (2 * np.pi)
            # 呼吸相位
            self.resp_phase_accum += 2 * np.pi * self.resp_hz.get() * dt
            self.resp_phase_accum %= (2 * np.pi)

            biological_wave = (np.sin(self.phase_accum) + 1.0) / 2.0
            resp_wave = (np.sin(self.resp_phase_accum) + 1.0) / 2.0

            # ---------- respiratory motion ----------
            resp_scale_amp = self.resp_amp.get() * 0.005
            scale = 1.0 + resp_scale_amp * (2.0 * resp_wave - 1.0)
            center_pt = (w // 2, h // 2)
            M_scale = cv2.getRotationMatrix2D(center_pt, 0, scale)
            scaled_base = cv2.warpAffine(self.ref_zero.astype(np.float32), M_scale, (w, h), borderMode=cv2.BORDER_REFLECT101)
            global_shift_mag = np.round(self.resp_amp.get() * 0.15 * resp_wave)
            y_shift = np.round(global_shift_mag * 0.5)
            M_shift = np.float32([[1, 0, global_shift_mag], [0, 1, y_shift]])
            base = cv2.warpAffine(scaled_base, M_shift, (w, h), borderMode=cv2.BORDER_REFLECT101)

            # ---------- Arterial pulsation ----------
            center = np.array([self.heart_x.get() + self.offset_x, self.heart_y.get() + self.offset_y])
            center[0] = np.clip(center[0], 0, w - 1)
            center[1] = np.clip(center[1], 0, h - 1)

            y, x = np.indices((h, w), dtype=np.float32)
            dx_grid = x - center[0]
            dy_grid = y - center[1]

            decay_factor = 20.0 - self.amp_decay.get()
            major_axis = max(5.0, float(self.heart_major.get()))
            minor_axis = max(5.0, float(self.heart_minor.get()))

            dist_sq = (dx_grid / major_axis) ** 2 + (dy_grid / minor_axis) ** 2
            falloff = np.exp(-decay_factor * dist_sq * 0.2)

            global_tremor_ratio = 0.3
            local_tremor_ratio = 0.7

            displacement = np.round(self.arterial_amp.get() * biological_wave * (
                    global_tremor_ratio + local_tremor_ratio * falloff))

            angle = np.arctan2(dy_grid, dx_grid) + 0.02 * biological_wave

            map_x = (x + displacement * np.cos(angle)).astype(np.float32)
            map_y = (y + displacement * np.sin(angle)).astype(np.float32)

            deformed = cv2.remap(base, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

            # variation in cardiac pulsation amplitude
            pulse_magnitude = self.arterial_amp.get() * 0.05
            scale_factor = 1.0 + pulse_magnitude * biological_wave
            major = max(1, int(self.heart_major.get() * scale_factor))
            minor = max(1, int(self.heart_minor.get() * scale_factor))
            heart_color = 200
            cv2.ellipse(deformed, (int(center[0]), int(center[1])), (major, minor), 0, 0, 360, heart_color, -1,
                        lineType=cv2.LINE_AA)
            frame_without_block = deformed.copy()

            # occluded
            b_major = self.block_major.get()
            b_minor = self.block_minor.get()
            bw = b_major * 2
            bh = b_minor * 2
            cx = w // 2
            cy = h // 2

            def rectangles_overlap(rect1, rect2):
                if rect1 is None or rect2 is None:
                    return False
                x1, y1, w1, h1 = rect1
                x2, y2, w2, h2 = rect2
                return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)

            x_neg = min(self.block_x_neg.get(), self.block_x_pos.get() - 1)
            x_pos = max(self.block_x_pos.get(), self.block_x_pos.get() + 1)
            y_neg = min(self.block_y_neg.get(), self.block_y_pos.get() - 1)
            y_pos = max(self.block_y_pos.get(), self.block_y_pos.get() + 1)

            heart_rect = (
                int(center[0]) - major - 15,
                int(center[1]) - minor - 15,
                major * 2 + 30,
                minor * 2 + 30
            )

            for attempt in range(50):
                x_offset = np.random.randint(x_neg, x_pos + 1)
                y_offset = np.random.randint(y_neg, y_pos + 1)
                x0 = np.clip(cx + x_offset - bw // 2, 0, w - bw)
                y0 = np.clip(cy + y_offset - bh // 2, 0, h - bh)
                block_rect = (x0, y0, bw, bh)

                if rectangles_overlap(block_rect, heart_rect):
                    continue
                if rectangles_overlap(block_rect, self.last_block_rect):
                    continue
                break
            else:
                center_x_img = self.canvas_w // 2 - self.offset_x
                center_y_img = self.canvas_h // 2 - self.offset_y
                crop_x1 = max(0, center_x_img - 256)
                crop_y1 = max(0, center_y_img - 256)
                crop_x2 = min(w, center_x_img + 256)
                crop_y2 = min(h, center_y_img + 256)

                corners = [
                    (crop_x1, crop_y1, bw, bh),
                    (crop_x2 - bw, crop_y1, bw, bh),
                    (crop_x1, crop_y2 - bh, bw, bh),
                    (crop_x2 - bw, crop_y2 - bh, bw, bh)
                ]
                valid_corners = [c for c in corners if not rectangles_overlap(c, heart_rect)]
                if self.last_block_rect is not None:
                    non_overlap_corners = [c for c in valid_corners if not rectangles_overlap(c, self.last_block_rect)]
                    if non_overlap_corners:
                        x0, y0, bw, bh = non_overlap_corners[np.random.randint(0, len(non_overlap_corners))]
                    elif valid_corners:
                        x0, y0, bw, bh = valid_corners[np.random.randint(0, len(valid_corners))]
                    else:
                        x0, y0, bw, bh = corners[np.random.randint(0, 4)]
                else:
                    if valid_corners:
                        x0, y0, bw, bh = valid_corners[np.random.randint(0, len(valid_corners))]
                    else:
                        x0, y0, bw, bh = corners[np.random.randint(0, 4)]

            self.last_block_rect = (x0, y0, bw, bh)

            frame_with_block = deformed.copy()
            ellipse_center = (int(x0 + bw // 2), int(y0 + bh // 2))
            cv2.ellipse(frame_with_block, ellipse_center, (b_major, b_minor), 0, 0, 360, 0, -1, lineType=cv2.LINE_AA)

            return (np.clip(frame_with_block, 0, 255).astype(np.uint8),
                    np.clip(frame_without_block, 0, 255).astype(np.uint8))
        except Exception as e:
            self.stat.set(f"Frame generation error: {str(e)}")
            return None, None

    def anim_thread_func(self):
        while self.running:
            frame_with_block, frame_without_block = self.make_frame_pair()
            if frame_with_block is not None:
                if self.save_animation_frames:
                    self.save_animation_frames_pair(frame_with_block, frame_without_block)
                    if self.frame_count % 5 == 0:
                        self.display = frame_with_block
                        self.update_show()
                else:
                    self.display = frame_with_block
                    self.update_show()
                    time.sleep(1 / 30.0)

    def start(self):
        if self.running or self.original is None:
            return

        self.update_ref_zero()
        self.last_block_rect = None
        self.display = self.original.copy()

        self.running = True
        self.last_phase_time = None
        self.phase_accum = 0.0
        self.resp_phase_accum = 0.0
        self.thread = threading.Thread(target=self.anim_thread_func, daemon=True)
        self.thread.start()
        self.stat.set("Animation running ...")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.save_animation_frames:
            self.save_status.set(f"Animation stopped, total saved frames: {self.frame_count}")
            self.save_animation_frames = False
        self.stat.set("Animation stopped")

    def update_show(self):
        if self.display is None:
            return
        try:
            img_region = self.display[
                         self.offset_y:self.offset_y + self.img_height,
                         self.offset_x:self.offset_x + self.img_width]
            h, w = img_region.shape[:2]
            max_side = 600
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img_region, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_region

            margin = 5
            display_h, display_w = img_resized.shape[0] + 2 * margin, img_resized.shape[1] + 2 * margin
            show = np.full((display_h, display_w), 0, dtype=np.uint8)
            show[margin:margin + img_resized.shape[0], margin:margin + img_resized.shape[1]] = img_resized
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(show))
            self.label.config(image=self.photo)

            if self.ref_zero is not None:
                ref_display = self.get_ref_image_with_heart()
                if ref_display is not None:
                    ref_roi = ref_display[
                              self.offset_y:self.offset_y + self.img_height,
                              self.offset_x:self.offset_x + self.img_width]
                    ref_resized = cv2.resize(ref_roi, (new_w, new_h), interpolation=cv2.INTER_AREA) \
                        if max(h, w) > max_side else ref_roi
                    ref_show = np.full((display_h, display_w), 0, dtype=np.uint8)
                    ref_show[margin:margin + ref_resized.shape[0], margin:margin + ref_resized.shape[1]] = ref_resized
                    self.ref_tk = ImageTk.PhotoImage(image=Image.fromarray(ref_show))
                    self.ref_label.config(image=self.ref_tk)
        except Exception as e:
            self.stat.set(f"Display Error: {str(e)}")

    def on_close(self):
        self.stop()
        self.root.attributes('-fullscreen', False)
        self.root.destroy()


def main():
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    root = tk.Tk()
    app = RealisticSimulationSystem(root)
    root.mainloop()


if __name__ == "__main__":
    main()
