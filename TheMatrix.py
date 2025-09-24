import sys
import types
import os
# --- Triton autotuner monkeypatch ---
try:
    import triton.runtime
except ImportError:
    sys.modules["triton"] = types.ModuleType("triton")
    sys.modules["triton.runtime"] = types.ModuleType("triton.runtime")
    import triton.runtime

if not hasattr(triton.runtime, "Autotuner"):
    class DummyAutotuner:
        def __init__(self, *args, **kwargs): pass
        def tune(self, *args, **kwargs): return None
    triton.runtime.Autotuner = DummyAutotuner

import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL, messagebox
import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
print(f"Running on: {device} with dtype: {torch_dtype}")
if device == "cuda":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

# --- Graphical Equalizer Widget ---
class GraphicalEQ(tk.Frame):
    def __init__(self, parent, num_bands=10, width=400, height=80):
        super().__init__(parent)
        self.width, self.height, self.num_bands = width, height, num_bands
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='#2E2E2E', highlightthickness=0)
        self.canvas.pack()
        self.gains = np.array([0.1, 0.4, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.4, 0.1])
        if num_bands != 10: self.gains = np.ones(num_bands)
        self.band_width = self.width / self.num_bands
        self.selected_band = None
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.draw()

    def _on_click(self, event):
        band_index = int(event.x // self.band_width)
        if 0 <= band_index < self.num_bands:
            self.selected_band = band_index
            self._update_gain(event.y)

    def _on_release(self, event): self.selected_band = None
    def _on_drag(self, event):
        if self.selected_band is not None: self._update_gain(event.y)

    def _update_gain(self, y_pos):
        gain = 1.0 - (max(0, min(self.height, y_pos)) / self.height)
        self.gains[self.selected_band] = gain
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        points = self._get_curve_points_for_drawing()
        self.canvas.create_polygon(points, fill='#4A90E2', outline='')
        for i, gain in enumerate(self.gains):
            x, y = (i + 0.5) * self.band_width, (1.0 - gain) * self.height
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill='white', outline='black')

    def _get_curve_points_for_drawing(self):
        curve_points = [0, self.height]
        x_coords = np.linspace(0, self.width, self.width)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        for x, gain in zip(x_coords, interp_gains):
            curve_points.extend([x, (1.0 - gain) * self.height])
        curve_points.extend([self.width, self.height])
        return curve_points

    def get_filter_shape_tensor(self, num_points=256):
        x_coords = np.linspace(0, 1, num_points)
        band_centers_x = np.linspace(0, 1, self.num_bands)
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        return torch.tensor(interp_gains, dtype=torch.float32, device=device)

# --- CORE STABILIZATION COMPONENTS ---
class MoireField(nn.Module):
    def __init__(self, base_frequency=8.0, field_size=32):
        super().__init__()
        x = torch.linspace(-1, 1, field_size, dtype=torch_dtype, device=device)
        y = torch.linspace(-1, 1, field_size, dtype=torch_dtype, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('pattern1', torch.sin(base_frequency * np.pi * xx))
        self.register_buffer('pattern2', torch.sin((base_frequency + 0.5) * np.pi * yy))

    def compute_phase_shift(self, current_frame, previous_frame):
        if previous_frame is None: return torch.zeros((1, 1, 32, 32), device=device, dtype=torch_dtype)
        with torch.cuda.amp.autocast():
            current_small = F.interpolate(current_frame, size=(32, 32), mode='bilinear')
            previous_small = F.interpolate(previous_frame, size=(32, 32), mode='bilinear')
            diff = current_small.mean(dim=1, keepdim=True) - previous_small.mean(dim=1, keepdim=True)
            phase_shift = torch.sqrt((diff * self.pattern1)**2 + (diff * self.pattern2)**2)
        return phase_shift

class HolographicSlowField(nn.Module):
    def __init__(self, dimensions=(64, 64), channels=4):
        super().__init__()
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32, device=device) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())

    def evolve(self, field_state, steps=1, custom_filter_shape=None):
        with torch.cuda.amp.autocast(enabled=False):
            field_fft = torch.fft.fft2(field_state.float())
            if custom_filter_shape is not None:
                num_points = len(custom_filter_shape)
                indices = (self.k2 * (num_points - 1)).long().clamp(0, num_points - 1)
                final_filter = custom_filter_shape[indices].unsqueeze(0).unsqueeze(0)
            else:
                final_filter = torch.exp(-self.k2 * 1.0).unsqueeze(0).unsqueeze(0)
            for _ in range(steps): field_fft = field_fft * final_filter
            result = torch.fft.ifft2(field_fft).real
        return result.to(torch_dtype)

@dataclass
class StabilizationState:
    slow_field: Optional[torch.Tensor] = None
    previous_frame: Optional[torch.Tensor] = None

class StabilizedDiffusionPipeline:
    def __init__(self, resolution=512):
        print("Loading Stable Diffusion components...")
        self.resolution = resolution
        self.latent_size = resolution // 8
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch_dtype, safety_checker=None, requires_safety_checker=False
        ).to(device)
        
        # REMOVED: self.pipe.unet.enable_xformers_memory_efficient_attention()
        # This line causes the xformers import error, so we remove it like in Matrix7
        
        self.vae, self.text_encoder, self.tokenizer, self.unet, self.scheduler = \
            self.pipe.vae, self.pipe.text_encoder, self.pipe.tokenizer, self.pipe.unet, self.pipe.scheduler
        self.moire_field = MoireField().to(device)
        self.slow_field_model = HolographicSlowField(dimensions=(self.latent_size, self.latent_size)).to(device)
        self.state = StabilizationState()
        self.ema_alpha, self.anchor_strength = 0.92, 0.6
        if device == "cuda": torch.cuda.empty_cache()
        print(f"Pipeline ready at {resolution}x{resolution}")

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        img_array = np.array(image.convert("RGB").resize((self.resolution, self.resolution))).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device, torch_dtype)
        with torch.cuda.amp.autocast():
            latent = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        return latent, img_tensor

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        with torch.cuda.amp.autocast():
            latent = latent / self.vae.config.scaling_factor
            image = self.vae.decode(latent).sample
            image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
        return Image.fromarray((image * 255).astype(np.uint8))

    @torch.no_grad()
    def generate_stabilized(self, image, prompt, strength, num_inference_steps, dream_mode, custom_filter_shape):
        current_latent, image_tensor = self.encode_image(image)
        phase_shift = self.moire_field.compute_phase_shift(image_tensor, self.state.previous_frame)
        gate = torch.exp(-phase_shift.mean() * 1.5)

        if self.state.slow_field is None: 
            self.state.slow_field = current_latent.clone()
        else:
            blend_factor = self.ema_alpha + (1 - self.ema_alpha) * gate
            # Ensure both tensors are float32 for lerp operation
            current_f32 = current_latent.float()
            slow_f32 = self.state.slow_field.float()
            slow_field_f32 = torch.lerp(current_f32, slow_f32, blend_factor)
            self.state.slow_field = slow_field_f32.to(torch_dtype)
        
        evolved_field = self.slow_field_model.evolve(self.state.slow_field, steps=2, custom_filter_shape=custom_filter_shape)
        self.state.slow_field = evolved_field

        # Ensure both tensors have same dtype for lerp
        if dream_mode:
            init_latent = evolved_field
        else:
            current_f32 = current_latent.float()
            evolved_f32 = evolved_field.float()
            init_latent = torch.lerp(current_f32, evolved_f32, self.anchor_strength).to(torch_dtype)
        
        self.scheduler.set_timesteps(num_inference_steps)
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.cuda.amp.autocast():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(device))[0]
        
        latents = init_latent
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps - 1)
        timesteps = self.scheduler.timesteps[-init_timestep:] if init_timestep > 0 else self.scheduler.timesteps
        
        for t in timesteps:
            with torch.cuda.amp.autocast():
                noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                # Ensure same dtype for lerp in denoising loop
                latents_f32 = latents.float()
                evolved_f32 = evolved_field.float()
                anchor_blend = torch.lerp(latents_f32, evolved_f32, self.anchor_strength * 0.1)
                latents = anchor_blend.to(torch_dtype)

        output_image = self.decode_latent(latents)
        self.state.previous_frame = image_tensor
        return output_image, {'gate': gate.item(), 'phase': phase_shift.mean().item()}

# --- GUI Application ---
class Matrix8EQ_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix8-EQ: True Graphical Frequency Diffusion")
        self.root.geometry("1200x900")
        self.pipeline = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): messagebox.showerror("Webcam Error", "Cannot open webcam"); self.root.destroy(); return
        self.processing = False
        self.current_frame = None
        self.result_queue = queue.Queue()
        self.setup_gui()
        self.init_pipeline()
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()
        self.update_gui()

    def init_pipeline(self):
        if self.pipeline: del self.pipeline; gc.collect(); torch.cuda.empty_cache()
        self.pipeline = StabilizedDiffusionPipeline(resolution=512)
        self.status_label.config(text="Pipeline loaded at 512x512")

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Prompt:").pack(side=tk.LEFT, padx=5)
        self.prompt_var = tk.StringVar(value="cinematic portrait, dramatic lighting")
        ttk.Entry(control_frame, textvariable=self.prompt_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.process_btn = ttk.Button(control_frame, text="Start Processing", command=self.toggle_processing)
        self.process_btn.pack(side=tk.RIGHT, padx=10)
        ttk.Button(control_frame, text="Reset State", command=self.reset_state).pack(side=tk.RIGHT, padx=5)
        
        filter_frame = ttk.LabelFrame(main_frame, text="Graphical Frequency Filter (Low Freq -> High Freq)", padding=10)
        filter_frame.pack(fill=tk.X, pady=10)
        self.eq_widget = GraphicalEQ(filter_frame, width=800, height=80)
        self.eq_widget.pack(anchor='center')
        
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(fill=tk.X, pady=5)
        
        def add_slider(parent, text, var, v_from, v_to, v_res, update_func=None):
            frame = ttk.Frame(parent)
            frame.pack(side=tk.LEFT, padx=10, expand=True)
            ttk.Label(frame, text=text).pack()
            scale = Scale(frame, from_=v_from, to=v_to, resolution=v_res, variable=var, orient=HORIZONTAL, length=200)
            scale.pack()
            value_label = ttk.Label(frame, text=f"{var.get():.2f}")
            value_label.pack()
            command = lambda v, l=value_label: (l.config(text=f"{float(v):.2f}"), update_func(v) if update_func else None)
            scale.configure(command=command)

        self.strength_var = tk.DoubleVar(value=0.6)
        add_slider(slider_frame, "Strength:", self.strength_var, 0.01, 1.0, 0.01)
        self.anchor_var = tk.DoubleVar(value=0.6)
        add_slider(slider_frame, "Anchor:", self.anchor_var, 0.0, 1.0, 0.05, lambda v: setattr(self.pipeline, 'anchor_strength', float(v)))
        self.ema_var = tk.DoubleVar(value=0.92)
        add_slider(slider_frame, "Smoothing:", self.ema_var, 0.8, 0.99, 0.01, lambda v: setattr(self.pipeline, 'ema_alpha', float(v)))
        self.steps_var = tk.IntVar(value=10)
        add_slider(slider_frame, "Steps:", self.steps_var, 5, 20, 1)

        self.dream_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(slider_frame, text="Dream Mode", variable=self.dream_mode_var).pack(side=tk.LEFT, padx=20)
        
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.webcam_label = ttk.Label(ttk.LabelFrame(video_frame, text="Webcam Input")); self.webcam_label.pack(); self.webcam_label.master.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        self.output_label = ttk.Label(ttk.LabelFrame(video_frame, text="Filtered Diffusion Output")); self.output_label.pack(); self.output_label.master.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_processing(self):
        self.processing = not self.processing
        self.process_btn.config(text="Stop" if self.processing else "Start")
        self.status_label.config(text="Processing..." if self.processing else "Stopped")
            
    def reset_state(self):
        if self.pipeline: self.pipeline.state = StabilizationState()
        self.status_label.config(text="State Reset")
        
    def capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret: self.current_frame = frame
            time.sleep(1/60)
            
    def process_loop(self):
        while True:
            if self.processing and self.current_frame is not None and self.pipeline:
                try:
                    pil_image = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                    filter_shape = self.eq_widget.get_filter_shape_tensor()
                    result_image, _ = self.pipeline.generate_stabilized(
                        pil_image, self.prompt_var.get(), self.strength_var.get(),
                        self.steps_var.get(), self.dream_mode_var.get(), filter_shape
                    )
                    self.result_queue.put(result_image)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                time.sleep(0.01)
                
    def update_gui(self):
        if self.current_frame is not None:
            img = Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)).resize((512, 512), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.config(image=imgtk)
            
        try:
            result_image = self.result_queue.get_nowait()
            imgtk_out = ImageTk.PhotoImage(image=result_image)
            self.output_label.imgtk = imgtk_out
            self.output_label.config(image=imgtk_out)
        except queue.Empty: pass
        self.root.after(16, self.update_gui)
        
    def cleanup(self):
        self.processing = False
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Matrix8EQ_GUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()