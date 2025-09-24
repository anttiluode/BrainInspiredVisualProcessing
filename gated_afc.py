import os
import sys
# --- Triton autotuner monkeypatch for compatibility ---
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
import types
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale, HORIZONTAL
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Running on device: {device} with dtype: {torch_dtype}")

# ============================================================================ #
#         SECTION 1: ENHANCED HOLOGRAPHIC FIELD WITH FREQUENCY CONTROL        #
# ============================================================================ #

class EnhancedHolographicField(nn.Module):
    """Enhanced field with controllable frequency gating like Matrix8"""
    def __init__(self, dimensions=(64, 64), num_channels=1):
        super().__init__()
        self.dimensions = dimensions
        self.num_channels = num_channels
        
        # Create frequency domain grids
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())

    def evolve(self, field_state, steps=1, low_pass_damping=5.0, high_pass_gain=0.0):
        """Evolve with controllable frequency filtering"""
        with torch.no_grad():
            field_state_f32 = field_state.float()
            field_fft = torch.fft.fft2(field_state_f32)
            
            # Low-pass filter (controls "gist" - overall form)
            low_pass_filter = torch.exp(-self.k2.unsqueeze(0).unsqueeze(0) * low_pass_damping)
            
            # High-pass filter (controls "detail" - fine features)  
            high_pass_filter = 1.0 - torch.exp(-self.k2.unsqueeze(0).unsqueeze(0) * high_pass_gain)
            
            final_filter = low_pass_filter * high_pass_filter
            
            for _ in range(steps):
                field_fft = field_fft * final_filter
            
            result = torch.fft.ifft2(field_fft).real
        return result.to(torch_dtype)

class SensoryEncoder(nn.Module):
    """The 'Eye' and 'V1'. Encodes images to a single-channel fast field."""
    def __init__(self, field_dims=(64, 64)):
        super().__init__()
        self.field = EnhancedHolographicField(field_dims, num_channels=1)
        self.image_to_drive = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, dtype=torch_dtype, device=device), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1, dtype=torch_dtype, device=device),
            nn.AdaptiveAvgPool2d(field_dims)
        )
        self.gamma_freq = 7.5
        self.receptive_threshold = 0.0

    def get_gamma_phase(self):
        return (time.time() * self.gamma_freq * 2 * np.pi) % (2 * np.pi)

    def is_receptive_phase(self, phase):
        return np.cos(phase) > self.receptive_threshold

    def forward(self, image_tensor, fast_low_pass=5.0, fast_high_pass=0.0):
        # Ensure input tensor is correct dtype
        image_tensor = image_tensor.to(torch_dtype)
        drive_pattern = self.image_to_drive(image_tensor)
        fast_pattern = self.field.evolve(drive_pattern, steps=5, 
                                       low_pass_damping=fast_low_pass, 
                                       high_pass_gain=fast_high_pass)
        phase = self.get_gamma_phase()
        receptive = self.is_receptive_phase(phase)
        return fast_pattern, phase, receptive

class AttentionalFieldComputer(nn.Module):
    """Enhanced AFC with frequency control for both fast and slow fields"""
    def __init__(self, text_encoder, vae):
        super().__init__()
        self.fast_field_dims = (64, 64)
        self.slow_field_dims = (64, 64)
        self.latent_channels = 4

        # Enhanced sub-systems with frequency control
        self.sensory_encoder = SensoryEncoder(self.fast_field_dims)
        self.conceptual_field = EnhancedHolographicField(self.slow_field_dims, num_channels=self.latent_channels)
        self.vae = vae
        self.text_encoder = text_encoder

        # Pathways - create with correct dtype and device from start
        self.promoter = nn.Sequential(
            nn.Upsample(size=self.slow_field_dims, mode='bicubic', align_corners=False),
            nn.Conv2d(1, self.latent_channels, kernel_size=7, padding=3, dtype=torch_dtype, device=device), 
            nn.Tanh()
        )
        
        text_embedding_dim = self.text_encoder.config.hidden_size
        slow_field_flat_dim = np.prod(self.slow_field_dims) * self.latent_channels
        self.text_to_field_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, slow_field_flat_dim, dtype=torch_dtype, device=device),
            nn.Tanh()
        )

    @torch.no_grad()
    def forward(self, webcam_frame, text_embedding, slow_field_state, 
                prompt_strength=0.5, slow_low_pass=5.0, slow_high_pass=0.0,
                fast_low_pass=5.0, fast_high_pass=0.0):
        
        # Ensure all inputs are correct dtype
        webcam_frame = webcam_frame.to(torch_dtype)
        slow_field_state = slow_field_state.to(torch_dtype)
        
        # PATH 1: SENSORY PERCEPTION with frequency control
        fast_pattern, phase, receptive = self.sensory_encoder(
            webcam_frame, fast_low_pass, fast_high_pass)
        
        # PATH 2: INTENTIONAL GUIDANCE
        sentence_embedding = text_embedding.mean(dim=1).to(torch_dtype)
        goal_field_flat = self.text_to_field_projector(sentence_embedding)
        goal_field = goal_field_flat.view(1, self.latent_channels, *self.slow_field_dims)

        # INTEGRATION & CONCEPTUALIZATION
        new_slow_field_state = slow_field_state.clone()
        if receptive:
            promoted_sensory = self.promoter(fast_pattern)
            new_slow_field_state = new_slow_field_state + promoted_sensory + (goal_field * prompt_strength)
        
        # Evolve slow field with frequency control
        evolved_slow_field = self.conceptual_field.evolve(
            new_slow_field_state, steps=10,
            low_pass_damping=slow_low_pass,
            high_pass_gain=slow_high_pass
        )
        
        # PREDICTION / "MIND'S EYE" with controlled field
        latent_for_decoder = evolved_slow_field / self.vae.config.scaling_factor
        predicted_percept = self.vae.decode(latent_for_decoder.to(torch_dtype)).sample
        
        return evolved_slow_field, predicted_percept, fast_pattern, receptive, phase

# ============================================================================ #
#         SECTION 2: ENHANCED GUI WITH FREQUENCY CONTROLS                     #
# ============================================================================ #

class EnhancedLiveDemoApp:
    def __init__(self, root, model, tokenizer):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
        self.slow_field_state = torch.zeros(1, self.model.latent_channels, *self.model.slow_field_dims, device=device)
        self.root.title("AFC6 Enhanced - Frequency Gated Prediction Explorer")
        self.root.geometry("1400x900")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Cannot open webcam.")
            self.root.destroy()
            return
            
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Resize((512, 512), antialias=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.setup_gui()
        self.update_loop()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel at the top
        control_frame = ttk.LabelFrame(main_frame, text="Frequency Gating Controls", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Prompt controls
        prompt_frame = ttk.Frame(control_frame)
        prompt_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(prompt_frame, text="Text Prompt (Intention):").pack(side=tk.LEFT, padx=5)
        self.prompt_var = tk.StringVar(value="cinematic, epic, hyperrealistic, portrait of a man")
        self.prompt_entry = ttk.Entry(prompt_frame, textvariable=self.prompt_var, width=40)
        self.prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(prompt_frame, text="Strength:").pack(side=tk.LEFT, padx=5)
        self.strength_var = tk.DoubleVar(value=0.5)
        ttk.Scale(prompt_frame, from_=0.0, to=1.0, variable=self.strength_var, orient=tk.HORIZONTAL, length=100).pack(side=tk.LEFT, padx=5)
        
        # Frequency control sliders
        slider_frame = ttk.Frame(control_frame)
        slider_frame.pack(fill=tk.X)
        
        def add_slider(parent, row, col, text, var, from_val, to_val, resolution=0.1):
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            ttk.Label(frame, text=text, width=15).pack()
            scale = Scale(frame, from_=from_val, to=to_val, resolution=resolution, 
                         variable=var, orient=HORIZONTAL, length=200)
            scale.pack()
            value_label = ttk.Label(frame, text=f"{var.get():.1f}")
            value_label.pack()
            scale.configure(command=lambda v: value_label.config(text=f"{float(v):.1f}"))
            return frame
        
        # Configure grid weights
        for i in range(4):
            slider_frame.grid_columnconfigure(i, weight=1)
        
        # Slow field (conceptual) controls
        self.slow_gist_var = tk.DoubleVar(value=5.0)
        self.slow_detail_var = tk.DoubleVar(value=0.0)
        add_slider(slider_frame, 0, 0, "Slow Gist\n(Low-Pass)", self.slow_gist_var, 0.0, 20.0)
        add_slider(slider_frame, 0, 1, "Slow Detail\n(High-Pass)", self.slow_detail_var, 0.0, 1.0, 0.01)
        
        # Fast field (sensory) controls
        self.fast_gist_var = tk.DoubleVar(value=5.0)
        self.fast_detail_var = tk.DoubleVar(value=0.0)
        add_slider(slider_frame, 0, 2, "Fast Gist\n(Low-Pass)", self.fast_gist_var, 0.0, 20.0)
        add_slider(slider_frame, 0, 3, "Fast Detail\n(High-Pass)", self.fast_detail_var, 0.0, 1.0, 0.01)
        
        # Reset button
        reset_btn = ttk.Button(control_frame, text="Reset Attractor State", command=self.reset_state)
        reset_btn.pack(pady=5)
        
        # Visualization area
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(2):
            vis_frame.grid_rowconfigure(i, weight=1)
            vis_frame.grid_columnconfigure(i, weight=1)

        # Create display frames
        frame1 = self.create_display_frame(vis_frame, "1. Live Input & 'Thalamic Gate'", 0, 0)
        self.input_label = ttk.Label(frame1)
        self.input_label.pack(pady=5, expand=True)
        self.gate_label = ttk.Label(frame1, text="GATE: ...", font=("Helvetica", 16, "bold"))
        self.gate_label.pack(pady=10)

        frame2 = self.create_display_frame(vis_frame, "2. Fast Field (Sensory Pattern)", 1, 0)
        self.fig_fast, self.ax_fast = plt.subplots(figsize=(4, 4))
        self.canvas_fast = self.add_plot_to_frame(self.fig_fast, frame2)

        frame3 = self.create_display_frame(vis_frame, "3. Slow Field (Conceptual Attractor)", 0, 1)
        self.fig_slow, self.ax_slow = plt.subplots(figsize=(4, 4))
        self.canvas_slow = self.add_plot_to_frame(self.fig_slow, frame3)

        frame4 = self.create_display_frame(vis_frame, "4. Prediction (Mind's Eye)", 1, 1)
        self.prediction_label = ttk.Label(frame4)
        self.prediction_label.pack(pady=5, expand=True)

    def create_display_frame(self, parent, title, r, c):
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=r, column=c, sticky="nsew", padx=10, pady=10)
        return frame

    def add_plot_to_frame(self, fig, frame):
        fig.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return canvas

    def update_plot(self, ax, canvas, data, cmap='viridis'):
        ax.clear()
        ax.imshow(data, cmap=cmap)
        ax.axis('off')
        canvas.draw()

    def reset_state(self):
        """Reset the slow field attractor to zero state"""
        self.slow_field_state = torch.zeros(1, self.model.latent_channels, *self.model.slow_field_dims, device=device)
        print("Attractor state reset")

    @torch.no_grad()
    def update_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            img_tk = ImageTk.PhotoImage(img_pil.resize((320, 240)))
            self.input_label.config(image=img_tk)
            self.input_label.image = img_tk

            input_tensor = self.transform(img_pil).unsqueeze(0).to(device)
            
            prompt = self.prompt_var.get()
            text_inputs = self.tokenizer(prompt, padding="max_length", 
                                       max_length=self.tokenizer.model_max_length, 
                                       truncation=True, return_tensors="pt")
            text_embeddings = self.model.text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Get control values
            prompt_strength = self.strength_var.get()
            slow_gist = self.slow_gist_var.get()
            slow_detail = self.slow_detail_var.get()
            fast_gist = self.fast_gist_var.get()
            fast_detail = self.fast_detail_var.get()

            # Run the enhanced model with frequency controls
            new_slow, prediction, fast_pattern, receptive, phase = self.model(
                input_tensor, text_embeddings, self.slow_field_state, 
                prompt_strength=prompt_strength,
                slow_low_pass=slow_gist, slow_high_pass=slow_detail,
                fast_low_pass=fast_gist, fast_high_pass=fast_detail
            )
            self.slow_field_state = new_slow.detach()

            # Update gate display
            gate_text = f"GATE: {'OPEN' if receptive else 'CLOSED'} (Phase: {np.degrees(phase):.1f}°)"
            self.gate_label.config(text=gate_text, foreground="green" if receptive else "red")
            
            # Update visualizations
            self.update_plot(self.ax_fast, self.canvas_fast, 
                           fast_pattern.cpu().squeeze().numpy(), cmap='inferno')
            self.update_plot(self.ax_slow, self.canvas_slow, 
                           self.slow_field_state.cpu().squeeze(0)[0].numpy(), cmap='magma')
            
            # Update prediction display
            pred_np = prediction.cpu().squeeze().permute(1, 2, 0).numpy()
            pred_np = (pred_np * 0.5 + 0.5)
            pred_img = Image.fromarray((np.clip(pred_np, 0, 1) * 255).astype(np.uint8))
            pred_tk = ImageTk.PhotoImage(pred_img.resize((320, 320)))
            self.prediction_label.config(image=pred_tk)
            self.prediction_label.image = pred_tk
            
        self.root.after(33, self.update_loop)

    def on_closing(self):
        if self.cap.isOpened(): 
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    try:
        print("Loading pre-trained models (Tokenizer, Text Encoder, VAE)...")
        MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
        
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch_dtype)
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch_dtype)

        text_encoder.to(device).eval()
        vae.to(device).eval()
        for param in text_encoder.parameters(): 
            param.requires_grad = False
        for param in vae.parameters(): 
            param.requires_grad = False
        
        afc_model = AttentionalFieldComputer(text_encoder, vae).to(device)
        
        print("Enhanced AFC6 model ready. Launching GUI...")

        root = tk.Tk()
        app = EnhancedLiveDemoApp(root, afc_model, tokenizer)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()