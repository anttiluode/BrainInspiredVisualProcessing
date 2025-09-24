import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import time
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# --- Graphical Equalizer Widget ---
class GraphicalEQ(tk.Frame):
    """A custom widget that provides a graphical equalizer interface."""
    def __init__(self, parent, num_bands=10, width=400, height=80):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.num_bands = num_bands
        
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='#2E2E2E', highlightthickness=0)
        self.canvas.pack()
        
        # Initialize band gains (y-positions) - from 0.0 (bottom) to 1.0 (top)
        # Default is a gentle curve, keeping center values and reducing extremes
        self.gains = np.array([0.1, 0.4, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.4, 0.1])
        if num_bands != 10: # Fallback for different band counts
            self.gains = np.ones(num_bands)

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

    def _on_release(self, event):
        self.selected_band = None

    def _on_drag(self, event):
        if self.selected_band is not None:
            self._update_gain(event.y)

    def _update_gain(self, y_pos):
        y_clamped = max(0, min(self.height, y_pos))
        gain = 1.0 - (y_clamped / self.height)
        self.gains[self.selected_band] = gain
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        points = self.get_curve_points()
        self.canvas.create_polygon(points, fill='#4A90E2', outline='')
        
        for i, gain in enumerate(self.gains):
            x = (i + 0.5) * self.band_width
            y = (1.0 - gain) * self.height
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill='white', outline='black')

    def get_curve_points(self):
        curve_points = [0, self.height]
        x_coords = np.linspace(0, self.width, self.width)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        for x, gain in zip(x_coords, interp_gains):
            y = (1.0 - gain) * self.height
            curve_points.extend([x, y])
        curve_points.extend([self.width, self.height])
        return curve_points

    def get_filter_shape(self, num_points=256):
        """Returns the filter shape as a lookup table."""
        x_coords = np.linspace(0, self.width, num_points)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        return np.interp(x_coords, band_centers_x, self.gains)

# --- Enhanced Holographic Field ---
class EnhancedHolographicField(nn.Module):
    def __init__(self, dimensions=(64, 64)):
        super().__init__()
        self.dimensions = dimensions
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())

    def evolve(self, field_state, steps=1, low_pass_damping=5.0, high_pass_gain=0.0):
        with torch.no_grad():
            field_fft = torch.fft.fft2(field_state.float())
            low_pass = torch.exp(-self.k2 * low_pass_damping)
            high_pass = 1.0 - torch.exp(-self.k2 * high_pass_gain)
            final_filter = low_pass * high_pass
            for _ in range(steps):
                field_fft = field_fft * final_filter
            return torch.fft.ifft2(field_fft).real.to(torch_dtype)

# --- Sensory Encoder ---
class SensoryEncoder(nn.Module):
    def __init__(self, field_dims=(64, 64)):
        super().__init__()
        self.field = EnhancedHolographicField(field_dims)
        self.image_to_drive = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, dtype=torch_dtype, device=device), nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1, dtype=torch_dtype, device=device),
            nn.AdaptiveAvgPool2d(field_dims)
        )

    def forward(self, image_tensor, fast_low_pass=5.0, fast_high_pass=0.0):
        drive = self.image_to_drive(image_tensor.to(torch_dtype))
        return self.field.evolve(drive, steps=5, low_pass_damping=fast_low_pass, high_pass_gain=fast_high_pass)

# --- Chessboard Structure Analyzer ---
class ChessboardAnalyzer:
    def __init__(self, field_size=64):
        self.field_size = field_size
        self.motion_accumulator = np.zeros((field_size, field_size), dtype=np.float32)
        self.static_accumulator = np.zeros((field_size, field_size), dtype=np.float32)
        self.prev_squares = None
        self.memory_decay = 0.95
        
    def detect_square_grid(self, pattern):
        pattern_norm = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        pattern_smooth = gaussian_filter(pattern_norm, sigma=0.3)
        local_maxima_mask = (pattern_smooth == maximum_filter(pattern_smooth, size=3)) & (pattern_smooth > 0.12)
        square_map = np.zeros_like(pattern_smooth)
        square_map[local_maxima_mask] = pattern_smooth[local_maxima_mask]
        return square_map, np.where(local_maxima_mask)
    
    def analyze_square_motion(self, current_pattern):
        current_squares, _ = self.detect_square_grid(current_pattern.astype(np.float32))
        if self.prev_squares is None:
            self.prev_squares = current_squares
            return np.zeros_like(current_pattern), current_squares
        motion_mask = np.abs(current_squares - self.prev_squares) > 0.1
        self.motion_accumulator = self.motion_accumulator * self.memory_decay + motion_mask
        self.static_accumulator = self.static_accumulator * self.memory_decay + ~motion_mask
        motion_strength = self.motion_accumulator / (self.motion_accumulator + self.static_accumulator + 1e-6)
        self.prev_squares = current_squares
        return motion_strength, current_squares

# --- GUI Application ---
class ChessboardMotionTracker:
    def __init__(self, root):
        self.root = root
        self.root.title("Gated AFC with Graphical Filter")
        self.root.geometry("1200x850")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): self.root.destroy(); return
            
        self.encoder = SensoryEncoder().to(device)
        self.analyzer = ChessboardAnalyzer()
        self.transform = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        
        self.slow_field_state = torch.zeros(1, 1, *self.encoder.field.dimensions, device=device, dtype=torch_dtype)
        self.promoter = nn.Sequential(
            nn.Upsample(size=self.encoder.field.dimensions, mode='bicubic', align_corners=False),
            nn.Conv2d(1, 1, kernel_size=7, padding=3, dtype=torch_dtype, device=device), nn.Tanh()
        ).to(device)
        
        self.fast_gist_var = tk.DoubleVar(value=3.0)
        self.fast_detail_var = tk.DoubleVar(value=0.5)
        self.motion_threshold_var = tk.DoubleVar(value=0.3)
        self.memory_decay_var = tk.DoubleVar(value=0.96)
        
        self.setup_gui()
        self.update_loop()

    def _get_gamma_phase(self, freq=7.5): return (time.time() * freq * 2 * np.pi) % (2 * np.pi)
    def _is_receptive(self, phase, threshold=0.0): return np.cos(phase) > threshold

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        controls1 = ttk.Frame(control_frame); controls1.pack(fill=tk.X, pady=2)
        controls2 = ttk.Frame(control_frame); controls2.pack(fill=tk.X, pady=2)
        
        ttk.Label(controls1, text="Fast Gist:").pack(side=tk.LEFT, padx=5)
        tk.Scale(controls1, from_=0.5, to=10.0, resolution=0.1, variable=self.fast_gist_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Label(controls1, text="Fast Detail:").pack(side=tk.LEFT, padx=5)
        tk.Scale(controls1, from_=0.05, to=1.0, resolution=0.01, variable=self.fast_detail_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls2, text="Motion Threshold:").pack(side=tk.LEFT, padx=5)
        tk.Scale(controls2, from_=0.1, to=0.8, resolution=0.05, variable=self.motion_threshold_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Label(controls2, text="Memory Decay:").pack(side=tk.LEFT, padx=5)
        tk.Scale(controls2, from_=0.8, to=0.99, resolution=0.01, variable=self.memory_decay_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls2, text="Reset Memory", command=self.reset_memory).pack(side=tk.LEFT, padx=10)

        # --- Graphical EQ Filter ---
        eq_frame = ttk.LabelFrame(control_frame, text="Graphical Value Filter", padding=5)
        eq_frame.pack(fill=tk.X, pady=5)
        self.eq_widget = GraphicalEQ(eq_frame, width=500, height=60)
        self.eq_widget.pack(padx=5, pady=5)

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(2): display_frame.grid_rowconfigure(i, weight=1); display_frame.grid_columnconfigure(i, weight=1)
        
        self.input_label = self._create_display_panel(display_frame, "Input Video", 0, 0)
        self.chess_label = self._create_display_panel(display_frame, "Filtered Chessboard Pattern", 0, 1)
        self.struct_label = self._create_display_panel(display_frame, "Square Structure", 1, 0)
        self.motion_label = self._create_display_panel(display_frame, "Motion Isolation", 1, 1)
        self.status_label = ttk.Label(main_frame, text="Ready"); self.status_label.pack(fill=tk.X, pady=5)

    def _create_display_panel(self, parent, text, row, col):
        frame = ttk.LabelFrame(parent, text=text, padding=5)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        label = ttk.Label(frame); label.pack(expand=True)
        return label

    def reset_memory(self):
        self.analyzer.motion_accumulator.fill(0); self.analyzer.static_accumulator.fill(0)
        self.analyzer.prev_squares = None; self.slow_field_state.fill_(0)
        self.status_label.config(text="Memory and Slow Field reset")

    def _apply_graphical_filter(self, array):
        lut = self.eq_widget.get_filter_shape(num_points=256)
        min_val, max_val = array.min(), array.max()
        range_val = max_val - min_val
        if range_val < 1e-6: return array
        norm_array = (array - min_val) / range_val
        indices = (norm_array * 255).astype(np.uint8)
        scaling_mask = lut[indices]
        filtered_norm_array = norm_array * scaling_mask
        return filtered_norm_array * range_val + min_val

    def numpy_to_tkimage(self, array, size=(300, 300), colormap='inferno'):
        if array.ndim != 2: array = (array * 255).clip(0, 255).astype(np.uint8)
        else:
            import matplotlib.cm as cm
            norm_array = (array - array.min()) / (array.max() - array.min() + 1e-6)
            mapped = getattr(cm, colormap)(norm_array)[:, :, :3]
            array = (mapped * 255).astype(np.uint8)
        return ImageTk.PhotoImage(Image.fromarray(array).resize(size, Image.LANCZOS))

    def update_loop(self):
        ret, frame = self.cap.read()
        if not ret: self.root.after(33, self.update_loop); return
        try:
            self.analyzer.memory_decay = self.memory_decay_var.get()
            frame_resized = cv2.resize(frame, (320, 240))
            input_tensor = self.transform(frame_resized).unsqueeze(0).to(device)
            
            receptive = self._is_receptive(self._get_gamma_phase())
            fast_pattern = self.encoder(input_tensor, self.fast_gist_var.get(), self.fast_detail_var.get())

            if receptive:
                self.slow_field_state = (self.slow_field_state * 0.98) + (self.promoter(fast_pattern) * 0.6).detach()
            else:
                self.slow_field_state *= 0.96

            slow_evolved = self.encoder.field.evolve(self.slow_field_state, steps=8,
                low_pass_damping=max(1.0, self.fast_gist_var.get()),
                high_pass_gain=min(1.0, self.fast_detail_var.get() * 3.0))
            
            pattern_np = (fast_pattern + (slow_evolved * 0.9)).clamp(-1, 1).cpu().squeeze().numpy()
            
            filtered_pattern_np = self._apply_graphical_filter(pattern_np)
            motion_strength, square_structure = self.analyzer.analyze_square_motion(filtered_pattern_np)
            motion_mask = motion_strength > self.motion_threshold_var.get()
            
            self._update_display("input", cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0)
            self._update_display("chess", filtered_pattern_np, 'inferno')
            self._update_display("struct", square_structure, 'viridis')
            self._update_display("motion", motion_mask.astype(np.float32), 'gray')
            
            self.status_label.config(text=f"Receptive: {receptive} | Motion points: {np.sum(motion_mask)}")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}"); print(f"Error: {e}")
        self.root.after(33, self.update_loop)

    def _update_display(self, label_name, data, colormap=None):
        label = getattr(self, f"{label_name}_label")
        img = self.numpy_to_tkimage(data, colormap=colormap) if colormap else self.numpy_to_tkimage(data)
        label.config(image=img); label.image = img

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessboardMotionTracker(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
