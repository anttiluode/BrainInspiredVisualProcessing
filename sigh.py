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
        points = self._get_curve_points_for_drawing()
        self.canvas.create_polygon(points, fill='#4A90E2', outline='')
        
        for i, gain in enumerate(self.gains):
            x = (i + 0.5) * self.band_width
            y = (1.0 - gain) * self.height
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill='white', outline='black')

    def _get_curve_points_for_drawing(self):
        curve_points = [0, self.height]
        x_coords = np.linspace(0, self.width, self.width)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        for x, gain in zip(x_coords, interp_gains):
            y = (1.0 - gain) * self.height
            curve_points.extend([x, y])
        curve_points.extend([self.width, self.height])
        return curve_points

    def get_filter_shape_tensor(self, num_points=256):
        """Returns the filter shape as a torch tensor on the correct device."""
        x_coords = np.linspace(0, 1, num_points) # Frequencies from 0 to 1
        band_centers_x = np.linspace(0, 1, self.num_bands)
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        return torch.tensor(interp_gains, dtype=torch.float32, device=device)

# --- Enhanced Holographic Field with True Graphical Filtering ---
class EnhancedHolographicField(nn.Module):
    def __init__(self, dimensions=(64, 64)):
        super().__init__()
        self.dimensions = dimensions
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max()) # k2 is now a map of squared frequencies from 0 (DC) to 1 (max freq)

    def evolve(self, field_state, steps=1, custom_filter_shape=None):
        with torch.no_grad():
            field_fft = torch.fft.fft2(field_state.float())
            
            if custom_filter_shape is not None:
                # --- THIS IS THE NEW CORE LOGIC ---
                # Use k2 map to look up gain values from the 1D EQ curve
                num_points = len(custom_filter_shape)
                # Map each pixel's frequency magnitude (0-1) to an index in the filter shape
                indices = (self.k2 * (num_points - 1)).long().clamp(0, num_points - 1)
                # Gather the gain values to create a 2D filter map
                final_filter = custom_filter_shape[indices]
            else:
                # Fallback to a default wide-pass filter if none is provided
                final_filter = torch.exp(-self.k2 * 1.0)
            
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

    def forward(self, image_tensor, custom_filter_shape=None):
        drive = self.image_to_drive(image_tensor.to(torch_dtype))
        return self.field.evolve(drive, steps=5, custom_filter_shape=custom_filter_shape)

# --- Chessboard Structure Analyzer (Unchanged) ---
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
        self.root.title("Gated AFC with True Graphical Frequency Filter")
        self.root.geometry("1200x850")
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): self.root.destroy(); return
            
        self.encoder = SensoryEncoder().to(device)
        self.analyzer = ChessboardAnalyzer()
        self.transform = lambda x: torch.from_numpy(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        
        self.motion_threshold_var = tk.DoubleVar(value=0.3)
        self.memory_decay_var = tk.DoubleVar(value=0.96)
        
        self.setup_gui()
        self.update_loop()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # --- Graphical EQ Filter ---
        eq_frame = ttk.LabelFrame(control_frame, text="Graphical Frequency Filter (Low Freq -> High Freq)", padding=5)
        eq_frame.pack(fill=tk.X, pady=5)
        self.eq_widget = GraphicalEQ(eq_frame, width=500, height=60, num_bands=10)
        self.eq_widget.pack(padx=5, pady=5, anchor='center')

        # --- Motion Analysis Controls ---
        motion_controls = ttk.Frame(control_frame)
        motion_controls.pack(fill=tk.X, pady=5)
        ttk.Label(motion_controls, text="Motion Threshold:").pack(side=tk.LEFT, padx=5)
        tk.Scale(motion_controls, from_=0.1, to=0.8, resolution=0.05, variable=self.motion_threshold_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Label(motion_controls, text="Memory Decay:").pack(side=tk.LEFT, padx=5)
        tk.Scale(motion_controls, from_=0.8, to=0.99, resolution=0.01, variable=self.memory_decay_var, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        ttk.Button(motion_controls, text="Reset Memory", command=self.reset_memory).pack(side=tk.LEFT, padx=10)

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        for i in range(2): display_frame.grid_rowconfigure(i, weight=1); display_frame.grid_columnconfigure(i, weight=1)
        
        self.input_label = self._create_display_panel(display_frame, "Input Video", 0, 0)
        self.chess_label = self._create_display_panel(display_frame, "Filtered Frequency Pattern", 0, 1)
        self.struct_label = self._create_display_panel(display_frame, "Detected Structure", 1, 0)
        self.motion_label = self._create_display_panel(display_frame, "Motion Isolation", 1, 1)
        self.status_label = ttk.Label(main_frame, text="Ready"); self.status_label.pack(fill=tk.X, pady=5)

    def _create_display_panel(self, parent, text, row, col):
        frame = ttk.LabelFrame(parent, text=text, padding=5)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        label = ttk.Label(frame); label.pack(expand=True)
        return label

    def reset_memory(self):
        self.analyzer.motion_accumulator.fill(0); self.analyzer.static_accumulator.fill(0)
        self.analyzer.prev_squares = None
        self.status_label.config(text="Memory reset")

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
            
            # --- MODIFIED: Get filter shape from EQ and pass to encoder ---
            custom_filter_shape = self.eq_widget.get_filter_shape_tensor()
            fast_pattern = self.encoder(input_tensor, custom_filter_shape=custom_filter_shape)
            
            pattern_np = fast_pattern.cpu().squeeze().numpy()
            
            motion_strength, square_structure = self.analyzer.analyze_square_motion(pattern_np)
            motion_mask = motion_strength > self.motion_threshold_var.get()
            
            self._update_display("input", cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            self._update_display("chess", pattern_np, 'inferno')
            self._update_display("struct", square_structure, 'viridis')
            self._update_display("motion", motion_mask.astype(np.float32), 'gray')
            
            self.status_label.config(text=f"Motion points: {np.sum(motion_mask)}")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}"); print(f"Error: {e}")
        self.root.after(33, self.update_loop)

    def _update_display(self, label_name, data, colormap=None):
        label = getattr(self, f"{label_name}_label")
        img = self.numpy_to_tkimage(data, colormap=colormap if colormap else 'gray')
        label.config(image=img); label.image = img

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessboardMotionTracker(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
