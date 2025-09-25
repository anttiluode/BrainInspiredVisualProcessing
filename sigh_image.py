import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog
import time
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter

# --- Environment Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

# --- Graphical Equalizer Widget ---
class GraphicalEQ(tk.Frame):
    """A custom widget that provides a graphical equalizer interface."""
    def __init__(self, parent, num_bands=10, width=400, height=80, callback=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.callback = callback
        
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
        if self.callback:
            self.callback()

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
                # --- THIS IS THE CORE LOGIC ---
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

# --- Sensory Encoder for Static Images ---
class StaticImageEncoder(nn.Module):
    def __init__(self, field_dims=(64, 64)):
        super().__init__()
        self.field = EnhancedHolographicField(field_dims)
        self.field_dims = field_dims

    def forward(self, image_array, custom_filter_shape=None):
        # Convert numpy array to tensor and resize to field dimensions
        if len(image_array.shape) == 3:
            # Convert RGB to grayscale
            image_gray = np.mean(image_array, axis=2)
        else:
            image_gray = image_array
            
        # Resize to field dimensions
        image_resized = cv2.resize(image_gray, self.field_dims)
        
        # Normalize to 0-1
        image_norm = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).float().to(device)
        
        return self.field.evolve(image_tensor, steps=1, custom_filter_shape=custom_filter_shape)

# --- Structure Analyzer (Simplified for Static Images) ---
class StaticAnalyzer:
    def __init__(self, field_size=64):
        self.field_size = field_size
        
    def detect_square_grid(self, pattern):
        pattern_norm = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-6)
        pattern_smooth = gaussian_filter(pattern_norm, sigma=0.3)
        local_maxima_mask = (pattern_smooth == maximum_filter(pattern_smooth, size=3)) & (pattern_smooth > 0.12)
        square_map = np.zeros_like(pattern_smooth)
        square_map[local_maxima_mask] = pattern_smooth[local_maxima_mask]
        return square_map, np.where(local_maxima_mask)
    
    def analyze_pattern(self, pattern):
        square_structure, _ = self.detect_square_grid(pattern.astype(np.float32))
        return square_structure

# --- GUI Application ---
class StaticImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Static Image Frequency Filter - Clean Sigh System")
        self.root.geometry("1200x850")
        
        self.encoder = StaticImageEncoder().to(device)
        self.analyzer = StaticAnalyzer()
        
        self.current_image = None
        self.original_image = None
        
        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File controls
        file_controls = ttk.Frame(control_frame)
        file_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_controls, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_controls, text="Reset Filter", command=self.reset_filter).pack(side=tk.LEFT, padx=5)
        
        self.status_text = tk.StringVar(value="No image loaded")
        ttk.Label(file_controls, textvariable=self.status_text).pack(side=tk.LEFT, padx=20)
        
        # --- Graphical EQ Filter ---
        eq_frame = ttk.LabelFrame(control_frame, text="Graphical Frequency Filter (Low Freq -> High Freq)", padding=5)
        eq_frame.pack(fill=tk.X, pady=5)
        
        self.eq_widget = GraphicalEQ(eq_frame, width=600, height=80, num_bands=10, callback=self.process_current_image)
        self.eq_widget.pack(padx=5, pady=5, anchor='center')

        # Preset buttons
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preset_frame, text="Low Pass", command=self.preset_low_pass).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="High Pass", command=self.preset_high_pass).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Band Pass", command=self.preset_band_pass).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="Notch", command=self.preset_notch).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_frame, text="All Pass", command=self.preset_all_pass).pack(side=tk.LEFT, padx=5)

        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        for i in range(2): 
            display_frame.grid_rowconfigure(i, weight=1)
            display_frame.grid_columnconfigure(i, weight=1)
        
        self.original_label = self._create_display_panel(display_frame, "Original Image", 0, 0)
        self.filtered_label = self._create_display_panel(display_frame, "Filtered Frequency Pattern", 0, 1)
        self.structure_label = self._create_display_panel(display_frame, "Detected Structure", 1, 0)
        self.info_label = self._create_display_panel(display_frame, "Frequency Information", 1, 1)

    def _create_display_panel(self, parent, text, row, col):
        frame = ttk.LabelFrame(parent, text=text, padding=5)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        label = ttk.Label(frame)
        label.pack(expand=True)
        return label

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image using PIL and convert to numpy
                pil_image = Image.open(file_path)
                self.original_image = np.array(pil_image)
                
                self.status_text.set(f"Loaded: {file_path.split('/')[-1]} ({self.original_image.shape})")
                
                # Display original image
                self._update_display("original", self.original_image)
                
                # Process with current filter settings
                self.process_current_image()
                
            except Exception as e:
                self.status_text.set(f"Error loading image: {str(e)}")

    def process_current_image(self):
        if self.original_image is None:
            return
            
        try:
            # Get filter shape from EQ
            custom_filter_shape = self.eq_widget.get_filter_shape_tensor()
            
            # Process image through encoder
            filtered_pattern = self.encoder(self.original_image, custom_filter_shape=custom_filter_shape)
            
            # Convert to numpy for analysis and display
            pattern_np = filtered_pattern.cpu().numpy()
            
            # Analyze structure
            structure = self.analyzer.analyze_pattern(pattern_np)
            
            # Update displays
            self._update_display("filtered", pattern_np, 'inferno')
            self._update_display("structure", structure, 'viridis')
            
            # Show frequency information
            self._show_frequency_info(pattern_np)
            
        except Exception as e:
            self.status_text.set(f"Processing error: {str(e)}")

    def _show_frequency_info(self, pattern):
        # Create simple frequency information display
        info_text = f"Pattern Stats:\nMin: {pattern.min():.3f}\nMax: {pattern.max():.3f}\nMean: {pattern.mean():.3f}\nStd: {pattern.std():.3f}"
        
        # Create a text image
        info_img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.putText(info_img, "FREQUENCY INFO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        lines = info_text.split('\n')
        for i, line in enumerate(lines):
            cv2.putText(info_img, line, (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        self._update_display("info", info_img)

    def preset_low_pass(self):
        self.eq_widget.gains = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
        self.eq_widget.draw()
        self.process_current_image()

    def preset_high_pass(self):
        self.eq_widget.gains = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.eq_widget.draw()
        self.process_current_image()

    def preset_band_pass(self):
        self.eq_widget.gains = np.array([0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 0.8, 0.4, 0.2, 0.1])
        self.eq_widget.draw()
        self.process_current_image()

    def preset_notch(self):
        self.eq_widget.gains = np.array([1.0, 1.0, 1.0, 0.2, 0.05, 0.05, 0.2, 1.0, 1.0, 1.0])
        self.eq_widget.draw()
        self.process_current_image()

    def preset_all_pass(self):
        self.eq_widget.gains = np.ones(10)
        self.eq_widget.draw()
        self.process_current_image()

    def reset_filter(self):
        self.eq_widget.gains = np.array([0.1, 0.4, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.4, 0.1])
        self.eq_widget.draw()
        self.process_current_image()

    def numpy_to_tkimage(self, array, size=(300, 300), colormap='gray'):
        if array.ndim == 3 and array.shape[2] == 3:
            # RGB image
            array_norm = np.clip(array, 0, 255).astype(np.uint8)
        elif array.ndim == 2:
            # Grayscale - apply colormap
            if colormap != 'gray':
                import matplotlib.cm as cm
                norm_array = (array - array.min()) / (array.max() - array.min() + 1e-6)
                mapped = getattr(cm, colormap)(norm_array)[:, :, :3]
                array_norm = (mapped * 255).astype(np.uint8)
            else:
                norm_array = (array - array.min()) / (array.max() - array.min() + 1e-6)
                array_norm = (norm_array * 255).astype(np.uint8)
        else:
            # Fallback
            array_norm = (array * 255).clip(0, 255).astype(np.uint8)
        
        return ImageTk.PhotoImage(Image.fromarray(array_norm).resize(size, Image.LANCZOS))

    def _update_display(self, label_name, data, colormap=None):
        label = getattr(self, f"{label_name}_label")
        img = self.numpy_to_tkimage(data, colormap=colormap if colormap else 'gray')
        label.config(image=img)
        label.image = img

if __name__ == "__main__":
    root = tk.Tk()
    app = StaticImageProcessor(root)
    root.mainloop()