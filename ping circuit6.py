import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import threading
from collections import deque

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32

class RetinalGanglionCell:
    """Simulates ON-center/OFF-surround and OFF-center/ON-surround RGCs"""
    def __init__(self, x, y, cell_type='on_center', receptive_field_size=3, spike_threshold=0.1):
        self.x = x
        self.y = y
        self.cell_type = cell_type
        self.rf_size = receptive_field_size
        self.last_response = 0
        self.refractory_period = 0
        self.spike_threshold = spike_threshold
        self.adaptation = 0
        
    def process(self, image_patch):
        """Process local image patch and generate spike probability"""
        if self.refractory_period > 0:
            self.refractory_period -= 1
            return 0
        
        # Calculate center and surround responses
        center = image_patch[self.rf_size//2, self.rf_size//2]
        surround = (image_patch.sum() - center) / (self.rf_size * self.rf_size - 1)
        
        # ON-center cells respond to bright centers with dark surrounds
        # OFF-center cells respond to dark centers with bright surrounds
        if self.cell_type == 'on_center':
            response = center - surround
        else:
            response = surround - center
            
        # Temporal differentiation - respond to changes
        diff = response - self.last_response
        self.last_response = response * 0.9  # Decay
        
        # Apply adaptation
        response_adapted = diff - self.adaptation * 0.1
        self.adaptation = self.adaptation * 0.95 + abs(diff) * 0.05
        
        # Generate spike probability
        if response_adapted > self.spike_threshold:
            spike_prob = min(1.0, response_adapted / self.spike_threshold)
            if np.random.random() < spike_prob:
                self.refractory_period = 3
                return 1
        return 0

class RetinaProcessor:
    """Processes webcam input through retinal ganglion cells"""
    def __init__(self, grid_size=64, rgc_density=2, spike_threshold=0.1):
        self.grid_size = grid_size
        self.rgc_density = rgc_density  # RGCs per grid location
        self.spike_threshold = spike_threshold
        self.cap = None
        self.webcam_active = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Create RGC grid - alternating ON and OFF center cells
        self.rgcs = []
        for i in range(0, grid_size, 2):  # Sample every other pixel for efficiency
            for j in range(0, grid_size, 2):
                for _ in range(rgc_density):
                    cell_type = 'on_center' if np.random.random() > 0.5 else 'off_center'
                    self.rgcs.append(RetinalGanglionCell(i, j, cell_type, spike_threshold=spike_threshold))
        
        # Output spike buffer
        self.spike_buffer = deque(maxlen=10)  # Keep last 10 timesteps
        self.current_spikes = torch.zeros(grid_size, grid_size, dtype=torch_dtype, device=device)
    
    def update_spike_threshold(self, threshold):
        """Update spike threshold for all RGCs"""
        self.spike_threshold = threshold
        for rgc in self.rgcs:
            rgc.spike_threshold = threshold
        
    def start(self):
        """Start webcam capture"""
        if not self.webcam_active:
            self.cap = cv2.VideoCapture(0)
            self.webcam_active = True
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop webcam capture"""
        self.webcam_active = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """Process frames through RGCs"""
        while self.webcam_active:
            ret, frame = self.cap.read()
            if ret:
                # Convert to grayscale and resize
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.grid_size, self.grid_size))
                normalized = resized.astype(float) / 255.0
                
                # Generate spikes from RGCs
                spike_map = np.zeros((self.grid_size, self.grid_size))
                
                for rgc in self.rgcs:
                    # Extract receptive field patch
                    x, y = rgc.x, rgc.y
                    rf = rgc.rf_size
                    x_start = max(0, x - rf//2)
                    x_end = min(self.grid_size, x + rf//2 + 1)
                    y_start = max(0, y - rf//2)
                    y_end = min(self.grid_size, y + rf//2 + 1)
                    
                    patch = normalized[x_start:x_end, y_start:y_end]
                    if patch.size > 0:
                        spike = rgc.process(patch)
                        if spike:
                            spike_map[x, y] += spike
                
                # Store spikes
                with self.lock:
                    self.current_spikes = torch.from_numpy(spike_map).float().to(device)
                    self.spike_buffer.append(self.current_spikes.clone())
            
            time.sleep(0.016)  # ~60 Hz, matching retinal output rate
    
    def get_spikes(self):
        """Get current RGC spike pattern"""
        with self.lock:
            return self.current_spikes.clone()

class LGNRelay:
    """Lateral Geniculate Nucleus - synchronizes retinal input with cortical rhythms"""
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.phase = 0
        self.frequency = 40  # Hz - gamma rhythm
        self.gain = 1.0
        
    def process(self, retinal_spikes, cortical_feedback):
        """Gate retinal signals based on cortical feedback and oscillatory phase"""
        # Update phase
        self.phase += 2 * np.pi * self.frequency / 60  # 60 Hz update rate
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        # Phase-dependent gating (maximum transmission at peak of cycle)
        gate = (np.sin(self.phase) + 1) / 2
        
        # Cortical feedback modulates gain
        if cortical_feedback is not None:
            feedback_gain = 1 + torch.tanh(cortical_feedback) * 0.5
            self.gain = self.gain * 0.95 + feedback_gain.mean().item() * 0.05
        
        # Apply gating and gain
        lgn_output = retinal_spikes * gate * self.gain
        
        # Add some temporal dispersion
        if hasattr(self, 'history'):
            lgn_output = lgn_output * 0.7 + self.history * 0.3
        self.history = lgn_output.clone()
        
        return lgn_output

class V1Layer:
    """Primary visual cortex layer with orientation-selective neurons"""
    def __init__(self, grid_size=64, n_orientations=4):
        self.grid_size = grid_size
        self.n_orientations = n_orientations
        
        # Create Gabor filters for different orientations
        self.gabor_filters = []
        for i in range(n_orientations):
            angle = i * np.pi / n_orientations
            gabor = self._create_gabor(angle)
            self.gabor_filters.append(torch.from_numpy(gabor).float().to(device))
        
        # Simple cells (orientation selective)
        self.simple_cells = torch.zeros(n_orientations, grid_size, grid_size, device=device)
        
        # Complex cells (position invariant)
        self.complex_cells = torch.zeros(n_orientations, grid_size, grid_size, device=device)
        
    def _create_gabor(self, theta, lambda_=10, psi=0, sigma=4, gamma=0.5):
        """Create a Gabor filter for edge detection at specific orientation"""
        size = 11  # Filter size
        x = np.arange(-size//2, size//2 + 1)
        y = np.arange(-size//2, size//2 + 1)
        X, Y = np.meshgrid(x, y)
        
        # Rotation
        X_rot = X * np.cos(theta) + Y * np.sin(theta)
        Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
        
        # Gabor formula
        envelope = np.exp(-(X_rot**2 + gamma**2 * Y_rot**2) / (2 * sigma**2))
        carrier = np.cos(2 * np.pi * X_rot / lambda_ + psi)
        
        gabor = envelope * carrier
        return gabor / np.abs(gabor).sum()
    
    def process(self, lgn_input):
        """Process LGN input through simple and complex cells"""
        # Pad input for convolution
        padded = F.pad(lgn_input.unsqueeze(0).unsqueeze(0), (5, 5, 5, 5), mode='replicate')
        
        # Simple cells: convolution with Gabor filters
        for i, gabor in enumerate(self.gabor_filters):
            response = F.conv2d(padded, gabor.unsqueeze(0).unsqueeze(0), padding=0)
            # Ensure the response matches grid size
            if response.shape[-1] != self.grid_size or response.shape[-2] != self.grid_size:
                response = F.interpolate(response, size=(self.grid_size, self.grid_size), mode='bilinear')
            self.simple_cells[i] = response.squeeze().abs()
        
        # Complex cells: local pooling of simple cells
        for i in range(self.n_orientations):
            pooled = F.avg_pool2d(self.simple_cells[i].unsqueeze(0).unsqueeze(0), 
                                  kernel_size=3, stride=1, padding=1)
            self.complex_cells[i] = pooled.squeeze()
        
        # Combine all orientations
        v1_output = self.complex_cells.max(dim=0)[0]  # Maximum response across orientations
        
        return v1_output, self.complex_cells

class HolographicField(nn.Module):
    """Frequency-domain field that evolves based on spiking input"""
    def __init__(self, dimensions=(64, 64), orientation_bias=None, decay_rate=0.98):
        super().__init__()
        self.dimensions = dimensions
        self.decay_rate = decay_rate
        
        # Create frequency coordinate system
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())
        
        # Optional orientation tuning
        self.orientation_bias = orientation_bias
        if orientation_bias is not None:
            kx, ky = k_grid
            angles = torch.atan2(ky, kx).to(device)
            self.register_buffer('orientation_mask', torch.cos(angles - orientation_bias)**2)
        else:
            self.orientation_mask = None
        
    def evolve(self, field_state):
        """Evolve field state in frequency domain"""
        with torch.no_grad():
            field_fft = torch.fft.fft2(field_state)
            
            if self.orientation_mask is not None:
                field_fft = field_fft * self.orientation_mask.to(device)
                
            evolved_field = torch.fft.ifft2(field_fft).real
            evolved_field = evolved_field * self.decay_rate
            return evolved_field.to(torch_dtype)

class BiologicalVisualSystem:
    """Complete visual processing pipeline from retina to cortex"""
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        
        # Visual processing stages
        self.retina = RetinaProcessor(grid_size)
        self.lgn = LGNRelay(grid_size)
        self.v1 = V1Layer(grid_size)
        
        # Higher cortical layers (holographic fields)
        self.cortical_fields = nn.ModuleList([
            HolographicField((grid_size, grid_size), orientation_bias=i*np.pi/3, 
                           decay_rate=0.95 + i*0.01)
            for i in range(3)
        ])
        
        # Field states
        self.field_states = [torch.zeros(grid_size, grid_size, dtype=torch_dtype, device=device) 
                            for _ in range(3)]
        
        # Parameters
        self.v1_to_field_strength = 0.5
        self.field_coupling = 0.3
        self.feedback_strength = 0.2
        
        # Webcam active flag
        self.webcam_active = False
        
    def step(self):
        """Single processing step through visual hierarchy"""
        # Get retinal spikes
        retinal_spikes = self.retina.get_spikes() if self.webcam_active else torch.zeros(self.grid_size, self.grid_size, device=device)
        
        # Process through LGN with cortical feedback
        cortical_feedback = self.field_states[0] * self.feedback_strength
        lgn_output = self.lgn.process(retinal_spikes, cortical_feedback)
        
        # Process through V1
        v1_output, orientation_maps = self.v1.process(lgn_output)
        
        # Inject V1 output into first cortical field
        self.field_states[0] += v1_output * self.v1_to_field_strength
        
        # Evolve cortical fields with inter-layer coupling
        for i in range(len(self.field_states)):
            # Evolve field
            self.field_states[i] = self.cortical_fields[i].evolve(self.field_states[i])
            
            # Forward coupling to next layer
            if i < len(self.field_states) - 1:
                self.field_states[i+1] += self.field_states[i] * self.field_coupling
            
            # Backward coupling to previous layer
            if i > 0:
                self.field_states[i-1] += self.field_states[i] * self.field_coupling * 0.5
        
        return retinal_spikes, lgn_output, v1_output, orientation_maps

class BiologicalVisualGUI:
    """GUI for the biological visual processing system"""
    def __init__(self, root):
        self.root = root
        self.root.title("Biological Visual Processing System")
        self.root.geometry("1400x900")
        
        self.system = BiologicalVisualSystem(grid_size=64)
        self.running = False
        self.step_count = 0
        
        self.setup_gui()
        self.update_loop()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        sim_controls = ttk.Frame(control_frame)
        sim_controls.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(sim_controls, text="Start/Stop", command=self.toggle_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(sim_controls, text="Reset", command=self.reset_system).pack(side=tk.LEFT, padx=5)
        
        self.webcam_var = tk.BooleanVar(value=False)
        self.webcam_check = ttk.Checkbutton(sim_controls, text="Enable Webcam", 
                                            variable=self.webcam_var, 
                                            command=self.toggle_webcam)
        self.webcam_check.pack(side=tk.LEFT, padx=5)
        
        # Parameters
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="RGC Threshold:").pack(side=tk.LEFT, padx=5)
        self.rgc_threshold_var = tk.DoubleVar(value=0.1)
        threshold_scale = tk.Scale(param_frame, from_=0.01, to=0.5, resolution=0.01, 
                                  variable=self.rgc_threshold_var, orient=tk.HORIZONTAL, length=150)
        threshold_scale.pack(side=tk.LEFT, padx=5)
        threshold_scale.configure(command=self.update_rgc_threshold)
        
        ttk.Label(param_frame, text="V1→Field:").pack(side=tk.LEFT, padx=5)
        self.v1_strength_var = tk.DoubleVar(value=0.5)
        v1_scale = tk.Scale(param_frame, from_=0.0, to=2.0, resolution=0.1, 
                           variable=self.v1_strength_var, orient=tk.HORIZONTAL, length=150)
        v1_scale.pack(side=tk.LEFT, padx=5)
        v1_scale.configure(command=self.update_v1_strength)
        
        ttk.Label(param_frame, text="Field Coupling:").pack(side=tk.LEFT, padx=5)
        self.coupling_var = tk.DoubleVar(value=0.3)
        coupling_scale = tk.Scale(param_frame, from_=0.0, to=1.0, resolution=0.05, 
                                 variable=self.coupling_var, orient=tk.HORIZONTAL, length=150)
        coupling_scale.pack(side=tk.LEFT, padx=5)
        coupling_scale.configure(command=self.update_coupling)
        
        ttk.Label(param_frame, text="Feedback:").pack(side=tk.LEFT, padx=5)
        self.feedback_var = tk.DoubleVar(value=0.2)
        feedback_scale = tk.Scale(param_frame, from_=0.0, to=1.0, resolution=0.05, 
                                 variable=self.feedback_var, orient=tk.HORIZONTAL, length=150)
        feedback_scale.pack(side=tk.LEFT, padx=5)
        feedback_scale.configure(command=self.update_feedback)
        
        ttk.Label(param_frame, text="LGN Gain:").pack(side=tk.LEFT, padx=5)
        self.lgn_gain_var = tk.DoubleVar(value=1.0)
        lgn_scale = tk.Scale(param_frame, from_=0.0, to=3.0, resolution=0.1, 
                            variable=self.lgn_gain_var, orient=tk.HORIZONTAL, length=150)
        lgn_scale.pack(side=tk.LEFT, padx=5)
        lgn_scale.configure(command=self.update_lgn_gain)
        
        # Display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig, axs = plt.subplots(3, 3, figsize=(12, 10))
        axs = axs.flatten()
        
        self.ax_retina = axs[0]
        self.ax_lgn = axs[1]
        self.ax_v1 = axs[2]
        self.ax_orientations = axs[3:7]
        self.ax_fields = axs[7:]
        
        self.ax_retina.set_title("Retinal Ganglion Cells")
        self.ax_lgn.set_title("LGN Output")
        self.ax_v1.set_title("V1 Combined")
        
        for i, ax in enumerate(self.ax_orientations):
            if i < 4:
                ax.set_title(f"V1 Orientation {i*45}°")
            else:
                ax.axis('off')
        
        for i, ax in enumerate(self.ax_fields[:3]):
            ax.set_title(f"Cortical Field {i+1}")
        
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(main_frame, text="Ready - Step: 0")
        self.status_label.pack(fill=tk.X, pady=(5, 0))
    
    def toggle_webcam(self):
        if self.webcam_var.get():
            self.system.retina.start()
            self.system.webcam_active = True
        else:
            self.system.retina.stop()
            self.system.webcam_active = False
    
    def update_rgc_threshold(self, value):
        self.system.retina.update_spike_threshold(float(value))
    
    def update_v1_strength(self, value):
        self.system.v1_to_field_strength = float(value)
    
    def update_coupling(self, value):
        self.system.field_coupling = float(value)
    
    def update_feedback(self, value):
        self.system.feedback_strength = float(value)
    
    def update_lgn_gain(self, value):
        self.system.lgn.gain = float(value)
    
    def toggle_simulation(self):
        self.running = not self.running
    
    def reset_system(self):
        self.system.retina.stop()
        self.system = BiologicalVisualSystem(grid_size=64)
        self.step_count = 0
        self.running = False
        self.webcam_var.set(False)
    
    def update_loop(self):
        if self.running:
            retinal, lgn, v1, orientations = self.system.step()
            self.step_count += 1
            
            self.update_plots(retinal, lgn, v1, orientations, self.system.field_states)
        
        self.root.after(16, self.update_loop)  # ~60 Hz
    
    def update_plots(self, retinal, lgn, v1, orientations, field_states):
        # Retinal spikes
        self.ax_retina.clear()
        retinal_np = retinal.cpu().numpy()
        self.ax_retina.imshow(retinal_np, cmap='Greys', vmin=0, vmax=1)
        self.ax_retina.set_title(f"RGC Spikes ({int(retinal_np.sum())} active)")
        
        # LGN
        self.ax_lgn.clear()
        lgn_np = lgn.cpu().numpy()
        self.ax_lgn.imshow(lgn_np, cmap='hot', vmin=0, vmax=np.percentile(lgn_np, 99))
        self.ax_lgn.set_title(f"LGN (Gain: {self.system.lgn.gain:.2f})")
        
        # V1 combined
        self.ax_v1.clear()
        v1_np = v1.cpu().numpy()
        self.ax_v1.imshow(v1_np, cmap='viridis', vmin=0, vmax=np.percentile(v1_np, 99))
        self.ax_v1.set_title("V1 Complex Cells")
        
        # Orientation maps
        for i, ax in enumerate(self.ax_orientations[:4]):
            ax.clear()
            orient_np = orientations[i].cpu().numpy()
            ax.imshow(orient_np, cmap='twilight', vmin=0, vmax=np.percentile(orient_np, 99))
            ax.set_title(f"Orientation {i*45}°")
        
        # Cortical fields
        for i, (ax, field) in enumerate(zip(self.ax_fields[:3], field_states)):
            ax.clear()
            field_np = field.cpu().numpy()
            ax.imshow(np.abs(field_np), cmap='plasma', vmin=0, vmax=np.percentile(np.abs(field_np), 95))
            ax.set_title(f"Cortical Field {i+1}")
        
        for ax in [self.ax_retina, self.ax_lgn, self.ax_v1] + list(self.ax_orientations[:4]) + list(self.ax_fields[:3]):
            ax.set_xticks([])
            ax.set_yticks([])
        
        self.canvas.draw()
        
        webcam_status = "ON" if self.system.webcam_active else "OFF"
        self.status_label.config(text=f"Step: {self.step_count} - Webcam: {webcam_status} - RGC Spikes: {int(retinal_np.sum())}")
    
    def __del__(self):
        if hasattr(self, 'system'):
            self.system.retina.stop()

if __name__ == "__main__":
    root = tk.Tk()
    app = BiologicalVisualGUI(root)
    
    def on_closing():
        app.system.retina.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()