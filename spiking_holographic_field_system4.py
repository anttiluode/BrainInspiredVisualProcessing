import sys
import types
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32  # Use float32 for better precision in this simulation

class GraphicalEQ(tk.Frame):
    """Graphical equalizer for frequency domain control"""
    def __init__(self, parent, num_bands=10, width=400, height=80):
        super().__init__(parent)
        self.width, self.height, self.num_bands = width, height, num_bands
        
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg='#2E2E2E', highlightthickness=0)
        self.canvas.pack()
        
        # Default filter shape - emphasizes mid frequencies
        self.gains = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.1])
        if num_bands != 10: 
            self.gains = np.ones(num_bands) * 0.5
            
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
        gain = 1.0 - (max(0, min(self.height, y_pos)) / self.height)
        self.gains[self.selected_band] = gain
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        points = self._get_curve_points()
        self.canvas.create_polygon(points, fill='#4A90E2', outline='')
        
        for i, gain in enumerate(self.gains):
            x, y = (i + 0.5) * self.band_width, (1.0 - gain) * self.height
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill='white', outline='black')

    def _get_curve_points(self):
        curve_points = [0, self.height]
        x_coords = np.linspace(0, self.width, self.width)
        band_centers_x = (np.arange(self.num_bands) + 0.5) * self.band_width
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        for x, gain in zip(x_coords, interp_gains):
            curve_points.extend([x, (1.0 - gain) * self.height])
        curve_points.extend([self.width, self.height])
        return curve_points

    def get_filter_shape_tensor(self, num_points=64):
        """Returns filter shape as tensor for frequency domain operations"""
        x_coords = np.linspace(0, 1, num_points)
        band_centers_x = np.linspace(0, 1, self.num_bands)
        interp_gains = np.interp(x_coords, band_centers_x, self.gains)
        return torch.tensor(interp_gains, dtype=torch_dtype, device=device)

class HolographicField(nn.Module):
    """Frequency-domain field that evolves based on spiking input"""
    def __init__(self, dimensions=(64, 64), orientation_bias=None):
        super().__init__()
        self.dimensions = dimensions
        
        # Create frequency coordinate system
        k_freq = [torch.fft.fftfreq(n, d=1/n, dtype=torch.float32) for n in dimensions]
        k_grid = torch.meshgrid(*k_freq, indexing='ij')
        k2_tensor = sum(k**2 for k in k_grid)
        self.register_buffer('k2', k2_tensor / k2_tensor.max())
        
        # Optional orientation tuning (for grid cell-like behavior)
        self.orientation_bias = orientation_bias
        if orientation_bias is not None:
            # Compute angle in frequency domain and move to device
            kx, ky = k_grid
            angles = torch.atan2(ky, kx).to(device)  # Move angles to device
            self.register_buffer('orientation_mask', torch.cos(angles - orientation_bias)**2)
        else:
            self.orientation_mask = None
        
    def evolve(self, field_state, custom_filter_shape=None):
        """Evolve field state in frequency domain with optional filtering"""
        with torch.no_grad():
            # Transform to frequency domain
            field_fft = torch.fft.fft2(field_state)
            
            if custom_filter_shape is not None:
                # Apply frequency-selective filtering
                num_points = len(custom_filter_shape)
                indices = (self.k2 * (num_points - 1)).long().clamp(0, num_points - 1)
                filter_2d = custom_filter_shape[indices]
                field_fft = field_fft * filter_2d
                
            if self.orientation_mask is not None:
                field_fft = field_fft * self.orientation_mask.to(device)  # Ensure mask is on the same device
                
            # Transform back to spatial domain
            evolved_field = torch.fft.ifft2(field_fft).real
            
            # Apply natural decay (increased slightly for persistence)
            evolved_field = evolved_field * 0.98  # Changed from 0.95 for slower decay
            
            return evolved_field.to(torch_dtype)

class SpikingNeuronGrid:
    """Grid of spiking neurons with ephaptic field coupling from multiple layers"""
    def __init__(self, grid_size=64):
        self.grid_size = grid_size
        self.n_neurons = grid_size * grid_size
        
        # Neuronal parameters
        self.v_membrane = torch.zeros(grid_size, grid_size, dtype=torch_dtype, device=device)
        self.v_threshold = torch.full((grid_size, grid_size), 1.0, dtype=torch_dtype, device=device)
        self.v_reset = torch.full((grid_size, grid_size), 0.0, dtype=torch_dtype, device=device)
        self.v_rest = torch.full((grid_size, grid_size), 0.0, dtype=torch_dtype, device=device)
        
        # Adaptation variables
        self.adaptation = torch.zeros(grid_size, grid_size, dtype=torch_dtype, device=device)
        
        # Parameters (tweaked for sustained activity)
        self.tau_membrane = 10.0
        self.tau_adaptation = 20.0  # Faster adaptation decay to reduce suppression
        self.adaptation_strength = 0.05  # Weaker adaptation to allow ongoing firing
        self.noise_strength = 0.05
        self.field_coupling = 2.0
        self.baseline_current = 0.2  # Initial baseline input for tonic excitation
        
    def update(self, field_states, dt=1.0):
        """Update neuron states based on summed Laplacians from multiple field layers"""
        laplacian_sum = torch.zeros(self.grid_size, self.grid_size, dtype=torch_dtype, device=device)
        
        for field_state in field_states:
            # Calculate field influence via curvature (Laplacian) for each layer
            h, w = field_state.shape
            field_padded = torch.zeros(h + 2, w + 2, dtype=field_state.dtype, device=field_state.device)
            
            # Copy center
            field_padded[1:-1, 1:-1] = field_state
            
            # Reflect edges
            field_padded[0, 1:-1] = field_state[0, :]      # top edge
            field_padded[-1, 1:-1] = field_state[-1, :]    # bottom edge
            field_padded[1:-1, 0] = field_state[:, 0]      # left edge
            field_padded[1:-1, -1] = field_state[:, -1]    # right edge
            
            # Reflect corners
            field_padded[0, 0] = field_state[0, 0]         # top-left
            field_padded[0, -1] = field_state[0, -1]       # top-right
            field_padded[-1, 0] = field_state[-1, 0]       # bottom-left
            field_padded[-1, -1] = field_state[-1, -1]     # bottom-right
            
            field_north = field_padded[:-2, 1:-1]
            field_south = field_padded[2:, 1:-1]
            field_east = field_padded[1:-1, 2:]
            field_west = field_padded[1:-1, :-2]
            field_center = field_state
            
            # Compute Laplacian for this layer
            laplacian = (field_north + field_south + field_east + field_west - 4 * field_center)
            laplacian_sum += laplacian
        
        # Update membrane potential with summed influence + baseline
        input_current = self.field_coupling * laplacian_sum - self.adaptation + self.baseline_current
        noise = torch.randn_like(self.v_membrane) * self.noise_strength
        dv = (-(self.v_membrane - self.v_rest) + input_current + noise) / self.tau_membrane * dt
        self.v_membrane += dv
        
        # Generate spikes
        spikes = (self.v_membrane >= self.v_threshold).float()
        
        # Reset membrane potential for neurons that spiked
        self.v_membrane = torch.where(spikes > 0, self.v_reset, self.v_membrane)
        
        # Update adaptation
        dadaptation = (-self.adaptation + self.adaptation_strength * spikes) / self.tau_adaptation * dt
        self.adaptation += dadaptation
        
        return spikes
    
    def update_baseline(self, value):
        """Update baseline current dynamically"""
        self.baseline_current = float(value)

class LayeredSpikingHolographicSystem:
    """Main system with layered holographic fields and spiking neurons"""
    def __init__(self, grid_size=64, num_layers=2):
        self.grid_size = grid_size
        self.neuron_grid = SpikingNeuronGrid(grid_size)
        
        # Multiple holographic field layers with optional orientation biases for grid-like patterns
        orientations = [0, torch.pi / 3, 2 * torch.pi / 3]  # 60-degree offsets for hexagonal grids
        self.holographic_layers = nn.ModuleList([
            HolographicField(dimensions=(grid_size, grid_size), orientation_bias=orientations[i % len(orientations)] if i > 0 else None)
            for i in range(num_layers)
        ])
        
        # Field states for each layer
        self.field_states = [torch.zeros(grid_size, grid_size, dtype=torch_dtype, device=device) for _ in range(num_layers)]
        
        # Spike injection parameters
        self.spike_strength = 0.5
        self.layer_coupling = 0.5  # Coupling between layers
        
    def step(self, custom_filter_shapes=None):
        """Single simulation step with layered fields"""
        if custom_filter_shapes is None or len(custom_filter_shapes) != len(self.holographic_layers):
            custom_filter_shapes = [None] * len(self.holographic_layers)
        
        # Update neurons based on all field layers
        spikes = self.neuron_grid.update(self.field_states)
        
        # Inject spikes into the first (input) layer
        spike_injection = spikes * self.spike_strength
        self.field_states[0] += spike_injection
        
        # Evolve each layer sequentially, with coupling from previous
        for i in range(len(self.holographic_layers)):
            # Add coupling from previous layer if exists
            if i > 0:
                self.field_states[i] += self.field_states[i-1] * self.layer_coupling
            
            # Evolve current layer
            self.field_states[i] = self.holographic_layers[i].evolve(
                self.field_states[i], custom_filter_shape=custom_filter_shapes[i]
            )
        
        return spikes, self.field_states
    
    def inject_pattern(self, pattern_type='random_cluster', strength=2.0):
        """Inject a specific pattern of spikes to seed activity"""
        if pattern_type == 'random_cluster':
            # Create a random cluster of activity
            center_x, center_y = self.grid_size // 2, self.grid_size // 2
            radius = 8
            for i in range(50):  # 50 random spikes in cluster
                dx = np.random.randint(-radius, radius)
                dy = np.random.randint(-radius, radius)
                x, y = center_x + dx, center_y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.neuron_grid.v_membrane[x, y] += strength
                    
        elif pattern_type == 'wave':
            # Create a traveling wave
            y_center = self.grid_size // 2
            for x in range(self.grid_size // 4, 3 * self.grid_size // 4):
                for dy in range(-3, 4):
                    y = y_center + dy
                    if 0 <= y < self.grid_size:
                        self.neuron_grid.v_membrane[x, y] += strength

class LayeredSpikingHolographicGUI:
    """GUI for the layered spiking holographic system"""
    def __init__(self, root, num_layers=2):
        self.root = root
        self.root.title("Layered Spiking Holographic Field System")
        self.root.geometry("1400x900")
        
        self.system = LayeredSpikingHolographicSystem(grid_size=64, num_layers=num_layers)
        self.running = False
        self.step_count = 0
        self.num_layers = num_layers
        
        self.setup_gui()
        self.update_loop()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Simulation controls
        sim_controls = ttk.Frame(control_frame)
        sim_controls.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(sim_controls, text="Start/Stop", command=self.toggle_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(sim_controls, text="Reset", command=self.reset_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_controls, text="Inject Cluster", 
                  command=lambda: self.system.inject_pattern('random_cluster')).pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_controls, text="Inject Wave", 
                  command=lambda: self.system.inject_pattern('wave')).pack(side=tk.LEFT, padx=5)
        
        # Frequency filters for each layer
        self.eq_widgets = []
        for i in range(self.num_layers):
            eq_frame = ttk.LabelFrame(control_frame, text=f"Layer {i+1} Frequency Filter (Low -> High)", padding=5)
            eq_frame.pack(fill=tk.X, pady=5)
            eq_widget = GraphicalEQ(eq_frame, width=600, height=60, num_bands=10)
            eq_widget.pack(pady=5, anchor='center')
            self.eq_widgets.append(eq_widget)
        
        # Parameter controls
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        # Field coupling
        ttk.Label(param_frame, text="Field Coupling:").pack(side=tk.LEFT, padx=5)
        self.coupling_var = tk.DoubleVar(value=2.0)
        coupling_scale = tk.Scale(param_frame, from_=0.0, to=5.0, resolution=0.1, 
                                 variable=self.coupling_var, orient=tk.HORIZONTAL, length=150)
        coupling_scale.pack(side=tk.LEFT, padx=5)
        coupling_scale.configure(command=self.update_coupling)
        
        # Noise strength
        ttk.Label(param_frame, text="Noise:").pack(side=tk.LEFT, padx=5)
        self.noise_var = tk.DoubleVar(value=0.05)
        noise_scale = tk.Scale(param_frame, from_=0.0, to=0.2, resolution=0.01, 
                              variable=self.noise_var, orient=tk.HORIZONTAL, length=150)
        noise_scale.pack(side=tk.LEFT, padx=5)
        noise_scale.configure(command=self.update_noise)
        
        # Layer coupling
        ttk.Label(param_frame, text="Layer Coupling:").pack(side=tk.LEFT, padx=5)
        self.layer_coupling_var = tk.DoubleVar(value=0.5)
        layer_coupling_scale = tk.Scale(param_frame, from_=0.0, to=1.0, resolution=0.1, 
                                       variable=self.layer_coupling_var, orient=tk.HORIZONTAL, length=150)
        layer_coupling_scale.pack(side=tk.LEFT, padx=5)
        layer_coupling_scale.configure(command=self.update_layer_coupling)
        
        # Baseline current
        ttk.Label(param_frame, text="Baseline Current:").pack(side=tk.LEFT, padx=5)
        self.baseline_var = tk.DoubleVar(value=0.2)
        baseline_scale = tk.Scale(param_frame, from_=0.0, to=1.0, resolution=0.01, 
                                 variable=self.baseline_var, orient=tk.HORIZONTAL, length=150)
        baseline_scale.pack(side=tk.LEFT, padx=5)
        baseline_scale.configure(command=self.update_baseline)
        
        # Display panels
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figures (dynamic based on layers)
        num_rows = 2 + self.num_layers // 2
        self.fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
        axs = axs.flatten()
        
        self.ax_spikes = axs[0]
        self.ax_membrane = axs[1]
        self.ax_adaptation = axs[2]
        self.ax_layers = axs[3:]
        
        self.ax_spikes.set_title("Spike Activity")
        self.ax_membrane.set_title("Membrane Potentials")
        self.ax_adaptation.set_title("Adaptation")
        
        for i, ax in enumerate(self.ax_layers):
            ax.set_title(f"Holographic Field Layer {i+1}")
        
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready - Step: 0")
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
    def update_coupling(self, value):
        self.system.neuron_grid.field_coupling = float(value)
        
    def update_noise(self, value):
        self.system.neuron_grid.noise_strength = float(value)
        
    def update_layer_coupling(self, value):
        self.system.layer_coupling = float(value)
        
    def update_baseline(self, value):
        self.system.neuron_grid.update_baseline(float(value))
        
    def toggle_simulation(self):
        self.running = not self.running
        
    def reset_system(self):
        self.system = LayeredSpikingHolographicSystem(grid_size=64, num_layers=self.num_layers)
        self.step_count = 0
        self.running = False
        
    def update_loop(self):
        if self.running:
            # Get filter shapes from EQs
            filter_shapes = [eq.get_filter_shape_tensor(num_points=64) for eq in self.eq_widgets]
            
            # Run simulation step
            spikes, field_states = self.system.step(custom_filter_shapes=filter_shapes)
            self.step_count += 1
            
            # Update visualizations
            self.update_plots(spikes, field_states)
            
        self.root.after(50, self.update_loop)  # ~20 FPS
        
    def update_plots(self, spikes, field_states):
        # Convert to numpy for plotting
        spikes_np = spikes.cpu().numpy()
        membrane_np = self.system.neuron_grid.v_membrane.cpu().numpy()
        adaptation_np = self.system.neuron_grid.adaptation.cpu().numpy()
        
        # Update spike plot
        self.ax_spikes.clear()
        self.ax_spikes.imshow(spikes_np, cmap='Reds', vmin=0, vmax=1)
        self.ax_spikes.set_title(f"Spike Activity (Count: {int(spikes_np.sum())})")
        
        # Update membrane potential plot
        self.ax_membrane.clear()
        self.ax_membrane.imshow(membrane_np, cmap='RdBu_r', vmin=-0.5, vmax=1.5)
        self.ax_membrane.set_title("Membrane Potentials")
        
        # Update adaptation plot
        self.ax_adaptation.clear()
        self.ax_adaptation.imshow(adaptation_np, cmap='Blues', vmin=0, vmax=np.percentile(adaptation_np, 95))
        self.ax_adaptation.set_title("Adaptation")
        
        # Update layer field plots
        for i, (ax, field_state) in enumerate(zip(self.ax_layers, field_states)):
            ax.clear()
            field_np = field_state.cpu().numpy()
            field_abs = np.abs(field_np)
            ax.imshow(field_abs, cmap='viridis', vmin=0, vmax=np.percentile(field_abs, 95))
            ax.set_title(f"Holographic Field Layer {i+1}")
        
        # Remove axes
        for ax in [self.ax_spikes, self.ax_membrane, self.ax_adaptation] + list(self.ax_layers):
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.canvas.draw()
        
        # Update status
        self.status_label.config(text=f"Running - Step: {self.step_count} - Spikes: {int(spikes_np.sum())}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LayeredSpikingHolographicGUI(root, num_layers=2)  # Change num_layers as needed
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()