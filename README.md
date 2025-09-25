# Frequency-Domain Holographic Video Processing

EDIT: 

The moire patterns - in the end of the day are born out of fft (fast fourier transform 
/ inverse fft): 

https://youtu.be/X88hy4pLSMU

I added the sigh_image.py file.. If you up the high frequencies, you will see that it will show 
64x64 grid when you just look at high frequencies as that is the finest pattern the fft can 
find from the 4096 (64x64 tensor that the systems use) tensor. No matter what the image. 
Meanwhile the large features are just the large features of the input webcam image and moire 
patterns between high and low, somehow represent the features between. 

EDIT: 

Added more realistic brain like system per claude opus - ping circuit 6

EDIT: 

Added the spiking holographic field system 4 per this video: 

https://youtu.be/Qcfq-01TKq4


Video where I explore these codes: (sorry I tend to take these down) 

[https://youtu.be/b9trcTw2EfMi](https://youtu.be/b9trcTw2EfM)

Inspired in part by this paper by Baker and Cariani: 

https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1540532/full

A collection of experimental applications exploring frequency-domain approaches to real-time video processing and pattern recognition, inspired by holographic field theories and neural oscillation research.

Applications

# 1. TheMatrix.py

An enhanced version of the Matrix8-EQ system with graphical frequency filtering capabilities. Combines Stable Diffusion with holographic field evolution for real-time video transformation.

Features:


Graphical 10-band equalizer for frequency domain control

Real-time webcam processing through holographic slow fields
Stable Diffusion integration for creative video effects
Temporal stabilization using moiré field phase detection

2. gated_afc.py (Enhanced AFC6)
An attentional field computer with frequency gating controls for both fast (sensory) and slow (conceptual) processing fields.
Features:

Dual-field architecture: fast sensory encoding and slow conceptual evolution
Independent frequency controls for low-pass (gist) and high-pass (detail) filtering
Text prompt integration for intentional guidance
Live "Mind's Eye" prediction display

# 3. isolate5.py

Motion detection and isolation system using frequency-domain chessboard pattern analysis.
Features:

Graphical value filtering with real-time pattern visualization
Chessboard structure detection and motion tracking
Memory-based motion accumulation with configurable decay
Multi-panel display showing filtered patterns, structure, and motion isolation

# 4. sigh.py

Simplified frequency-domain motion tracker focusing on natural background subtraction through spatial frequency analysis.
Features:

Real-time chessboard pattern generation from webcam input
Automatic moving object detection without explicit background subtraction
Frequency-based motion isolation
Live visualization of detected structures and motion patterns

# Core Concepts

These applications explore frequency-domain approaches to video processing, where spatial patterns are analyzed and manipulated in the Fourier domain. The systems use:

Holographic Fields: 2D spatial patterns that evolve through FFT-based filtering
Frequency Gating: Selective filtering of spatial frequencies to isolate different types of visual information
Phase Coherence: Maintaining temporal stability through phase relationship tracking
Multi-Scale Processing: Simultaneous analysis at different spatial frequency scales

# Dependencies

torch
torchvision
diffusers
transformers
opencv-python (cv2)
numpy
scipy
PIL (Pillow)
tkinter
matplotlib
For GPU acceleration:
CUDA-compatible PyTorch installation

# Usage

Run any of the applications directly:

bashpython TheMatrix.py

python gated_afc.py

python isolate5.py

python sigh.py

Each application will attempt to access your default webcam (index 0). Adjust camera settings in the code if needed.

# Controls

Graphical EQ (Matrix10_Fixed.py): Click and drag on the frequency bands to shape the filter response

Slider Controls: Real-time adjustment of frequency filtering parameters

Reset Buttons: Clear accumulated state and memory buffers

# Technical Notes

The systems operate in the frequency domain using 2D FFTs for spatial pattern analysis
Temporal stability is maintained through exponential moving averages and phase tracking
GPU acceleration is recommended for real-time performance
The holographic field approach allows for natural emergence of spatial patterns without explicit programming

Experimental Status
These are research prototypes exploring novel approaches to video processing through frequency-domain holographic fields. The techniques demonstrated may have applications in computer vision, motion detection, and creative video processing.
