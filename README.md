# Simu-Bacteria-Holograms

This code simulates a Gabor hologram obtained by illuminating a volume containing bacterias or spheres arranged within it. The script saves the simulated holograms along with the position of the bacteria and their angles in a tabulated text file.

This Python code uses the Cupy library and thus requires an NVIDIA graphics card for the computation of hologram propagation.

## 🚀 Getting Started

### 1. Generate Training Data for Neural Networks

To generate training data for future convolutional neural networks, run the simulation GUI:

```bash
python "simu bact GUI/simulation_gui.py"
```

This GUI allows you to:
- Configure simulation parameters (bacteria properties, hologram size, optical parameters)
- Set the number of holograms to generate
- Choose output formats (BMP, TIFF, NPY)
- Generate datasets with known ground truth for supervised learning

### 2. Visualize Generated Data

To explore and visualize the generated hologram datasets:

```bash
python "simu bact GUI/visualizer_gui.py" 
```

This visualization tool helps you:
- Browse generated holograms and their corresponding 3D volumes
- Inspect bacteria positions and orientations
- Validate the quality of training data

### 3. Test Classical Localization Pipeline (No AI)

To understand the principles of holographic reconstruction and backpropagation without AI:

```bash
python pipeline_holotracker_locate_simple.py
```

This educational pipeline demonstrates:
- **Hologram propagation**: Angular spectrum method for 3D reconstruction
- **Focus calculation**: Finding the optimal focus plane for each object
- **Object detection**: Classical thresholding and connected components
- **3D localization**: Extracting precise 3D coordinates

This classical approach serves as a baseline and helps understand the physical principles before implementing AI-based methods.

## 📁 Project Structure

- `simu bact GUI/`: Interactive tools for data generation and visualization
  - `simulation_gui.py`: Main simulation interface
  - `visualizer_gui.py`: Data exploration tool
  - `processor_simu_bact.py`: Background simulation engine
- `pipeline_holotracker_locate_simple.py`: Educational classical localization pipeline
- Legacy scripts: `simu_hologram_*.py` for direct script-based simulations

## 🔧 Requirements

- **GPU**: NVIDIA graphics card (CUDA-compatible)
- **Python libraries**: CuPy, NumPy, PIL, matplotlib, tkinter

