# Unified Hologram Simulation Configuration

## Quick Start

### 1. Edit Configuration File
Modify `configs/config_bacteria_random.json` or `configs/config_sphere_random.json` to match your needs:

```json
{
    "mode": "bacteria_random",
    "nb_holo": 100,
    "output_dir": null,
    "nb_objects": 50,
    "holo_size_xy": 1024,
    "z_size": 200,
    "length_min": 3.0e-6,
    "length_max": 4.0e-6,
    "thickness_min": 1.0e-6,
    "thickness_max": 2.0e-6,
    "wavelength": 660e-9,
    ...
}
```

### 2. Run Simulation
```bash
python main_simu_hologram.py configs/config_bacteria_random.json
```

## Configuration Parameters

### Mode
- `bacteria_random`: Random bacteria placement
- `bacteria_list`: Bacteria from predefined positions
- `sphere_random`: Random sphere placement
- `sphere_list`: Spheres from predefined positions

### Volume Parameters
| Parameter | Unit | Description |
|-----------|------|-------------|
| `nb_holo` | - | Number of holograms to generate |
| `holo_size_xy` | pixels | Hologram XY size (will be cropped to this) |
| `border` | pixels | Border padding for FFT (doubled on each side) |
| `upscale_factor` | - | Upscaling for bacteria insertion accuracy |
| `z_size` | planes | Number of Z propagation planes |
| `output_dir` | path | Output directory (null = current dir with timestamp) |

### Object Parameters (Bacteria)
| Parameter | Unit | Description |
|-----------|------|-------------|
| `nb_objects` | - | Number of bacteria per hologram |
| `length_min` | m | Minimum bacteria length |
| `length_max` | m | Maximum bacteria length |
| `thickness_min` | m | Minimum bacteria thickness |
| `thickness_max` | m | Maximum bacteria thickness |

### Object Parameters (Spheres)
| Parameter | Unit | Description |
|-----------|------|-------------|
| `radius_min` | m | Minimum sphere radius |
| `radius_max` | m | Maximum sphere radius |

### Optical Parameters
| Parameter | Unit | Description |
|-----------|------|-------------|
| `pix_size` | m | Camera pixel size |
| `magnification` | - | Microscope magnification |
| `index_medium` | - | Refractive index of medium (e.g., 1.33 for water) |
| `index_object` | - | Refractive index of object (e.g., 1.335 for bacteria) |
| `wavelength` | m | Illumination wavelength |
| `distance_volume_camera` | m | Distance from volume to camera |

### Illumination Parameters
| Parameter | Unit | Description |
|-----------|------|-------------|
| `illumination_mean` | - | Mean illumination level |
| `noise_std_min` | - | Minimum noise standard deviation |
| `noise_std_max` | - | Maximum noise standard deviation |

### Save Options
See [CONFIG_SAVE_OPTIONS.md](CONFIG_SAVE_OPTIONS.md) for details.

## Output Structure

```
simu_bacteria/
├── YYYY_MM_DD_HH_MM_SS/
    ├── positions/
    │   ├── bacteria_0.txt
    │   ├── bacteria_0.csv
    │   └── ...
    ├── holograms/
    │   ├── holo_0.bmp
    │   ├── holo_0.tiff (optional)
    │   ├── propagated_volume_0.tiff
    │   ├── segmentation_0.tiff
    │   └── ...
    └── data_holograms/
        ├── data_0.npz
        ├── data_1.npz
        └── ...
```

## Example Configurations

### High-Resolution Bacteria
```json
{
    "mode": "bacteria_random",
    "nb_holo": 500,
    "holo_size_xy": 2048,
    "z_size": 300,
    "nb_objects": 100,
    "length_min": 2.0e-6,
    "length_max": 5.0e-6,
    "pix_size": 5.5e-6,
    "magnification": 60.0,
    "wavelength": 405e-9
}
```

### Fast Testing (Small Dataset)
```json
{
    "mode": "bacteria_random",
    "nb_holo": 10,
    "holo_size_xy": 512,
    "z_size": 100,
    "nb_objects": 10
}
```

### Custom Output Directory
```json
{
    "mode": "bacteria_random",
    "nb_holo": 100,
    "output_dir": "/path/to/my/data/bacteria_exp_1"
}
```

## Performance Tips

1. **Reduce for testing**: Lower `nb_holo`, `nb_objects`, `holo_size_xy` for quick tests
2. **GPU memory**: `holo_size_xy` and `z_size` affect GPU memory usage
3. **Upscaling**: Higher `upscale_factor` improves bacteria insertion accuracy but is slower
4. **Border**: Larger `border` reduces FFT periodicity artifacts but uses more memory
5. **Save options**: Disabling optional formats saves disk space and time

## Notes

- All parameters are in SI units (meters, nanometers, etc.)
- Typical bacteria: length 2-5 µm, thickness 0.5-1 µm
- Typical water index: 1.33
- Typical UV wavelength: 405 nm, visible: 660 nm
- Default `distance_volume_camera`: 0.01 m (1 cm)
