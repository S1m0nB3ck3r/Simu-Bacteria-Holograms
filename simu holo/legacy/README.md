# LEGACY SCRIPTS ARCHIVE

This directory contains the original simulation scripts before consolidation.

**Status**: These scripts are kept for historical reference only. Use [../main_simu_hologram.py](../main_simu_hologram.py) instead.

## Original Scripts

- `main_simu_hologram_bacteria_list.py` - Bacteria simulation from predefined list
- `main_simu_hologram_random_bact.py` - Bacteria simulation with random placement
- `main_simu_hologram_random_sphere.py` - Sphere simulation with random placement
- `main_simu_hologram_sphere_list.py` - Sphere simulation from predefined list

## Migration Guide

All functionality has been consolidated into a single unified script:

```bash
# OLD way (CLI arguments - no longer used)
python main_simu_hologram_random_bact.py

# NEW way (JSON configuration - recommended)
python main_simu_hologram.py ../configs/config_bacteria_random.json
```

## Benefits of New Approach

✅ Single unified script for all 4 modes  
✅ JSON configuration for reproducibility  
✅ Version control friendly  
✅ Better error handling and logging  
✅ Flexible save options  
✅ Improved documentation  

See [../README.md](../README.md) for detailed usage instructions.
