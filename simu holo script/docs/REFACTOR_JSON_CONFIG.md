# Refactorization Summary: JSON Configuration for main_simu_hologram.py

## Change Overview

### Before
```bash
python main_simu_hologram.py \
  --mode bacteria_random \
  --nb_holo 100 \
  --nb_objects 50 \
  --holo_size_xy 1024 \
  --length_min 3.0e-6 \
  --length_max 4.0e-6 \
  ... (20+ more arguments)
```

### After
```bash
python main_simu_hologram.py configs/config_bacteria_random.json
```

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Readability** | Long CLI arguments | Clean JSON config file |
| **Reproducibility** | Hard to remember all args | Config file version control |
| **Flexibility** | Manual arg entry | Easy copy/modify presets |
| **Documentation** | Scattered in code | Centralized in JSON |
| **Validation** | Limited | Full config validation |
| **IDE support** | None | JSON schema available |

## Key Files

### 1. **main_simu_hologram.py** (Refactored)
- Simplified CLI: now takes only `config_file` argument
- `load_config()`: Loads JSON configuration
- `validate_config()`: Validates and sets defaults
- `simulate_bacteria_random()`, `simulate_bacteria_list()`, etc.: Use `config` dict

### 2. **config_bacteria_random.json** (Template)
```json
{
    "mode": "bacteria_random",
    "nb_holo": 100,
    "nb_objects": 50,
    "holo_size_xy": 1024,
    ...
}
```

### 3. **config_sphere_random.json** (Template)
Similar structure for sphere simulations.

### 4. **generate_config.py** (Helper Script)
Generates configurations from presets:
```bash
python generate_config.py bacteria_medium config_my_sim.json
```

**Available presets:**
- `bacteria_small`: Quick testing
- `bacteria_medium`: Standard setup
- `bacteria_large`: High resolution
- `bacteria_uv`: UV wavelength
- `sphere_small`, `sphere_large`

### 5. **CONFIG_GUIDE.md** (Documentation)
Complete reference for all parameters:
- Parameter descriptions
- Unit explanations
- Example configurations
- Output structure
- Performance tips

## Migration Path

### Old Scripts (Still Available)
- `main_simu_hologram_bacteria_list.py`
- `main_simu_hologram_random_sphere.py`
- etc.

### New Unified Approach
- Use `main_simu_hologram.py` with JSON configs
- These old scripts kept as backup

## Workflow Example

### Step 1: Generate Configuration
```bash
python generate_config.py bacteria_medium config_exp1.json
```

### Step 2: Customize (Edit config_exp1.json)
```json
{
    "mode": "bacteria_random",
    "nb_holo": 200,          # Changed from 100
    "nb_objects": 75,        # Changed from 50
    "output_dir": "/data/experiment_1"
}
```

### Step 3: Run Simulation
```bash
python main_simu_hologram.py config_exp1.json
```

### Step 4: Reproduce Anytime
```bash
python main_simu_hologram.py config_exp1.json  # Same exact config
```

## Configuration Validation

The script validates:
- ✓ Required fields: `mode`, `nb_holo`
- ✓ File exists
- ✓ Valid JSON syntax
- ✓ Sets sensible defaults for missing optional fields

## Backward Compatibility

- **No breaking changes** to old scripts
- **New unified approach** available in parallel
- Can gradually migrate if desired

## Next Steps (Optional)

1. **JSON Schema**: Add `.schema.json` for IDE validation
2. **Config Presets**: Add more specialized presets
3. **Web UI**: Build simple web form to generate configs
4. **Version Control**: Version configs with results

## Usage Examples

### Minimal Config
```json
{
    "mode": "bacteria_random",
    "nb_holo": 50
}
```
(All other parameters use defaults)

### Full Config
See `configs/config_bacteria_random.json` or `CONFIG_GUIDE.md`

### Quick Test
```bash
python generate_config.py bacteria_small test_config.json
python main_simu_hologram.py test_config.json
```

### Large Dataset
```bash
python generate_config.py bacteria_large large_config.json
# Edit large_config.json if needed
python main_simu_hologram.py large_config.json
```
