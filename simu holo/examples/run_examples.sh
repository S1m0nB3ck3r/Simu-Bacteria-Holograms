#!/bin/bash
# Example usage of the unified hologram simulation with JSON configurations

echo "=========================================="
echo "Hologram Simulation Examples"
echo "=========================================="
echo ""

# Example 1: Quick test with small dataset
echo "Example 1: Quick test (small dataset)"
echo "----"
python generate_config.py bacteria_small config_test.json
python main_simu_hologram.py configs/config_test.json
echo ""

# Example 2: Standard bacteria simulation
echo "Example 2: Standard bacteria simulation"
echo "----"
python generate_config.py bacteria_medium config_bacteria_standard.json
# Optional: Edit config_bacteria_standard.json here
python main_simu_hologram.py configs/config_bacteria_standard.json
echo ""

# Example 3: High-resolution bacteria simulation
echo "Example 3: High-resolution bacteria"
echo "----"
python generate_config.py bacteria_large config_bacteria_hires.json
# Optional: Customize output directory
# sed -i 's/"output_dir": null/"output_dir": "\/path\/to\/output"/' config_bacteria_hires.json
python main_simu_hologram.py configs/config_bacteria_hires.json
echo ""

# Example 4: UV wavelength (405 nm) simulation
echo "Example 4: UV wavelength simulation"
echo "----"
python generate_config.py bacteria_uv config_bacteria_uv.json
python main_simu_hologram.py configs/config_bacteria_uv.json
echo ""

# Example 5: Sphere simulation
echo "Example 5: Sphere simulation"
echo "----"
python generate_config.py sphere_large config_sphere.json
python main_simu_hologram.py configs/config_sphere.json
echo ""

echo "=========================================="
echo "All examples complete!"
echo "=========================================="
