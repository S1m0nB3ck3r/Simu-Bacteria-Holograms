#!/bin/bash
# Setup script for simu_holo environment
# This script creates symbolic links or copies the main scripts into the simu_holo directory

echo "Setting up simu_holo environment..."

# Check if we're in the simu_holo directory
if [ ! -f "README.md" ] || [ ! -d "configs" ]; then
    echo "Error: Please run this script from the simu_holo directory"
    exit 1
fi

# Get the parent directory (project root)
PARENT_DIR="$(cd .. && pwd)"

echo "Project root: $PARENT_DIR"
echo ""

# Option 1: Create symbolic links (Unix/Linux/macOS)
if command -v ln &> /dev/null; then
    echo "Creating symbolic links to main scripts..."
    ln -sf "$PARENT_DIR/main_simu_hologram.py" ./main_simu_hologram.py
    ln -sf "$PARENT_DIR/generate_config.py" ./generate_config.py
    ln -sf "$PARENT_DIR/simu_hologram.py" ./simu_hologram.py
    ln -sf "$PARENT_DIR/propagation.py" ./propagation.py
    ln -sf "$PARENT_DIR/traitement_holo.py" ./traitement_holo.py
    echo "✓ Symbolic links created"
fi

echo ""
echo "Setup complete!"
echo ""
echo "You can now use:"
echo "  python main_simu_hologram.py configs/config_bacteria_random.json"
echo ""
