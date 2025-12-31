#!/bin/bash
# Test script to run inside Docker container

set -e

echo "========================================="
echo "Starting Inference Pipeline"
echo "========================================="

# Check if we're inside container
if [ ! -f "/.dockerenv" ]; then
    echo "Error: This script should be run inside the Docker container"
    echo "Run: docker exec -it option-diffusion-trainer bash"
    echo "Then: ./scripts/test.sh"
    exit 1
fi

# Run inference
echo "Running inference..."
python -m option_diffusion.test "$@"

echo "========================================="
echo "Inference Complete!"
echo "Check results: cat results/predictions.csv | head -20"
echo "========================================="
