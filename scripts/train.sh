#!/bin/bash
# Training script to run inside Docker container

set -e

echo "========================================="
echo "Starting Training Pipeline"
echo "========================================="

# Check if we're inside container
if [ ! -f "/.dockerenv" ]; then
    echo "Error: This script should be run inside the Docker container"
    echo "Run: docker exec -it option-diffusion-trainer bash"
    echo "Then: ./scripts/train.sh"
    exit 1
fi

# Ensure dependencies are installed
echo "Installing dependencies..."
poetry install

# Run training
echo "Running training..."
python -m option_diffusion.train "$@"

echo "========================================="
echo "Training Complete!"
echo "Check MLflow: http://localhost:5051"
echo "Check models: ls -lh models/"
echo "========================================="
