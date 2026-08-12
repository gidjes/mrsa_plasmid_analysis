#!/bin/bash
set -e

echo "Creating all conda environments..."
for env_file in envs/*.yml; do
    env_name=$(basename "$env_file" .yml)
    echo "Setting up $env_name"
    conda env create -f "$env_file" || conda env update -f "$env_file"
done

echo "Installing Python CLI..."
pip install -e .

echo "Setup complete! You can now run the pipeline with: mge_bootstrap"