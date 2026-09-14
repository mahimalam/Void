#!/usr/bin/env bash
# setup.sh - Environment bootstrapping for J.A.R.V.I.S. on Ubuntu/Linux

set -e

echo "=== J.A.R.V.I.S. Environment Setup (Linux) ==="

# Check for apt and install system dependencies
if command -v apt-get &> /dev/null; then
    echo "Installing required system packages (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y libportaudio2 ffmpeg xdotool xclip python3.11-venv python3.11-dev build-essential
else
    echo "Warning: apt-get not found. Please ensure portaudio, ffmpeg, xdotool, and xclip are installed."
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment in .venv..."
    python3.11 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate venv and install requirements
echo "Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! Please configure data/.env and then run ./run.sh"
