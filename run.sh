#!/usr/bin/env bash
# run.sh - Main entry point for J.A.R.V.I.S. on Linux

# Change to the directory of the script
cd "$(dirname "$0")"

if [ ! -d ".venv-linux" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

source .venv-linux/bin/activate

# Optional: Set environment variables for Audio (PulseAudio/ALSA) if needed
# export PA_ALSA_PLUGHW=1

# Terminate any previously orphaned JARVIS process so ports and locks are immediately freed
CURRENT_PID=$$
for pid in $(pgrep -f "python.*core.main" 2>/dev/null); do
    if [ "$pid" != "$CURRENT_PID" ]; then
        echo "Cleaning up previous JARVIS instance (PID $pid)..."
        kill -15 $pid 2>/dev/null || true
        sleep 0.5
        kill -9 $pid 2>/dev/null || true
    fi
done

echo "Starting J.A.R.V.I.S..."
python -m core.main
EXIT_CODE=$?

# If Jarvis requested a reboot (exit code 42), the restart_jarvis.sh 
# script handles it. Here we just exit normally.
exit $EXIT_CODE
