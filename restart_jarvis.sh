#!/usr/bin/env bash
# restart_jarvis.sh - Handles graceful rebooting of the J.A.R.V.I.S. process

cd "$(dirname "$0")"

JARVIS_PID=$1

if [ -n "$JARVIS_PID" ] && ps -p $JARVIS_PID > /dev/null; then
    echo "Sending SIGTERM to JARVIS (PID $JARVIS_PID)..."
    kill -15 $JARVIS_PID
    
    # Wait up to 5 seconds for it to exit
    for i in {1..5}; do
        if ps -p $JARVIS_PID > /dev/null; then
            sleep 1
        else
            break
        fi
    done
    
    # Force kill if still alive
    if ps -p $JARVIS_PID > /dev/null; then
        echo "Process didn't exit, sending SIGKILL..."
        kill -9 $JARVIS_PID
    fi
fi

# Wait a brief moment to ensure ports are freed
sleep 1

# Launch fresh instance
echo "Relaunching J.A.R.V.I.S..."
nohup ./run.sh > /dev/null 2>&1 &

echo "Restart triggered successfully."
