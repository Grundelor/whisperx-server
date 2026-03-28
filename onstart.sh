#!/bin/bash
# Vast.ai onstart script — launches the transcription server
# This runs after Vast's SSH entrypoint initializes

set -e

echo "[onstart] Starting WhisperX transcription server..."

# Ensure HF_TOKEN is set (needed for pyannote)
if [ -z "$HF_TOKEN" ]; then
    echo "[onstart] WARNING: HF_TOKEN not set! Pyannote diarization won't work."
fi

# Start the FastAPI server in background
cd /app
nohup python server.py > /var/log/whisperx_server.log 2>&1 &
SERVER_PID=$!
echo "[onstart] Server started (PID=$SERVER_PID), listening on port 5000"

# Wait for server to be ready
for i in $(seq 1 150); do
    if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
        echo "[onstart] Server is healthy and ready!"
        exit 0
    fi
    echo "[onstart] Waiting for server... ($i/150)"
    sleep 2
done

echo "[onstart] ERROR: Server failed to start in 300 seconds"
cat /var/log/whisperx_server.log
exit 1
