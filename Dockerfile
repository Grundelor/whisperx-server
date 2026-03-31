FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .
COPY onstart.sh .
RUN chmod +x onstart.sh

# Pre-download the podlodka-turbo model at build time (caches in image)
# This is the CTranslate2-converted version of bond005/whisper-podlodka-turbo
# optimized for Russian ASR with lower WER and built-in punctuation
RUN python -c "\
from faster_whisper import WhisperModel; \
print('Downloading bzikst/faster-whisper-podlodka-turbo...'); \
WhisperModel('bzikst/faster-whisper-podlodka-turbo', device='cpu', compute_type='int8')" || true

EXPOSE 5000

CMD ["python", "server.py"]
