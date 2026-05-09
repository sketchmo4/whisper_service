FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps: build tools + ffmpeg for audio conversion
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    ffmpeg \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp
WORKDIR /opt
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git
WORKDIR /opt/whisper.cpp
RUN cmake -S . -B build -DWHISPER_BUILD_TESTS=OFF \
 && cmake --build build -j

# Models directory
RUN mkdir -p /models

# Default: download base.en model at build time (can be overridden by mounting /models)
# Using upstream script
RUN bash ./models/download-ggml-model.sh base.en \
 && mv ./models/ggml-base.en.bin /models/ggml-base.en.bin

# App
WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt
COPY app /app

ENV DATA_DIR=/data
ENV WHISPER_BIN=/opt/whisper.cpp/build/bin/whisper-cli
ENV WHISPER_MODEL=/models/ggml-base.en.bin

EXPOSE 4060
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4060"]
