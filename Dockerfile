# Use NVIDIA's CUDA base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support and necessary libraries
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies like FastAPI, Pillow, and Diffusers
RUN pip3 install fastapi uvicorn Pillow diffusers transformers

# Pre-download and install the Flux.1 Schnell model during Docker build stage
# Clone the repo and install any additional dependencies
WORKDIR /models/flux
RUN git clone https://github.com/black-forest-labs/flux-model-repo.git . && \
    git checkout schnell-1.0 && \
    pip3 install -e .

# Copy the FastAPI app into the container
WORKDIR /app
COPY ./app /app

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]