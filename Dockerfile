FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create app user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install CPU-only PyTorch first (much smaller than full torch)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py config.py ./
COPY templates/ templates/
COPY static/ static/
COPY models/ models/

# HF Spaces uses port 7860
EXPOSE 7860

# Run as non-root user
USER user

CMD ["python", "app.py"]
