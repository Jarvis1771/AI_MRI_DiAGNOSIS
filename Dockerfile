FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create app user (HF Spaces requirement)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py config.py ./
COPY templates/ templates/
COPY models/ models/

# HF Spaces uses port 7860
EXPOSE 7860

# Run as non-root user
USER user

CMD ["python", "app.py"]
