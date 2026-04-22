# Use Python base image
FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import pathlib; import tensorflow as tf; p = pathlib.Path('/usr/local/lib/python3.11/site-packages/tensorflow/python/util/lazy_loader.py'); txt = p.read_text(encoding='utf-8', errors='ignore'); assert p.stat().st_size > 0 and 'KerasLazyLoader' in txt, 'Broken TensorFlow lazy_loader.py'; import typing_inspection.introspection as intro; assert hasattr(intro, 'Qualifier'), 'Broken typing_inspection'; print('Dependency sanity check passed')"

# Copy the model and code
COPY . .

# Expose port 8000 for the FastAPI web server
EXPOSE 8000

# Command to run the app
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000", "--workers", "1"]
