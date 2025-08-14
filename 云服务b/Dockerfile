
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages one by one to avoid conflicts
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic python-multipart
RUN pip install --no-cache-dir numpy opencv-python-headless Pillow
RUN pip install --no-cache-dir ultralytics


COPY cloudpose_service.py .


COPY model3-yolol/ ./model3-yolol/


EXPOSE 60000


RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app


HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:60000/health')"


CMD ["python", "cloudpose_service.py"]