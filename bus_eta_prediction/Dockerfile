FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY predict_v2.py .
COPY utils.py .
COPY artifacts_v2/ ./artifacts_v2/

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Entrypoint
ENTRYPOINT ["python", "main.py"]
