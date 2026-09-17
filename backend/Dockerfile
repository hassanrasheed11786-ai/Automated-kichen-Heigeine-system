# 1. Base Python image (Lightweight)
FROM python:3.10-slim

# 2. Working directory set karein
WORKDIR /app

# 3. OpenCV ke liye zaroori Linux system dependencies install karein
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Requirements file copy karein aur Python libraries install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. App ka baqi saara code aur model weights copy karein
COPY . .

# 6. FastAPI ka port expose karein
EXPOSE 8000

# 7. Server run karne ki final command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
