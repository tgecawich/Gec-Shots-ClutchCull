FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ML models into the image so cold starts don't re-download them
# (YOLOX person detector ~35MB, YuNet face detector ~230KB).
RUN mkdir -p models && \
    curl -sL -o models/object_detection_yolox_2022nov.onnx \
      "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx" && \
    curl -sL -o models/face_detection_yunet_2023mar.onnx \
      "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

COPY . ./

# Listens on $PORT (Render) with a 7860 fallback (Hugging Face).
EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
